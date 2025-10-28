import time

import numpy as np                  #type: ignore
import torch                        #type: ignore 
import torch.nn.functional as F     #type: ignore

import preprocessing
import arc_compressor
import solution_selection
import utils.visualization as visualization


"""
This file trains a model for every ARC-AGI task in a split.
"""

np.random.seed(0)
torch.manual_seed(0)


def vectorized_mask_select_logprobs(mask, length):
    """
    Vectorized version of mask_select_logprobs using 1D convolution.
    Calculates the unnormalized log probability of taking each slice.
    """
    if length == 0:
        # Handle empty slice case
        return torch.tensor(0.0, device=mask.device), torch.tensor([], device=mask.device)
    if length > mask.shape[0]:
        # Handle case where length is larger than mask
        return torch.tensor(0.0, device=mask.device), torch.tensor([], device=mask.device)

    # Use a 1D convolution to get sliding window sums
    # 1. Prepare mask: [L] -> [1, 1, L] (Batch, In_Channels, Length)
    mask_padded = mask.view(1, 1, -1)
    # 2. Prepare weight: [1, 1, K] (Out_Channels, In_Channels, Kernel_Size)
    weight = torch.ones(1, 1, length, device=mask.device, dtype=mask.dtype)
    
    # 3. Convolve
    # Output shape [1, 1, L - K + 1] -> [L - K + 1]
    window_sums = F.conv1d(mask_padded, weight, padding=0, stride=1).squeeze()

    # 4. Calculate logprobs
    # logprob = 2 * sum(inside) - sum(all)
    total_sum = torch.sum(mask)
    logprobs = 2.0 * window_sums - total_sum

    if logprobs.numel() == 0:
        # Handle edge case where L - K + 1 = 0
        return torch.tensor(0.0, device=mask.device), logprobs

    log_partition = torch.logsumexp(logprobs, dim=0)
    return log_partition, logprobs

# In train.py

def vectorized_reconstruction_error(logits_slice, target_crop, 
                                  x_logprobs, y_logprobs, 
                                  x_log_partition, y_log_partition, 
                                  output_shape, coefficient):
    """
    Calculates the reconstruction error for all windows at once.
    """
    X_out, Y_out = output_shape[0], output_shape[1]
    
    # 1. Get all logit windows (patches)
    # logits_slice shape: [C, N_x, N_y]
    # all_logits_patches shape: [C, N_x_offsets, N_y_offsets, X_out, Y_out]
    all_logits_patches = logits_slice.unfold(1, X_out, 1).unfold(2, Y_out, 1)
    
    # C_dim will be 3, N_x_off and N_y_off will be 15, X_out and Y_out will be 6
    C_dim, N_x_off, N_y_off, _, _ = all_logits_patches.shape
    N_batch = N_x_off * N_y_off

    # 2. Batch them for cross_entropy
    # Permute to: [N_x_offsets, N_y_offsets, C, X_out, Y_out]
    all_logits_permuted = all_logits_patches.permute(1, 2, 0, 3, 4)
    
    # --- FIX: Flatten offset dims into one batch dim ---
    # Reshape from [N_x_off, N_y_off, C, X_out, Y_out]
    # to [N_x_off * N_y_off, C, X_out, Y_out]
    all_logits_final_batch = all_logits_permuted.reshape(N_batch, C_dim, X_out, Y_out)
    
    # 3. Create a matching target batch
    # target_crop shape: [X_out, Y_out]
    # --- FIX: Expand target to match the new batch size N_batch ---
    # Shape: [N_batch, X_out, Y_out]
    target_batched = target_crop.expand(N_batch, X_out, Y_out)

    # 4. Calculate cross-entropy loss for ALL windows at once
    # This is now correct:
    #   input:  [N_batch, C, X_out, Y_out] (e.g., [225, 3, 6, 6])
    #   target: [N_batch, X_out, Y_out]   (e.g., [225, 6, 6])
    all_ce_losses = F.cross_entropy(all_logits_final_batch, target_batched, reduction='none')
    
    # all_ce_losses shape is [N_batch, X_out, Y_out]
    # Sum over spatial dims to get one loss per window
    # all_ce_losses_sum_flat shape: [N_batch]
    all_ce_losses_sum_flat = all_ce_losses.sum(dim=(1, 2))
    
    # --- FIX: Reshape the flat losses back into the 2D grid ---
    # Shape: [N_x_off, N_y_off]
    all_ce_losses_sum = all_ce_losses_sum_flat.view(N_x_off, N_y_off)

    # 5. Calculate logprobs for each window (This part was correct)
    # Use broadcasting to create an [N_x_offsets, N_y_offsets] grid
    xy_logprobs = x_logprobs.view(-1, 1) + y_logprobs.view(1, -1)
    
    # 6. Combine all log probabilities (This part was correct)
    total_logprobs = (
        xy_logprobs - 
        all_ce_losses_sum - # This now has the correct [N_x_off, N_y_off] shape
        x_log_partition - 
        y_log_partition
    )

    # 7. Final logsumexp (This part was correct)
    logprob = torch.logsumexp(coefficient * total_logprobs, dim=(0, 1)) / coefficient
    return -logprob  # Return the negative log-prob (the error)

def take_step(task, model, optimizer, train_step, train_history_logger):
    """
    Runs a forward pass of the model on the ARC-AGI task.
    Args:
        task (Task): The ARC-AGI task containing the problem.
        model (ArcCompressor): The VAE decoder model to run the forward pass with.
        optimizer (torch.optim.Optimizer): The optimizer used to take the step on the model weights.
        train_step (int): The training iteration number.
        train_history_logger (Logger): A logger object used for logging the forward pass outputs
                of the model, as well as accuracy and other things.
    """


    optimizer.zero_grad()
    logits, x_mask, y_mask, KL_amounts, KL_names, = model.forward()
    logits = torch.cat([torch.zeros_like(logits[:,:1,:,:]), logits], dim=1)  # add black color to logits

    # Compute the total KL loss
    total_KL = 0
    for KL_amount in KL_amounts:
        total_KL = total_KL + torch.sum(KL_amount)

    # Compute the reconstruction error
    reconstruction_error = 0
    for example_num in range(task.n_examples):  # sum over examples
        for in_out_mode in range(2):  # sum over in/out grid per example
            if example_num >= task.n_train and in_out_mode == 1:
                continue

            # Determine whether the grid size is already known.
            grid_size_uncertain = not (task.in_out_same_size or task.all_out_same_size and in_out_mode==1 or task.all_in_same_size and in_out_mode==0)
            if grid_size_uncertain:
                coefficient = 0.01**max(0, 1-train_step/100)
            else:
                coefficient = 1
                
            current_x_mask = x_mask[example_num,:,in_out_mode]
            current_y_mask = y_mask[example_num,:,in_out_mode]
            output_shape = task.shapes[example_num][in_out_mode]
            
            # Use NEW vectorized function
            x_log_partition, x_logprobs = vectorized_mask_select_logprobs(coefficient * current_x_mask, output_shape[0])
            y_log_partition, y_logprobs = vectorized_mask_select_logprobs(coefficient * current_y_mask, output_shape[1])

            # Account for probability of getting right grid size, if grid size is not known
            if grid_size_uncertain:
                x_log_partitions = []
                # This loop is still here, but it's small (runs ~30 times, not 30*30*30)
                # and uses the FAST function inside.
                for length in range(1, x_mask.shape[1]+1):
                    log_part, _ = vectorized_mask_select_logprobs(coefficient * current_x_mask, length)
                    x_log_partitions.append(log_part)
                
                y_log_partitions = []
                for length in range(1, y_mask.shape[1]+1):
                    log_part, _ = vectorized_mask_select_logprobs(coefficient * current_y_mask, length)
                    y_log_partitions.append(log_part)

                x_log_partition = torch.logsumexp(torch.stack(x_log_partitions, dim=0), dim=0)
                y_log_partition = torch.logsumexp(torch.stack(y_log_partitions, dim=0), dim=0)

            
            # --- THIS IS THE MAIN OPTIMIZATION ---
            # The nested Python loops are replaced by this single function call.
            
            logits_slice = logits[example_num,:,:,:,in_out_mode]  # color, x, y
            target_crop = task.problem[example_num, :output_shape[0], :output_shape[1], in_out_mode] # x, y

            if grid_size_uncertain:
                coefficient = 0.1**max(0, 1-train_step/100)
            else:
                coefficient = 1

            # Check if there are any valid windows to compute
            if x_logprobs.numel() > 0 and y_logprobs.numel() > 0:
                rec_err = vectorized_reconstruction_error(
                    logits_slice, target_crop,
                    x_logprobs, y_logprobs,
                    x_log_partition, y_log_partition,
                    output_shape, coefficient
                )
                reconstruction_error = reconstruction_error + rec_err
            # --- END OF OPTIMIZATION ---

    loss = total_KL + 10*reconstruction_error
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Performance recording
    train_history_logger.log(train_step,
                             logits,
                             x_mask,
                             y_mask,
                             KL_amounts,
                             KL_names,
                             total_KL,
                             reconstruction_error,
                             loss)


if __name__ == "__main__":
    start_time = time.time()

    task_nums = list(range(400))
    split = "training_small"  # "training", "training_small", "evaluation, or "test" 

    # Preprocess all tasks, make models, optimizers, and loggers. Make plots.
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    models = []
    optimizers = []
    train_history_loggers = []
    for task in tasks:
        model = arc_compressor.ARCCompressor(task)
        models.append(model)
        optimizer = torch.optim.Adam(model.weights_list, lr=0.01, betas=(0.5, 0.9))
        optimizers.append(optimizer)
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
        train_history_loggers.append(train_history_logger)

    # Get the solution hashes so that we can check for correctness
    true_solution_hashes = [task.solution_hash for task in tasks]

    # Train the models one by one
    for i, (task, model, optimizer, train_history_logger) in enumerate(zip(tasks, models, optimizers, train_history_loggers)):
        n_iterations = 2000
        for train_step in range(n_iterations):
            take_step(task, model, optimizer, train_step, train_history_logger)
        visualization.plot_solution(train_history_logger)
        solution_selection.save_predictions(train_history_loggers[:i+1])
        solution_selection.plot_accuracy(true_solution_hashes)

    # Write down how long it all took
    with open('timing_result.txt', 'w') as f:
        f.write("Time elapsed in seconds: " + str(time.time() - start_time))

