
import preprocessing
import solution_selection
import utils.visualization as visualization

"""
Plot all of the ARC-AGI problems in the split.
"""


if __name__ == "__main__":
    task_nums = list(range(400))
    split = input('Enter which split you want to find the task in (training, evaluation, test): ')
    tasks = preprocessing.preprocess_tasks(split, task_nums)
    train_history_loggers = []
    for task in tasks:
        train_history_logger = solution_selection.Logger(task)
        visualization.plot_problem(train_history_logger)
