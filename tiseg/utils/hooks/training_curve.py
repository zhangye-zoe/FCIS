import json
import os.path as osp

import matplotlib.pyplot as plt
from mmcv.runner import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module()
class TrainingCurveHook(Hook):
    """Training loss & metric curve drawing in real-time.

    This hook only support iter-based runner now.

    Args:
        save_dir (str, optional): The training curve picture save folder path.
        interval (int): The interval iterations of hook call. Default: 50
        plot_keys (list[str]): Select the value to draw in log json.
            Default: ['loss']
        plot_groups (list[list[str]]): Split the plot value into several group
            to show in several sub pictures. Default: [['loss']]
        axis_groups (list[list]): Set the axis range of each sub picture.
            Default: [[0, 'max_iters', 0, 0]]
            ('max_iters' means use _max_iters attribute of runner)
        filename (str): The training curve save name.
            Default: training_curve.png
        num_rows (int): The number of picture rows. Default: 1
        num_cols (int): The number of picture cols. Default: 1
    """

    def __init__(self,
                 save_dir=None,
                 interval=50,
                 plot_keys=['loss'],
                 plot_groups=[['loss']],
                 axis_groups=[[0, 'max_iters', 0, 7]],
                 filename='training_curve.png',
                 num_rows=1,
                 num_cols=1):
        assert len(plot_groups) <= num_rows * num_cols
        self.save_dir = save_dir
        self.interval = interval
        self.plot_keys = plot_keys
        self.plot_groups = plot_groups
        self.axis_groups = axis_groups
        self.filename = filename

        self.figure = plt.figure(figsize=(14 * num_cols, 7 * num_rows))
        self.axes_list = self.figure.subplots(num_rows, num_cols)
        # flatten two-dimension axes to one-dimension axes
        self.axes_list = self.axes_list.flatten()

    @master_only
    def after_train_iter(self, runner):
        """The behavior after each train iteration.

        Args:
            runner (object): The runner.
        """
        if not self.every_n_iters(runner, self.interval):
            return

        if self.save_dir is None:
            self.save_dir = runner.work_dir

        prefix, suffix = osp.splitext(self.filename)
        fig_save_path = osp.join(self.save_dir,
                                 f'{prefix}_{runner.timestamp}{suffix}')

        # using timestamp to get log file path
        json_log_path = osp.join(runner.work_dir,
                                 f'{runner.timestamp}.log.json')

        iter_json_logs = [
            json.loads(x.strip())
            for x in open(json_log_path, 'r').readlines()
        ]

        # strip environment information
        _ = iter_json_logs[0]
        iter_json_logs = iter_json_logs[1:]

        iters = []
        collect_dict = {key: [] for key in self.plot_keys}
        for iter_json_log in iter_json_logs:
            if iter_json_log['mode'] != 'train':
                continue

            iters.append(iter_json_log['iter'])
            for plot_key in self.plot_keys:
                collect_dict[plot_key].append(iter_json_log[plot_key])

        max_iters = runner._max_iters

        # draw training curve
        for axes, plot_group, axis_group in zip(self.axes_list,
                                                self.plot_groups,
                                                self.axis_groups):
            for plot_key in plot_group:
                axes.plot(iters, collect_dict[plot_key], label=plot_key)
                axes.legend(loc='best')
                axes.grid(
                    color='black', linestyle='--', linewidth=1, alpha=0.3)
                if axis_group[1] == 'max_iters':
                    axis_group[1] = max_iters
                # refine axis
                left = axis_group[0] - (axis_group[1] - axis_group[0]) / 20
                right = axis_group[1] + (axis_group[1] - axis_group[0]) / 20
                bottom = axis_group[2] - (axis_group[3] - axis_group[2]) / 20
                top = axis_group[3] + (axis_group[3] - axis_group[2]) / 20
                axes.axis((left, right, bottom, top))
                axes.xaxis.set_major_locator(
                    plt.MultipleLocator(max_iters // 10))

        self.figure.tight_layout()
        self.figure.savefig(fig_save_path)

        # clean axes
        for axes in self.axes_list:
            axes.cla()
