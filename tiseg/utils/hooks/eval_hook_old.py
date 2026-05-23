import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm
import os
import os.path as osp


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        eval_start (int, optional): The evaluate start iteration (When this arg
            is set to None, this arg will be invalidate.). Default: None
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self,
                 *args,
                 by_epoch=False,
                 eval_start=None,
                 epoch_iter=None,
                 max_iters=None,
                 last_epoch_num=None,
                 **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.eval_start = 0 if eval_start is None else eval_start
        self.epoch_iter = epoch_iter
        self.max_iters = max_iters
        self.last_epoch_num = last_epoch_num

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if runner.iter + 1 < self.eval_start:
            return False

        # if self.start is None:
        #     if not check_time(runner, self.interval):
        #         # No evaluation during the interval.
        #         return False
        # elif (current + 1) < self.start:
        #     # No evaluation if start is larger than the current time.
        #     return False
        # else:
        # Evaluation only at epochs/iters 3, 5, 7...
        # if start==3 and interval==2
        # print(self.last_epoch_num)
        # print(self.epoch_iter)
        if runner.iter + 1 + (self.last_epoch_num + 1) * self.epoch_iter < self.max_iters:
            if (current + 1 - self.eval_start) % self.interval:
                return False
        else:
            if (current + 1 - self.eval_start) % self.epoch_iter:
                return False
            else:
                ckpt_name = 'epoch' + str((current + 1 - self.eval_start) // self.epoch_iter) + '.pth'
                runner.save_checkpoint(runner.work_dir, ckpt_name, create_symlink=False)
        runner.logger.info(f'Now evaluation on {runner.iter + 1}.')
        return True

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from tiseg.apis import single_gpu_test
        results = single_gpu_test(runner.model, self.dataloader, pre_eval=True)

        runner.log_buffer.clear()
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def _save_ckpt(self, runner, key_score):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta['hook_msgs'].get('best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            runner.meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and osp.isfile(self.best_ckpt_path):
                os.remove(self.best_ckpt_path)

            best_ckpt_name = f'best_{self.key_indicator}_{current}.pth'
            self.best_ckpt_path = osp.join(runner.work_dir, best_ckpt_name)
            runner.meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner.save_checkpoint(runner.work_dir, best_ckpt_name, create_symlink=False)
            runner.logger.info(f'Now best checkpoint is saved as {best_ckpt_name}.')
            runner.logger.info(f'Best {self.key_indicator} is {best_score:0.4f} ' f'at {cur_time} {cur_type}.')


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, eval_start=None, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.eval_start = 0 if eval_start is None else eval_start

    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if runner.iter + 1 < self.eval_start:
            return False

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        from tiseg.apis import multi_gpu_test
        results = multi_gpu_test(runner.model, self.dataloader, pre_eval=True)

        runner.log_buffer.clear()

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)
