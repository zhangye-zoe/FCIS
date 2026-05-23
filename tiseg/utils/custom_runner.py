from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS


@RUNNERS.register_module()
class CustomRunner(EpochBasedRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)

            visual = outputs['extra_vars'].pop('visual')
            img = visual['img']
            sem = visual['sem_gt']
            inst = visual['inst_gt']
            dir = visual['dir_gt']
            point = visual['point_gt']
            tc = visual['tc_gt']
            dir_loss = visual['full_dir_ce_loss']
            dir_loss_w = visual['full_dir_ce_loss_dirw']
            weight_map = visual['weight_map']
            metas = visual['metas']

            import numpy as np

            tmp = None
            for idx, items in enumerate(zip(img, sem, inst, dir, point, tc, dir_loss, dir_loss_w, weight_map)):
                collect = []
                for item in items:
                    if len(item.shape) == 2:
                        item = np.expand_dims(item, axis=-1)
                    collect.append(item)
                tmp = np.concatenate(collect, axis=-1)
                np.save(f'temp/{self.epoch}_{metas[idx]["data_id"]}.npy', tmp)
                break

            import matplotlib.pyplot as plt
            plt.figure(dpi=100)
            plt.subplot(331)
            plt.imshow(tmp[:, :, :3].astype(np.uint8))
            plt.subplot(332)
            plt.imshow(tmp[:, :, 3])
            plt.subplot(333)
            plt.imshow(tmp[:, :, 4])
            plt.subplot(334)
            plt.imshow(tmp[:, :, 5])
            plt.subplot(335)
            plt.imshow(tmp[:, :, 6])
            plt.subplot(336)
            plt.imshow(tmp[:, :, 7])
            plt.subplot(337)
            plt.imshow(tmp[:, :, 8])
            plt.subplot(338)
            plt.imshow(tmp[:, :, 9])
            plt.subplot(339)
            plt.imshow(tmp[:, :, 10])

            plt.savefig(f'temp/{self.epoch}_{metas[0]["data_id"]}.png')
            plt.close()
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"' 'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
