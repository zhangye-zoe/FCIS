import argparse
import os
import os.path as osp
import pickle

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction, get_logger

from tiseg.apis import multi_gpu_test, single_gpu_test
from tiseg.datasets import build_dataloader, build_dataset
from tiseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(description='test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--show', action='store_true', help='Whether to illustrate evaluation results.')
    parser.add_argument(
        '--show-folder', default='.nuclei_show', type=str, help='The storage folder of illustration results.')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        model_name = osp.dirname(args.config).replace('configs/', '')
        config_name = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f'./work_dirs/{model_name}/{config_name}'

    # create work dir
    eval_dir = osp.join(cfg.work_dir, 'eval')
    mmcv.mkdir_or_exist(osp.abspath(eval_dir))
    log_file = osp.join(eval_dir, 'eval.log')
    logger = get_logger(name='TorchImageSeg', log_file=log_file, log_level=cfg.log_level)

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not isinstance(cfg.data.test, list):
        cfg.data.test = [cfg.data.test]

    data_test_list = cfg.data.test
    for data_test in data_test_list:
        data_test.test_mode = True

        # build the dataloader
        # TODO: support multiple images per gpu (only minor changes are needed)
        dataset = build_dataset(data_test)
        data_loader = build_dataloader(
            dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)

        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE

        eval_kwargs = {} if args.eval_options is None else args.eval_options

        if args.show:
            eval_kwargs['show'] = True

        if args.show_folder is not None:
            eval_kwargs['show_folder'] = args.show_folder

            if not osp.exists(args.show_folder):
                os.makedirs(args.show_folder, 0o775)

        # clean gpu memory when starting a new evaluation.
        torch.cuda.empty_cache()

        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            results = single_gpu_test(model, data_loader, pre_eval=True, pre_eval_args=eval_kwargs)
        else:
            model = MMDistributedDataParallel(
                model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
            results = multi_gpu_test(model, data_loader, pre_eval=True, pre_eval_args=eval_kwargs)

        rank, _ = get_dist_info()
        if rank == 0:
            ckpt_name = osp.splitext(osp.basename(args.checkpoint))[0]
            eval_res, sotrage_res = dataset.evaluate(results, logger=logger, **eval_kwargs)
            pickle.dump(sotrage_res, open(osp.join(eval_dir, f'{ckpt_name}.p'), 'wb'))


if __name__ == '__main__':
    main()
