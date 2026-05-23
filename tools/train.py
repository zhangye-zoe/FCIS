import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash, collect_env, get_logger, print_log

from tiseg import __version__
from tiseg.apis import train_segmentor, set_random_seed, init_random_seed
from tiseg.datasets import build_dataset
from tiseg.models import build_segmentor
from tiseg.models.utils import revert_sync_batchnorm
from log_analysis import log_analysis


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    # Whether to evaluation when training
    parser.add_argument(
        '--no-validate', action='store_true', help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus', type=int, help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids', type=int, nargs='+', help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    # Set pytorch initial seed and cudnn op selection
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument(
        '--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    # Manual set some config option
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    # Set runtime launcher. If launcher is not None, we will use MMDDP；
    # If None, we will use MMDP. MMDDP & MMDP can compat DP & DDP.
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        model_name = osp.dirname(args.config).replace('configs/', '')
        config_name = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = f'./work_dirs/{model_name}/{config_name}'

    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # create work dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger(name='TorchImageSeg', log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    # pretty_text is actually the fancy display of config options.
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, ' f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # Build top level model
    model = build_segmentor(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if not distributed:
        warnings.warn('SyncBN only support DDP. In order to compat with DP, we convert '
                      'SyncBN tp BN. Please to use dist_train.py which has official '
                      'support to avoid this problem.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    # Workflow is deprecated now.
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save tiseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(tiseg_version=f'{__version__}+{get_git_hash()[:7]}', config=cfg.pretty_text)
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model, datasets, cfg, distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
    # after train results collection
    model_name = osp.dirname(args.config).replace('configs/', '')
    config_name = osp.splitext(osp.basename(args.config))[0]
    # calculate metrics
    print_log('\n' + log_analysis(cfg.work_dir), logger=logger)


if __name__ == '__main__':
    main()
