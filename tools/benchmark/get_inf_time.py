import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

from tiseg.datasets import build_dataloader, build_dataset
from tiseg.models import build_segmentor


def test_inf_time(data_loader, model, log_interval):
    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = 10

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(f'Overall fps: {fps:.2f} img / s')
            break


def parse_args():
    parser = argparse.ArgumentParser(description='Test model inference time.')
    parser.add_argument('config', help='test config file path.')
    parser.add_argument(
        '--log-interval', type=int, default=50, help='interval of logging.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    # cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # model prepare
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    # deploy model
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    test_inf_time(data_loader, model, args.log_interval)


if __name__ == '__main__':
    main()
