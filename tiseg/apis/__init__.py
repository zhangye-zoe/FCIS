from .test import multi_gpu_test, single_gpu_test
from .train import set_random_seed, train_segmentor, init_random_seed

__all__ = ['init_random_seed', 'set_random_seed', 'train_segmentor', 'single_gpu_test', 'multi_gpu_test']
