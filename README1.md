# Tissue-Image-Segmentation

unofficial re-implementation of popular tissue image segmentation models

Support Model:

- [x] UNet
- [x] Dist
- [x] DCAN
- [x] MicroNet
- [x] FullNet
- [x] CDNet

## Dataset Prepare

Please check [this doc](docs/data_prepare.md)

Supported Dataset:

- [x] MoNuSeg;
- [x] CoNSeP;
- [x] CPM17;
- [x] CoNIC;

## Installation

1. Install MMCV-full (Linux recommend): `pip install MMCV-full==1.3.13`;
2. Install requirements package: `pip install -r requirements.txt`;
3. Download tiseg: `git clone https://github.com/sennnnn/Torch-Image-Segmentation`;
4. Install tiseg: `pip install -e .`;

## Usage

### Training

```Bash
# single gpu training
python tools/train.py [config_path]
# multiple gpu training
./tools/dist_train.sh [config_path] [num_gpu]
# demo (cdnet for CPM17 dataset on 1 gpu)
python tools/train.py configs/unet/unet_vgg16_radam-lr5e-4_bs16_256x256_7k_cpm17.py
# demo (unet for CPM17 dataset on 4 gpu)
./tools/dist_train.py configs/unet/unet_vgg16_radam-lr5e-4_bs16_256x256_7k_cpm17.py 4
```

# Evaluation

```Bash
# single gpu evaluation
python tools/test.py [config_path]
# multiple gpu evaluation
./tools/dist_test.py [config_path] [num_gpu]
```

## Thanks

This repo follow the design mode of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) & [detectron2](https://github.com/facebookresearch/detectron2).
