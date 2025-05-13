 # The Four Color Theorem for Cell Instance Segmentation


 ## Dataset Preparation
First, download dataset from the following links:
BBBC006v1:https://bbbc.broadinstitute.org/BBBC006

 DSB2018:https://www.kaggle.com/competitions/data-science-bowl-2018/data

 PanNuke: https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/

 Yeaz: https://www.epfl.ch/labs/lpbs/data-and-software/

 Second, data preprocessing according to the file: processing.ipynb, which is saved in the projext folder: /mnt/data/ISAS.DE/ye.zhang/FCIS/preprocessing.ipynb

## Installation 
# 1. Create environment
conda create -n seine python=3.7 -y
conda activate seine

# 2. Install PyTorch (ensure CUDA 11.1 support)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 3. Install MMCV-full (Linux recommended)
pip install mmcv-full==1.3.13

# 4. Install required packages
pip install -r requirements.txt

# 5. Clone and install this repo
git clone https://github.com/zhangye-zoe/FCIS.git
cd FCIS
pip install -e .
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

 ## Thanks

This repo follow the design mode of [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) & [detectron2](https://github.com/facebookresearch/detectron2).
