# The Four Color Theorem for Cell Instance Segmentation

[![GitHub Stars](https://img.shields.io/github/stars/zhangye-zoe/FCIS?style=social)](https://github.com/zhangye-zoe/FCIS/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/zhangye-zoe/FCIS?style=social)](https://github.com/zhangye-zoe/FCIS/network/members)
[![License](https://img.shields.io/github/license/zhangye-zoe/FCIS)](https://github.com/zhangye-zoe/FCIS/blob/main/LICENSE)


This is the official code repository for our paper accepted at ICML 2025:
**The Four Color Theorem for Cell Instance Segmentation**

* **Paper Link:** [Read the Paper](https://openreview.net/pdf?id=VK8SuRaJfX)
* **Publication:** International Conference on Machine Learning (ICML) 2025
* **Authors:** Ye Zhang, Yu Zhou, Yifeng Wang, Jun Xiao, Ziyue Wang, Yongbing Zhang, Jianxu Chen


## 🌟 Highlights

* Implementation of a novel approach applying graph coloring principles (inspired by the Four Color Theorem) for consistent cell instance visualization or downstream tasks.
* Integration with popular computer vision frameworks ([mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [Detectron2](https://github.com/facebookresearch/detectron2)) for robust instance segmentation and training pipelines.
* Support for processing various biomedical image datasets (BBBC006, DSB2018, PanNuke, Yeaz).

---

## 🚀 Installation

This project requires Python 3.7 and was developed with specific versions of PyTorch and MMCV.

**1. Create a Conda environment:**

```bash
conda create -n fcis python=3.7 -y
conda activate fcis
```
**2. Install PyTorch:**

Ensure you install the version compatible with your CUDA version. The original development used CUDA 11.1.
```bash
# For CUDA 11.1
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Install MMCV-full:**

Installing mmcv-full with CUDA support is crucial for leveraging GPU acceleration with the OpenMMLab frameworks.

```bash
# pip install mmcv-full==1.3.13
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9/index.html
```
**4. Install Required Packages:**

Install the remaining dependencies listed in the requirements.txt file.
```bash
pip install -r requirements.txt
```
**5. Clone and Install this repository:**
```bash
git clone https://github.com/zhangye-zoe/FCIS.git
cd FCIS
pip install -e .
```

## 📊 Dataset Preparation

This project utilizes data from the following publicly available datasets:

* **BBBC006v1:** [https://bbbc.broadinstitute.org/BBBC006](https://bbbc.broadinstitute.org/BBBC006)
* **DSB2018:** [https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
* **PanNuke:** [https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/)
* **Yeaz:** [https://www.epfl.ch/labs/lpbs/data-and-software/](https://www.epfl.ch/labs/lpbs/data-and-software/)

Preprocessing is required to convert raw datasets into the format expected by our training and evaluation pipelines.

For detailed instructions on how to download and preprocess the data, please refer to:

[**&#x1F4C4; Data Preparation Guide**](./docs/data_prepare.md)

This guide includes information on the required data structure and the scripts/notebooks located in the [`./preprocessing/`](./preprocessing/) directory.

## 💥 A frequently asked question

> Difference between `inst/` and `fcis_inst/`
>- **`inst/*.png`**: <span style="color:#1f77b4"><b>binary encoding</b></span> (`0`: background, `1`: foreground)
>- **`fcis_inst/*.png`**: <span style="color:#1f77b4"><b>four-color >encoding</b></span> (`0–4`),  
>- **`inst/*.npy`** and **`fcis_inst/*.npy`**: <span style="color:#1f77b4"><b>instance ID maps</b></span>  (`0–N`)

> **Note:** The `*.npy` files in both folders are identical;  
> the difference lies **only** in the `*.png` encodings.


## 🚀 Quick Start: Inference & Visualization

Download the pre-trained model weights from:
👉 [Hugging Face – FCIS Weights](https://huggingface.co/zhangye-zoe/FCIS-Weight/tree/main/checkpoints)

To quickly evaluate the trained model and visualize the prediction results, run the following command:

```bash
python tools/test.py \
configs/FCIS/fcis_bbbc.py \
./work_dirs/fcis_bbbc/fcis_bbbc.pth \
--show
--show-fold ./z_visual/BBBC
```

## 🧪 Training and Inference

```bash
python tools/train.py \
configs/FCIS/fcis_bbbc.py
```


```bash
python tools/test.py \
configs/FCIS/fcis_bbbc.py \
./work_dirs/fcis_bbbc/latest.pth \
```

## 📖 Citation

If you find our work useful, please cite our paper:
```bash
@inproceedings{zhang2025fourcolor,
  title     = {The Four Color Theorem for Cell Instance Segmentation},
  author    = {Zhang, Ye and Zhou, Yu and Wang, Yifeng and Xiao, Jun and Wang, Ziyue and Zhang, Yongbing and Chen, Jianxu},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {77194--77215},
  year      = {2025},
  publisher = {PMLR}
}

```