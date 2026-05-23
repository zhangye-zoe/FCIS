# Detailed Dataset Preparation Guide

This document provides comprehensive instructions on how to download, preprocess, and organize the datasets used in the **The Four Color Theorem for Cell Instance Segmentatio** project. Following these steps is crucial to ensure the data is in the correct format and location for training, evaluation, and inference using the project's code.

## 📚 Datasets

The project utilizes data from the following publicly available datasets:

* **BBBC006v1:** [https://bbbc.broadinstitute.org/BBBC006](https://bbbc.broadinstitute.org/BBBC006)
* **DSB2018:** [https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)
* **PanNuke:** [https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke/)
* **Yeaz:** [https://www.epfl.ch/labs/lpbs/data-and-software/](https://www.epfl.ch/labs/lpbs/data-and-software/)

Please download the raw data files for the datasets you intend to use. Save them in a location of your choice (e.g., a `raw_data/` directory outside this repository).

## ✨ Preprocessing Steps

The raw datasets need to be preprocessed into a unified format that includes images, segmentation masks, instance masks, adjacency lists, and four-colored instance maps.

The core preprocessing scripts and notebooks are located in the [`./preprocessing/`](../preprocessing/) directory relative to this file.

The preprocessing workflow typically involves the following steps:

1.  **Extracting and Converting Basic Formats:** Convert raw image files and their corresponding segmentation/instance masks into standard formats like PNG or NPY arrays, if necessary.
2.  **Generating Binary Masks:** Create binary segmentation masks (foreground/background) from the instance masks.
3.  **Generating Adjacency Lists:** For each instance mask, compute the adjacency relationships between individual cell instances and save them as a list of edges (e.g., `u,v` pairs).
4.  **Applying Four-Coloring:** Apply the graph coloring algorithm (based on the generated adjacency lists) to the instance masks to assign a color index (0-3) to each instance. Save the resulting color index maps and their visualizations.

**Detailed Instructions:**

Please follow the instructions and run the code provided in the [`./preprocessing/processing.ipynb`](../preprocessing/processing.ipynb) notebook for a step-by-step guide and examples.

*(Optional: If you have separate scripts for different parts of preprocessing, mention them here, e.g., "Alternatively, you can use the script [`./preprocessing/generate_masks_adj_colors.py`](../preprocessing/generate_masks_adj_colors.py) directly after obtaining basic image and instance mask files.")*


## 📂 Expected Data Structure

After successfully running the preprocessing steps, your `./data/` directory should be organized as follows. This structure is essential because the project's configuration files (`./configs/`) are set up to read data from these specific locations.

It is highly recommended to either place your processed data directly in `FCIS/data/` or create a [symlink](https://en.wikipedia.org/wiki/Symbolic_link) from `FCIS/data/` to the location where you store your processed data.

Example using symlinking (Linux/macOS):

```bash
# Assuming your processed DSB2018 data is in /path/to/your/processed_data/DSB2018
# Navigate to the root of the FCIS repository
cd /path/to/FCIS_repo
mkdir data # Create the data directory if it doesn't exist
ln -s /path/to/your/processed_data/DSB2018 ./data/DSB2018
```

The expected structure within ./data/DSB2018/ is:

```
./data/
└── [DATASET_NAME]/ # e.g., DSB2018
    ├── images/
    │   ├── train/         # Original or preprocessed images (e.g., .png, .jpg)
    │   ├── val/
    │   └── test/
    ├── masks/
    │   ├── train/          # Binary segmentation masks (e.g., .png)
    │   ├── val/
    │   └── test/
    ├── insts/
    │   ├── train/          # Original instance masks (e.g., .npy or .png)
    │   ├── val/
    │   └── test/
    ├── fcis_insts_color_idx/
    │   ├── train/ # Four-colored instance index maps (e.g., .png)
    │   ├── val/
    │   └── test/
    ├── fcis_insts_visual/
    │   ├── train/ # Visualizations of four-colored instances (e.g., .png)
    │   ├── val/
    │   └── test/
    └── fcis_insts_adj_list/
        ├── train/ # Adjacency lists (e.g., .txt with u,v per line)
        ├── val/
        └── test/
```

## ❓ Troubleshooting

If you encounter issues during dataset preparation or when using the processed data for training/testing, consider the following:

* If you encounter `FileNotFoundError`, double-check that your data structure exactly matches the "Expected Data Structure" described above and that your symlink (if used) is correctly pointing to the data location.
* Ensure the preprocessing scripts completed without errors for the dataset and split you are trying to use.
* Verify that the file paths specified in your configuration files (`./configs/`) correctly point to the locations within your `./data/` directory where the processed data is stored.
* Check file permissions to ensure the script has read access to the data files and write access to the output directories.