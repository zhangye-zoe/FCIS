# U-Net

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

## Results

> The metrics value is the average of the last five epochs.

### MoNuSeg (kumar)

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   | imwDice | imwAji | imwDQ | imwSQ | imwPQ | 
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  | :-:     | :--:   | :--:  | :--:  | :--:  | 
| UNet   | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 82.24 | 57.38 | 75.21 | 78.88 | 59.32 | 81.91   | 57.71  | 72.03 | 77.73 | 56.17 | 

### CoNIC

| Method | Crop Size | Batch Size | Slide Size | Learning Rate | mDice | mAji  | mDQ   | mSQ   | mPQ   |
| :--    | :--:      | :--        | :--:       | :--           | :--:  | :--:  | :--:  | :--:  | :--:  |
| UNet   | 256x256   | 8          | 40x40      | Adam-Lr1e-4   | 67.11 | 48.76 | 61.74 | 79.17 | 49.2 |
