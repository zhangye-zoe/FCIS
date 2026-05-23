# FCIS Reproducibility Fixes

This patch addresses the PanNuke reproduction issues related to binary Dice,
empty-GT patches, fragmented predictions, and dataset split configuration.

## Main changes

1. **Binary foreground Dice**
   - Evaluation now computes `sem_pred` and `sem_gt` from instance maps:
     `sem_pred = inst_pred > 0`, `sem_gt = inst_gt > 0`.
   - The four-color prediction is kept as `fc_pred` for visualization/debugging,
     but is no longer used directly as a two-class semantic prediction.

2. **Empty-GT patch handling**
   - For image-wise AJI / DQ / SQ / PQ / InstDice, samples where both GT and
     prediction are empty are set to `NaN` and ignored by `nanmean`.
   - Binary foreground Dice treats empty-GT and empty-pred as correct (`Dice=1`).
   - Additional metrics are reported:
     - `imwEmptyCorrect`
     - `EmptyAcc`
     - `NumEmptyGT`
     - `NumEmptyPred`

3. **PanNuke split fix**
   - `configs/FCIS/pannuke.py` now uses:
     - train: `images/train`, `inst/train`, `train.txt`
     - val: `images/val`, `inst/val`, `val.txt`
     - test: `images/test`, `inst/test`, `test.txt`

4. **Foreground-normalized four-color loss**
   - `_cls_loss()` is normalized by the number of foreground pixels rather than
     by all image pixels, preventing weak supervision on sparse nuclei patches.

5. **Fragmentation reduction**
   - Added optional intra-instance consistency loss controlled by
     `train_cfg['intra_loss_weight']`.
   - PanNuke config enables it with `intra_loss_weight=0.1`.
   - Post-processing now supports `min_size` and `hole_size` in `test_cfg`.

## Important note

This patch changes the evaluation protocol and training loss. Results may differ
from previously released logs. For a fair comparison, rerun validation/testing
with the updated evaluation code.
