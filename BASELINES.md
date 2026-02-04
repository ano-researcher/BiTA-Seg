# Baseline Models and Training Protocol

All baseline models were retrained on the Lung Field Segmentation dataset
using identical data splits, preprocessing, augmentation, and evaluation
protocols as BiTA-Seg.

# Baselines
- U-Net
- Attention U-Net
- DeepLabV3+
- TransUNet
- Swin-UNet
- nnU-Net

# Common Settings
- Input size: 256×256
- Optimizer: AdamW
- Learning rate: 1e-4
- Weight decay: 1e-4
- Epochs: 50
- Batch size: 4
- Loss: BCE + Dice
- Cross-validation: 5-fold on 6,129 images
- Random seeds: 42, 123, 2023

# Model-Specific Details
See `configs/baselines/*.yaml` for exact hyperparameters.

# Implementation Notes
- nnU-Net follows the official 2D configuration.
- TransUNet and Swin-UNet follow original paper settings with input
  resolution adjusted to 256×256.
- All models were trained from scratch.

# Evaluation
Metrics are averaged across folds and seeds and reported as mean ± std.
Latency is measured using batch size = 1, FP32, averaged over 100 runs.
