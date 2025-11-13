# Land Cover Classification using Deep Learning

This project performs **land cover classification** using satellite imagery and deep learning with PyTorch. 
It uses segmentation masks to learn how to classify each pixel into land cover categories (e.g., water, urban, vegetation, etc.).

## Project Structure

```
LandCover/
│
├── LandCover.py               # Main training & inference script
├── data/
│   ├── train/                 # Training images and masks
│   │   ├── images/            # Satellite images (e.g., *_sat.jpg)
│   │   └── masks/             # Segmentation masks (e.g., *_mask.png)
│   ├── val/                   # Validation images and masks
│   └── test/                  # Test images (no masks needed)
└── README.txt                 # Project description
```

## Model Overview

The model is a **U-Net**-based architecture for semantic segmentation.
Each input image (e.g., satellite photo) is paired with a segmentation mask of the same size.
The model learns to predict the class of each pixel.

### Supported Classes
The dataset includes **7 land cover classes**, such as:
- Urban
- Agriculture
- Rangeland
- Forest
- Water
- Barren
- Unknown / Background

## 🧩 Requirements

Make sure you have the following Python libraries installed:
```bash
pip install torch torchvision albumentations numpy pillow tqdm
```

## How to Run

1. Place your dataset under `data/` as shown above.
2. Ensure that each image in `train/images/` has a corresponding mask in `train/masks/`.
3. Run training:
   ```bash
   python LandCover.py --mode train
   ```
4. Run inference (to predict on test images):
   ```bash
   python LandCover.py --mode test
   ```

## Common Issues

### 1. Mask not found error
Ensure each image has a corresponding mask file with the same base name, e.g.:
```
100694_sat.jpg  →  100694_mask.png
```

### 2. Size mismatch during training
Make sure both images and masks are resized to the same resolution (default: 256x256).

### 3. No training images found
Check that your folder structure matches the one shown above and file names are correct.

## Outputs

During training, you will see loss and IoU metrics printed per epoch.
After inference, the segmented mask predictions will be saved in the output folder.

