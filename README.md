# Land Cover Classification using Deep Learning

This project performs land cover classification using satellite imagery and deep learning techniques implemented in PyTorch. The goal is to automatically analyze aerial or satellite images and classify each pixel into meaningful land cover categories such as water, urban areas, vegetation, barren land, roads, and others.

By leveraging semantic segmentation, the model learns to associate visual patterns in high-resolution satellite imagery with specific land use types. During training, it uses segmentation masks—ground truth images where each pixel is labeled according to its class—to learn spatial and contextual relationships across the landscape.

This approach enables detailed mapping of urban and environmental regions, which is valuable for urban planning, environmental monitoring, and geospatial analysis. The project uses the DeepGlobe Land Cover Classification dataset, a benchmark dataset designed for large-scale land cover segmentation tasks.

Ultimately, the project demonstrates how deep learning can be applied to spatial data to produce high-quality, automated land cover maps from satellite imagery.

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

The dataset is downloadable from Kaggle at: https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset

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

