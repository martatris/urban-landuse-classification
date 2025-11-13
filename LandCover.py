"""
DeepGlobe Multi-class Segmentation

This script prepares and trains a multi-class segmentation model (U-Net with a pretrained ResNet34 encoder
via segmentation_models_pytorch if available) on the DeepGlobe Land Cover Classification dataset.

Expectations
- Dataset folder structure (root passed via --data-root):
    <data_root>/
        class_dict.csv
        metadata.csv
        train/
            images/   (RGB .jpg/.png/.tif)
            masks/    (color-coded .png masks where colors map to classes)
        valid/
            images/
            masks/
        test/
            images/
            masks/

If masks are stored directly inside train/ without subfolders the script will try to auto-detect them.

Features
- Reads class color map from class_dict.csv and converts color masks to class index masks
- Dataset loader with albumentations augmentations
- Option to use a lightweight vanilla UNet or a pretrained encoder U-Net via segmentation_models_pytorch
- CrossEntropyLoss for multi-class segmentation, IoU metric per class
- Checkpointing, resume training, and visualization of predictions

Usage example:
    python train_deepglobe.py --data-root ./deepglobe --batch-size 8 --epochs 30 --use-smp

Notes:
- For best performance install segmentation_models_pytorch (smp) and pretrained weights. If not available the script falls back to a simple UNet implementation.
- Install required packages: pip install torch torchvision albumentations opencv-python matplotlib tqdm pandas Pillow scikit-learn
- If you use `--use-smp`, also install: pip install segmentation-models-pytorch==0.3.5
- Make sure data directory is correct.
"""

import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    raise ImportError("Please install albumentations and albumentations.pytorch: pip install albumentations")

from tqdm import tqdm

# Try optional dependency segmentation_models_pytorch
USE_SMP_AVAILABLE = False
try:
    import segmentation_models_pytorch as smp
    USE_SMP_AVAILABLE = True
except Exception:
    USE_SMP_AVAILABLE = False

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_class_dict(csv_path: Path) -> Tuple[Dict[Tuple[int,int,int], int], Dict[int, Tuple[int,int,int]], Dict[int,str]]:
    """
    Reads class_dict.csv assumed to have columns like: r,g,b,class_name,class_id
    Returns:
      - color2idx: {(r,g,b): idx}
      - idx2color: {idx: (r,g,b)}
      - idx2name: {idx: name}
    """
    df = pd.read_csv(csv_path)
    # try to locate columns
    # support several common formats
    cols = df.columns.str.lower()
    # find r,g,b
    if {'r','g','b'}.issubset(set(cols)):
        rcol = df.columns[[c.lower() for c in df.columns].index('r')]
        gcol = df.columns[[c.lower() for c in df.columns].index('g')]
        bcol = df.columns[[c.lower() for c in df.columns].index('b')]
    elif {'red','green','blue'}.issubset(set(cols)):
        rcol = df.columns[[c.lower() for c in df.columns].index('red')]
        gcol = df.columns[[c.lower() for c in df.columns].index('green')]
        bcol = df.columns[[c.lower() for c in df.columns].index('blue')]
    else:
        raise ValueError('class_dict.csv must contain r,g,b columns')

    # class name column
    name_col = None
    for candidate in ['class_name','name','label']:
        if candidate in [c.lower() for c in df.columns]:
            name_col = df.columns[[c.lower() for c in df.columns].index(candidate)]
            break
    # class id column
    id_col = None
    for candidate in ['class_id','id','label_id','index']:
        if candidate in [c.lower() for c in df.columns]:
            id_col = df.columns[[c.lower() for c in df.columns].index(candidate)]
            break

    color2idx = {}
    idx2color = {}
    idx2name = {}
    for _, row in df.iterrows():
        r = int(row[rcol]); g = int(row[gcol]); b = int(row[bcol])
        if id_col is not None:
            idx = int(row[id_col])
        else:
            # assign incremental index if not provided
            idx = len(idx2color)
        color2idx[(r,g,b)] = idx
        idx2color[idx] = (r,g,b)
        if name_col is not None:
            idx2name[idx] = str(row[name_col])
        else:
            idx2name[idx] = str(idx)
    return color2idx, idx2color, idx2name


def mask_color_to_index(mask: np.ndarray, color2idx: Dict[Tuple[int,int,int], int], default: int = 0) -> np.ndarray:
    h,w,_ = mask.shape
    out = np.zeros((h,w), dtype=np.int64)
    # build lookup for fast mapping (flatten colors)
    # convert to single integer key
    key_map = { (r<<16) + (g<<8) + b: idx for (r,g,b), idx in color2idx.items() }
    flat = mask.reshape(-1,3)
    keys = (flat[:,0].astype(np.int32) << 16) + (flat[:,1].astype(np.int32) << 8) + flat[:,2].astype(np.int32)
    out_flat = np.full(keys.shape, fill_value=default, dtype=np.int64)
    for k, idx in key_map.items():
        matches = (keys == k)
        out_flat[matches] = idx
    out = out_flat.reshape(h,w)
    return out

# -----------------------------
# Dataset
# -----------------------------
class DeepGlobeDataset(Dataset):
    def __init__(self, img_paths, mask_paths, color2idx, image_size=(256, 256), augment=False):
        if len(mask_paths) > 0:
            assert len(img_paths) == len(mask_paths), 'images and masks length mismatch'
        else:
            print(f"[INFO] No masks found — running in inference mode with {len(img_paths)} images.")

        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.color2idx = color2idx
        self.augment = augment
        self.image_size = image_size

        if self.augment:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = np.array(Image.open(img_path).convert('RGB'))

        if len(self.mask_paths) > 0:
            mask_path = self.mask_paths[idx]
            mask_color = np.array(Image.open(mask_path).convert('RGB'))
            mask_idx = mask_color_to_index(mask_color, self.color2idx)
            augmented = self.transform(image=img, mask=mask_idx)
            image = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = mask.long()
        else:
            augmented = self.transform(image=img)
            image = augmented['image']
            mask = torch.zeros((self.image_size[0], self.image_size[1]), dtype=torch.long)

        return image, mask

# -----------------------------
# Models
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetSimple(nn.Module):
    def __init__(self, in_channels=3, out_channels=7, features=[64,128,256,512]):
        super(UNetSimple, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feat in features:
            self.downs.append(DoubleConv(in_channels, feat))
            in_channels = feat
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev_features = features[::-1]
        for feat in rev_features:
            self.ups.append(nn.ConvTranspose2d(feat*2, feat, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feat*2, feat))
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)

# -----------------------------
# Metrics & Loss
# -----------------------------

def compute_iou_per_class(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, eps=1e-6):
    # preds: BxCxHxW logits
    preds = torch.argmax(preds, dim=1)  # BxHxW
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection + eps) / (union + eps)
        ious.append(iou)
    return ious

# -----------------------------
# Helpers to gather file lists
# -----------------------------

def gather_image_mask_pairs(folder: Path) -> Tuple[List[Path], List[Path]]:
    imgs = sorted([p for p in folder.glob('*_sat.*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    msks = []
    for img_path in imgs:
        mask_candidate = folder / (img_path.stem.replace('_sat', '_mask') + '.png')
        if mask_candidate.exists():
            msks.append(mask_candidate)
        else:
            print(f"[WARN] No mask found for {img_path.name}")
    print(f"[INFO] Found {len(imgs)} images and {len(msks)} masks in {folder}")
    return imgs, msks

# -----------------------------
# Training / Validation loops
# -----------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        pbar.set_description(f"loss: {loss.item():.4f}")
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    iou_list = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)
            ious = compute_iou_per_class(logits, masks, num_classes)
            iou_list.append(ious)
    # average
    mean_iou = np.nanmean(np.array(iou_list), axis=0)
    return running_loss / len(loader.dataset), mean_iou

# -----------------------------
# Visualization
# -----------------------------

def decode_segmap(mask: np.ndarray, idx2color: Dict[int, Tuple[int,int,int]]) -> np.ndarray:
    h,w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for idx, color in idx2color.items():
        out[mask==idx] = color
    return out

def visualize_predictions(model, dataset, idx2color, device, n=4, save_path=None):
    model.eval()
    plt.figure(figsize=(12, 4*n))
    for i in range(n):
        img, mask = dataset[i]
        inp = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            pred = torch.argmax(logits, dim=1).cpu().squeeze().numpy()
        img_np = np.transpose(img.cpu().numpy(), (1,2,0))
        gt_color = decode_segmap(mask.numpy(), idx2color)
        pred_color = decode_segmap(pred, idx2color)

        ax = plt.subplot(n, 3, i*3 + 1)
        ax.imshow(img_np)
        ax.set_title('Image'); ax.axis('off')
        ax = plt.subplot(n, 3, i*3 + 2)
        ax.imshow(gt_color); ax.set_title('Ground Truth'); ax.axis('off')
        ax = plt.subplot(n, 3, i*3 + 3)
        ax.imshow(pred_color); ax.set_title('Prediction'); ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        type=str,
        default="/Users/tristonmarta/Desktop/Land Cover/data", # dataset change to one own path
        help='root folder of DeepGlobe dataset'
    )
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-smp', action='store_true', help='use segmentation_models_pytorch pretrained encoder')
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    class_csv = data_root / 'class_dict.csv'
    if not class_csv.exists():
        raise FileNotFoundError('class_dict.csv not found in data root')
    color2idx, idx2color, idx2name = read_class_dict(class_csv)
    num_classes = len(idx2color)
    print(f'Found {num_classes} classes')

    train_folder = data_root / 'train'
    valid_folder = data_root / 'valid'
    test_folder = data_root / 'test'

    train_imgs, train_masks = gather_image_mask_pairs(train_folder)
    print(f"Found {len(train_imgs)} training images and {len(train_masks)} masks")
    val_imgs, val_masks = gather_image_mask_pairs(valid_folder)
    test_imgs = sorted([p for p in test_folder.glob('*_sat.*')])
    test_masks = []

    print(f'Train pairs: {len(train_imgs)} | Val pairs: {len(val_imgs)} | Test pairs: {len(test_imgs)}')

    train_ds = DeepGlobeDataset(train_imgs, train_masks, color2idx, image_size=(args.image_size, args.image_size), augment=True)
    val_ds = DeepGlobeDataset(val_imgs, val_masks, color2idx, image_size=(args.image_size, args.image_size), augment=False)
    test_ds = DeepGlobeDataset(test_imgs, test_masks, color2idx, image_size=(args.image_size, args.image_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model choice
    if args.use_smp:
        if not USE_SMP_AVAILABLE:
            raise ImportError('segmentation_models_pytorch not installed. Install with pip install segmentation-models-pytorch')
        model = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=num_classes)
    else:
        model = UNetSimple(in_channels=3, out_channels=num_classes)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_mIoU = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'===== Epoch {epoch}/{args.epochs} =====')
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device, num_classes)
        mean_iou = np.nanmean(val_iou)
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {mean_iou:.4f}')
        for i, name in idx2name.items():
            print(f"  class {i} ({name}): IoU={val_iou[i]:.4f}")

        # checkpoint
        ckpt_path = save_dir / f'model_epoch{epoch}.pth'
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)
        if mean_iou > best_mIoU:
            best_mIoU = mean_iou
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f'New best model saved (mIoU={best_mIoU:.4f})')

    # load best and evaluate on test set
    model.load_state_dict(torch.load(save_dir / 'best_model.pth', map_location=device))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loss, test_iou = validate(model, test_loader, criterion, device, num_classes)
    print(f'Test Loss: {test_loss:.4f} | Test mIoU: {np.nanmean(test_iou):.4f}')
    for i, name in idx2name.items():
        print(f"  class {i} ({name}): IoU={test_iou[i]:.4f}")

    # visualize some predictions
    visualize_predictions(model, test_ds, idx2color, device, n=4, save_path=str(save_dir / 'predictions.png'))

if __name__ == '__main__':
    main()
