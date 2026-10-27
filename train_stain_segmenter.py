#!/usr/bin/env python3
"""
train_stain_segmenter.py - Stage B: Stain Segmentation and Severity Modeling.
Trains U-Net / DeepLabV3+ with Dice + Focal Loss, Cosine Annealing, 
Two-Phase (Frozen Encoder -> Unfrozen) Schedule, and Albumentations.
"""

import os
import sys
import copy
import yaml
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SolarStainDataset(Dataset):
    def __init__(self, df, transform=None, img_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        mask_path = row["mask_path"]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if pd.notna(mask_path) and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Ensure fixed size
        if image.shape[:2] != self.img_size:
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
            mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1).float()
            mask = torch.tensor(mask).unsqueeze(0).float()

        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "path": img_path,
            "original_label": row.get("original_label", "clean"),
            "coverage_pct": float(row.get("coverage_pct", 0.0)),
            "high_stain": int(row.get("high_stain", 0)),
            "source": row.get("source", "unknown"),
            "lighting": row.get("lighting", "sunny"),
            "distance": row.get("distance", "medium")
        }

def get_train_transforms(img_size=(512, 512)):
    """
    Albumentations pipeline: flips, rotations, shadows, glare, blur, JPEG compression,
    random crop/scale. CRITICAL: Strictly avoid hue shifts that change dust appearance.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.ImageCompression(quality_range=(75, 95), p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal loss for segmentation with severe class imbalance (small stains).
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Dice Loss
        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score

        # Focal Loss
        bce = nn.functional.binary_cross_entropy_with_logits(logits.view(-1), targets_flat, reduction='none')
        p_t = torch.exp(-bce)
        focal_loss = ((1.0 - p_t) ** self.gamma * bce).mean()

        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

def compute_iou_and_dice(preds_bin, targets_bin, smooth=1e-6):
    """
    Computes Intersection over Union (IoU) and Dice coefficient for binary masks.
    """
    intersection = np.logical_and(preds_bin, targets_bin).sum()
    union = np.logical_or(preds_bin, targets_bin).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (preds_bin.sum() + targets_bin.sum() + smooth)
    return float(iou), float(dice)

def compute_boundary_f1(preds_bin, targets_bin, theta=3):
    """
    Computes Boundary F1 score (contour alignment accuracy).
    """
    preds_u = (preds_bin * 255).astype(np.uint8)
    targets_u = (targets_bin * 255).astype(np.uint8)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (theta, theta))
    pred_boundary = cv2.morphologyEx(preds_u, cv2.MORPH_GRADIENT, k) > 0
    target_boundary = cv2.morphologyEx(targets_u, cv2.MORPH_GRADIENT, k) > 0

    if target_boundary.sum() == 0 and pred_boundary.sum() == 0:
        return 1.0
    if target_boundary.sum() == 0 or pred_boundary.sum() == 0:
        return 0.0

    inter = np.logical_and(pred_boundary, target_boundary).sum()
    precision = inter / (pred_boundary.sum() + 1e-6)
    recall = inter / (target_boundary.sum() + 1e-6)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def predict_with_tta(model, tensor_img, device="cpu"):
    """
    Test-time augmentation: evaluates original, hflip, and vflip, then averages.
    """
    model.eval()
    with torch.no_grad():
        x = tensor_img.to(device)
        logits1 = model(x)
        p1 = torch.sigmoid(logits1)

        # Horizontal flip
        x_h = torch.flip(x, dims=[3])
        logits2 = model(x_h)
        p2 = torch.flip(torch.sigmoid(logits2), dims=[3])

        # Vertical flip
        x_v = torch.flip(x, dims=[2])
        logits3 = model(x_v)
        p3 = torch.flip(torch.sigmoid(logits3), dims=[2])

        p_tta = (p1 + p2 + p3) / 3.0
    return p_tta

def train_segmenter(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Stage B Stain Segmenter on device: {device}")

    # Load dataset manifest
    df = pd.read_csv(cfg["data"]["manifest_path"])
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    img_size = tuple(cfg["data"]["img_size"])
    train_dataset = SolarStainDataset(train_df, transform=get_train_transforms(img_size), img_size=img_size)
    val_dataset = SolarStainDataset(val_df, transform=get_val_transforms(), img_size=img_size)

    batch_size = cfg["stage_b_stain_segmenter"]["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model with SMP
    encoder_name = cfg["stage_b_stain_segmenter"]["encoder_name"]
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    ).to(device)

    loss_fn = DiceFocalLoss(
        dice_weight=cfg["stage_b_stain_segmenter"]["loss"]["dice_weight"],
        focal_weight=cfg["stage_b_stain_segmenter"]["loss"]["focal_weight"],
        gamma=cfg["stage_b_stain_segmenter"]["loss"]["focal_gamma"]
    )

    epochs_frozen = cfg["stage_b_stain_segmenter"]["training"]["epochs_frozen"]
    epochs_unfrozen = cfg["stage_b_stain_segmenter"]["training"]["epochs_unfrozen"]
    lr_frozen = cfg["stage_b_stain_segmenter"]["training"]["lr_frozen"]
    lr_unfrozen = cfg["stage_b_stain_segmenter"]["training"]["lr_unfrozen"]

    best_val_iou = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    os.makedirs("models", exist_ok=True)
    save_path = "models/stain_segmenter_best.pt"

    # ================= PHASE 1: FROZEN ENCODER =================
    print(f"\n[*] --- Phase 1: Training Decoder Only ({epochs_frozen} epochs, lr={lr_frozen}) ---")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen, weight_decay=1e-4)

    for epoch in range(1, epochs_frozen + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_ious, val_dices = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
                masks_np = masks.cpu().numpy()

                for p, m in zip(probs, masks_np):
                    iou, dice = compute_iou_and_dice(p[0] > 0.5, m[0] > 0.5)
                    val_ious.append(iou)
                    val_dices.append(dice)

        mean_iou = np.mean(val_ious)
        mean_dice = np.mean(val_dices)
        print(f"Epoch {epoch:02d}/{epochs_frozen:02d} [Decoder] - Loss: {train_loss:.4f} | Val IoU: {mean_iou:.4f} | Val Dice: {mean_dice:.4f}")

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "best_val_iou": best_val_iou,
                "encoder_name": encoder_name,
                "config": cfg
            }, save_path)
            print(f"    --> Saved best checkpoint (Val IoU: {best_val_iou:.4f}) to {save_path}")

    # ================= PHASE 2: UNFREEZE ALL LAYERS =================
    print(f"\n[*] --- Phase 2: Fine-Tuning Full Network ({epochs_unfrozen} epochs, lr={lr_unfrozen}) ---")
    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=lr_unfrozen, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_unfrozen, eta_min=1e-6)

    patience = cfg["stage_b_stain_segmenter"]["training"]["early_stopping_patience"]
    patience_counter = 0

    for epoch in range(1, epochs_unfrozen + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation (direct for speed during training)
        model.eval()
        val_ious, val_dices, val_bf1s = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                probs = torch.sigmoid(model(imgs)).cpu().numpy()
                masks_np = masks.cpu().numpy()

                for p, m in zip(probs, masks_np):
                    iou, dice = compute_iou_and_dice(p[0] > 0.5, m[0] > 0.5)
                    bf1 = compute_boundary_f1(p[0] > 0.5, m[0] > 0.5)
                    val_ious.append(iou)
                    val_dices.append(dice)
                    val_bf1s.append(bf1)

        mean_iou = np.mean(val_ious)
        mean_dice = np.mean(val_dices)
        mean_bf1 = np.mean(val_bf1s)
        print(f"Epoch {epoch:02d}/{epochs_unfrozen:02d} [Full] - Loss: {train_loss:.4f} | Val IoU: {mean_iou:.4f} | Val Dice: {mean_dice:.4f} | Boundary F1: {mean_bf1:.4f}")

        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "best_val_iou": best_val_iou,
                "encoder_name": encoder_name,
                "config": cfg
            }, save_path)
            print(f"    --> Saved new best checkpoint (Val IoU: {best_val_iou:.4f}) to {save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[!] Early stopping triggered after {epoch} unfrozen epochs.")
                break

    print(f"\n[+] Segmenter Training Complete! Best Validation IoU: {best_val_iou:.4f}")
    return save_path

if __name__ == "__main__":
    train_segmenter()
