#!/usr/bin/env python3
"""
train_classifier.py - Baseline Image-Level High/Low Stain Classifier.
Trains an image-level classifier (ResNet18 / EfficientNet-B0) to predict High/Low directly
as a comparison baseline and for ensembling with the Stage B segmenter.
"""

import os
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
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SolarClassifierDataset(Dataset):
    def __init__(self, df, transform=None, img_size=(256, 256)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        label = int(row["high_stain"])

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        if self.transform:
            augmented = self.transform(image=img)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.tensor(img / 255.0).permute(2, 0, 1).float()

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "path": img_path,
            "coverage_pct": float(row.get("coverage_pct", 0.0))
        }

def get_classifier_transforms(img_size=(256, 256)):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_classifier_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train_classifier(config_path="config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Baseline Classifier on device: {device}")

    manifest_path = cfg["data"]["manifest_path"]
    df = pd.read_csv(manifest_path)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    img_size = (256, 256)
    train_dataset = SolarClassifierDataset(train_df, transform=get_classifier_transforms(img_size), img_size=img_size)
    val_dataset = SolarClassifierDataset(val_df, transform=get_val_classifier_transforms(), img_size=img_size)

    batch_size = cfg["baseline_classifier"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    backbone = cfg["baseline_classifier"]["backbone"]
    model = timm.create_model(backbone, pretrained=True, num_classes=2).to(device)

    # Class weights to prioritize recall on High class (label=1)
    # Penalize missing High stain 2x more than false alarms
    weights = torch.tensor([1.0, cfg["evaluation"]["high_class_weight_penalty"]]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    epochs = cfg["baseline_classifier"]["epochs"]
    lr = float(cfg["baseline_classifier"]["lr"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    save_path = "models/classifier_baseline_best.pt"
    os.makedirs("models", exist_ok=True)

    patience = cfg["baseline_classifier"]["early_stopping_patience"]
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())
                val_probs.extend(probs)

        macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        high_recall = recall_score(val_targets, val_preds, pos_label=1, zero_division=0)
        high_prec = precision_score(val_targets, val_preds, pos_label=1, zero_division=0)

        print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} | Val Macro-F1: {macro_f1:.4f} | High Recall: {high_recall:.4f} | High Prec: {high_prec:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_weights,
                "backbone": backbone,
                "best_val_f1": best_val_f1,
                "config": cfg
            }, save_path)
            print(f"    --> Saved best classifier checkpoint (Macro-F1: {best_val_f1:.4f}) to {save_path}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    print(f"\n[+] Classifier Training Complete! Best Validation Macro-F1: {best_val_f1:.4f}")
    return save_path

if __name__ == "__main__":
    train_classifier()
