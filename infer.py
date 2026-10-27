#!/usr/bin/env python3
"""
infer.py - End-to-End Inference for Solar Panel Stain Detection & Severity Classification.
Performs:
1. Stage A: Panel localization and perspective rectification to 512x512.
2. Stage B: Stain segmentation with optional TTA.
3. Stain coverage calculation (% of panel pixels).
4. Connected component analysis for localized bird-dropping blobs (blob area > 2% of panel).
5. 4x4 grid cell area breakdown for sub-panel hotspot diagnosis.
6. Rule application: Coverage >= 10% or Bird Blob >= 2% -> High Stain, else Low Stain.
7. Generates visual overlay diagrams and outputs summary predictions CSV.
"""

import os
import sys
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import segmentation_models_pytorch as smp
import timm
from albumentations.pytorch import ToTensorV2
import albumentations as A

from train_panel_detector import SolarPanelDetectorModule
from train_stain_segmenter import predict_with_tta, SolarStainDataset, get_val_transforms

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class SolarStainInferencePipeline:
    def __init__(self, config_path="config.yaml", segmenter_weights=None, classifier_weights=None):
        self.cfg = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Stage A: Panel Detector
        self.detector = SolarPanelDetectorModule(
            target_size=tuple(self.cfg["stage_a_panel_detector"]["output_size"]),
            pad_margin=self.cfg["stage_a_panel_detector"]["pad_margin"]
        )

        # Stage B: Segmenter
        self.segmenter = None
        seg_path = segmenter_weights or "models/stain_segmenter_best.pt"
        if os.path.exists(seg_path):
            checkpoint = torch.load(seg_path, map_location=self.device, weights_only=False)
            enc = checkpoint.get("encoder_name", self.cfg["stage_b_stain_segmenter"]["encoder_name"])
            self.segmenter = smp.Unet(
                encoder_name=enc,
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
            self.segmenter.load_state_dict(checkpoint["model_state_dict"])
            self.segmenter.to(self.device)
            self.segmenter.eval()
            print(f"[+] Loaded Stage B Segmenter from {seg_path}")
        else:
            print(f"[!] Warning: Segmenter weights not found at {seg_path}. Running with initialized model.")
            self.segmenter = smp.Unet(
                encoder_name=self.cfg["stage_b_stain_segmenter"]["encoder_name"],
                encoder_weights="imagenet",
                in_channels=3,
                classes=1
            ).to(self.device)
            self.segmenter.eval()

        # Baseline Classifier
        self.classifier = None
        cls_path = classifier_weights or "models/classifier_baseline_best.pt"
        if os.path.exists(cls_path):
            checkpoint_cls = torch.load(cls_path, map_location=self.device, weights_only=False)
            bb = checkpoint_cls.get("backbone", self.cfg["baseline_classifier"]["backbone"])
            self.classifier = timm.create_model(bb, pretrained=False, num_classes=2)
            self.classifier.load_state_dict(checkpoint_cls["model_state_dict"])
            self.classifier.to(self.device)
            self.classifier.eval()
            print(f"[+] Loaded Baseline Classifier from {cls_path}")

        # Configurable rules
        self.cov_threshold = float(self.cfg["label_rules"]["coverage_threshold_pct"])
        self.bird_blob_threshold = float(self.cfg["label_rules"]["bird_blob_threshold_pct"])
        self.grid_rows = int(self.cfg["label_rules"]["grid_rows"])
        self.grid_cols = int(self.cfg["label_rules"]["grid_cols"])

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def analyze_blobs(self, binary_mask, total_panel_pixels):
        """
        Extract connected components to detect localized bird-dropping blobs.
        Returns: max_blob_pct, has_large_blob
        """
        mask_u8 = (binary_mask * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        max_blob_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > max_blob_area:
                max_blob_area = area

        max_blob_pct = round((max_blob_area / total_panel_pixels) * 100.0, 2)
        has_large_blob = (max_blob_pct >= self.bird_blob_threshold)
        return max_blob_pct, has_large_blob

    def compute_grid_coverage(self, binary_mask):
        """
        Calculates stain coverage percentage inside each grid cell (e.g. 4x4).
        Returns: 2D numpy array of shape (grid_rows, grid_cols) with percentages.
        """
        h, w = binary_mask.shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols
        grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y1 = r * cell_h
                y2 = (r + 1) * cell_h if r < self.grid_rows - 1 else h
                x1 = c * cell_w
                x2 = (c + 1) * cell_w if c < self.grid_cols - 1 else w

                cell_patch = binary_mask[y1:y2, x1:x2]
                cell_stain_pct = (np.sum(cell_patch) / cell_patch.size) * 100.0
                grid[r, c] = round(cell_stain_pct, 1)

        return grid

    def process_image(self, image_bgr, use_tta=True, ensemble=False):
        """
        Runs the full inspection pipeline on a single BGR image.
        """
        # Step 1: Stage A Localization & Rectification
        rectified_bgr, corners, M = self.detector.extract_and_rectify(image_bgr)

        # Step 2: Prepare tensor for Stage B Segmenter
        rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rectified_rgb)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)

        # Step 3: Stage B Segmenter Inference
        if use_tta:
            prob_tensor = predict_with_tta(self.segmenter, img_tensor, device=self.device)
        else:
            with torch.no_grad():
                prob_tensor = torch.sigmoid(self.segmenter(img_tensor))

        prob_map = prob_tensor.squeeze().cpu().numpy()
        binary_mask = (prob_map > 0.5).astype(np.uint8)

        # Step 4: Metric & Coverage Calculation
        total_pixels = binary_mask.size
        stained_pixels = int(np.sum(binary_mask))
        coverage_pct = round((stained_pixels / total_pixels) * 100.0, 2)

        # Step 5: Bird-dropping blob inspection
        max_blob_pct, has_large_blob = self.analyze_blobs(binary_mask, total_pixels)

        # Step 6: 4x4 Grid Breakdown
        grid_coverage = self.compute_grid_coverage(binary_mask)

        # Step 7: Classification Decision
        is_high_stain = (coverage_pct >= self.cov_threshold) or has_large_blob
        decision_reasons = []
        if coverage_pct >= self.cov_threshold:
            decision_reasons.append(f"Coverage ({coverage_pct}%) >= {self.cov_threshold}%")
        if has_large_blob:
            decision_reasons.append(f"Large Bird Blob ({max_blob_pct}%) >= {self.bird_blob_threshold}%")
        if not decision_reasons:
            decision_reasons.append("Below severity thresholds")

        # Optional Classifier / Ensemble
        cls_prob = None
        if self.classifier is not None:
            with torch.no_grad():
                cls_in = cv2.resize(rectified_rgb, (256, 256))
                cls_tensor = self.transform(image=cls_in)["image"].unsqueeze(0).to(self.device)
                logits_cls = self.classifier(cls_tensor)
                cls_prob = float(torch.softmax(logits_cls, dim=1)[0, 1].cpu().item())

            if ensemble:
                # Combined logic: high stain if segmenter triggers OR classifier is strongly confident (>0.60)
                if cls_prob > 0.60 and not is_high_stain:
                    is_high_stain = True
                    decision_reasons.append(f"Classifier ensemble override (p={cls_prob:.2f})")

        label_str = "High stain" if is_high_stain else "Low stain"

        return {
            "rectified_bgr": rectified_bgr,
            "binary_mask": binary_mask,
            "prob_map": prob_map,
            "coverage_pct": coverage_pct,
            "max_blob_pct": max_blob_pct,
            "has_large_blob": has_large_blob,
            "grid_coverage": grid_coverage,
            "is_high_stain": is_high_stain,
            "label": label_str,
            "reasons": "; ".join(decision_reasons),
            "classifier_prob": cls_prob
        }

    def render_overlay(self, image_bgr, result):
        """
        Creates a high-resolution, explainable visual overlay:
        - Rectified panel with translucent stain heatmap
        - 4x4 grid cell boundaries labeled with individual cell coverage %
        - Professional header banner with High/Low badge, total coverage, and rationale
        """
        rect = result["rectified_bgr"].copy()
        mask = result["binary_mask"]
        grid = result["grid_coverage"]
        h, w = rect.shape[:2]

        # Colored mask overlay: Crimson Red for stains [BGR: 0, 0, 235]
        overlay = rect.copy()
        stain_indices = mask > 0
        overlay[stain_indices] = [0, 20, 230]
        alpha = 0.45
        blended = cv2.addWeighted(overlay, alpha, rect, 1 - alpha, 0)

        # Draw 4x4 grid lines
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols

        for r in range(1, self.grid_rows):
            y = r * cell_h
            cv2.line(blended, (0, y), (w, y), (255, 255, 0), 1, cv2.LINE_AA)

        for c in range(1, self.grid_cols):
            x = c * cell_w
            cv2.line(blended, (x, 0), (x, h), (255, 255, 0), 1, cv2.LINE_AA)

        # Label each cell's coverage %
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                val = grid[r, c]
                text = f"{val:.0f}%"
                cx = c * cell_w + 12
                cy = r * cell_h + 24
                # Cell pill background
                cv2.rectangle(blended, (cx - 4, cy - 14), (cx + 42, cy + 5), (20, 20, 20), -1)
                cv2.putText(blended, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Create diagnostic top banner
        banner_h = 75
        canvas = np.zeros((h + banner_h, w, 3), dtype=np.uint8)
        canvas[banner_h:, :] = blended

        # Banner color: Crimson for High Stain, Forest Green for Low Stain
        banner_color = (30, 30, 195) if result["is_high_stain"] else (35, 140, 40)
        cv2.rectangle(canvas, (0, 0), (w, banner_h), banner_color, -1)

        # Header text
        status_text = f"[{result['label'].upper()}] - Coverage: {result['coverage_pct']:.1f}% (Max Blob: {result['max_blob_pct']:.1f}%)"
        cv2.putText(canvas, status_text, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        detail_text = f"Rule: {result['reasons']}"
        cv2.putText(canvas, detail_text, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        return canvas

def run_inference(image_path=None, input_dir=None, output_csv="outputs/predictions.csv", output_dir="outputs/overlays", ensemble=False):
    pipeline = SolarStainInferencePipeline()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    targets = []
    if image_path:
        targets.append(image_path)
    elif input_dir:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            targets.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if not targets:
        print("[!] No input images found.")
        return

    print(f"[*] Running inference on {len(targets)} image(s)...")
    results = []

    for path in targets:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        res = pipeline.process_image(img_bgr, use_tta=True, ensemble=ensemble)
        overlay = pipeline.render_overlay(img_bgr, res)

        basename = os.path.basename(path)
        overlay_name = f"overlay_{basename}"
        overlay_path = os.path.join(output_dir, overlay_name)
        cv2.imwrite(overlay_path, overlay)

        grid_flat = res["grid_coverage"].flatten().tolist()
        grid_str = ";".join([f"{v:.1f}" for v in grid_flat])

        results.append({
            "file": path.replace("\\", "/"),
            "overlay_path": overlay_path.replace("\\", "/"),
            "label": res["label"],
            "coverage_pct": res["coverage_pct"],
            "max_blob_pct": res["max_blob_pct"],
            "large_blob_detected": res["has_large_blob"],
            "per_cell_coverage": grid_str,
            "decision_rule": res["reasons"],
            "classifier_prob": res["classifier_prob"]
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"[+] Saved inference results to {output_csv}")
    print(f"[+] Saved visual overlays to {output_dir}/")
    print(df_out[["file", "label", "coverage_pct", "decision_rule"]].head(10))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solar panel stain inference.")
    parser.add_argument("--image", type=str, default=None, help="Path to single image")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--output-csv", type=str, default="outputs/predictions.csv", help="Path for CSV output")
    parser.add_argument("--output-dir", type=str, default="outputs/overlays", help="Directory for overlay images")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble Stage B and Classifier")
    args = parser.parse_args()

    run_inference(
        image_path=args.image,
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        output_dir=args.output_dir,
        ensemble=args.ensemble
    )
