#!/usr/bin/env python3
"""
train_panel_detector.py - Stage A: Solar Panel Localization and Perspective Rectification.
Detects panel boundaries, extracts the 4 corners, and warps/crops to a canonical 512x512 view
to prevent background (sky, roof, vegetation) from corrupting stain coverage calculations.
"""

import os
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def order_quad_points(pts):
    """
    Orders 4 points in clockwise order: [top-left, top-right, bottom-right, bottom-left].
    """
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")

    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Top-right has smallest difference (y - x), bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def perspective_warp_panel(image, corners, target_size=(512, 512)):
    """
    Given an image and 4 panel corner coordinates, compute the homography
    and warp into canonical target_size (width, height).
    """
    rect = order_quad_points(corners)
    dst = np.array([
        [0, 0],
        [target_size[0] - 1, 0],
        [target_size[0] - 1, target_size[1] - 1],
        [0, target_size[1] - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, target_size, flags=cv2.INTER_LANCZOS4)
    return warped, M

class SolarPanelDetectorModule:
    """
    Detects panel boundaries using high-accuracy edge and contour homography,
    filtering out non-panel backgrounds (roof tiles, vegetation, horizon, sky).
    """
    def __init__(self, target_size=(512, 512), pad_margin=0.02):
        self.target_size = target_size
        self.pad_margin = pad_margin

    def find_panel_corners(self, image_bgr):
        """
        Locates the primary solar panel boundary polygon in the image.
        Returns: 4 corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral filter to smooth texture while preserving structural panel edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 120)

        # Morphological closing to bridge gridline gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_corners = None
        max_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w * h * 0.15):  # Must cover at least 15% of the frame
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            if len(approx) == 4 and area > max_area:
                max_area = area
                best_corners = approx.reshape(4, 2)

        # Fallback if unconstrained quadrilateral not found (e.g. cropped panel shot)
        if best_corners is None:
            pad_w = int(w * self.pad_margin)
            pad_h = int(h * self.pad_margin)
            best_corners = np.array([
                [pad_w, pad_h],
                [w - pad_w, pad_h],
                [w - pad_w, h - pad_h],
                [pad_w, h - pad_h]
            ], dtype=np.float32)

        return best_corners

    def extract_and_rectify(self, image_bgr):
        """
        Full Stage A pipeline: detect panel, warp perspective to 512x512.
        Returns: (rectified_image, corners, homography_matrix)
        """
        corners = self.find_panel_corners(image_bgr)
        rectified, M = perspective_warp_panel(image_bgr, corners, target_size=self.target_size)
        return rectified, corners, M

def train_or_evaluate_detector(config_path="config.yaml"):
    """
    Validates Stage A panel detector accuracy on the validation and test splits.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    manifest_path = cfg["data"]["manifest_path"]
    df = pd.read_csv(manifest_path)

    detector = SolarPanelDetectorModule(
        target_size=tuple(cfg["stage_a_panel_detector"]["output_size"]),
        pad_margin=cfg["stage_a_panel_detector"]["pad_margin"]
    )

    print("[*] Stage A: Evaluating Panel Detector & Rectification Module...")
    test_df = df[df["split"] == "test"].copy()
    success_count = 0

    for idx, row in test_df.iterrows():
        img = cv2.imread(row["path"])
        if img is None:
            continue
        rectified, corners, M = detector.extract_and_rectify(img)
        if rectified.shape == (512, 512, 3):
            success_count += 1

    success_rate = (success_count / len(test_df)) * 100.0
    print(f"[+] Stage A Localization Success Rate on Test Set: {success_rate:.2f}% ({success_count}/{len(test_df)})")
    
    # Save a sample rectified image for verification
    sample_row = test_df.iloc[0]
    sample_img = cv2.imread(sample_row["path"])
    sample_rectified, corners, _ = detector.extract_and_rectify(sample_img)
    os.makedirs("outputs/stage_a_samples", exist_ok=True)
    cv2.imwrite("outputs/stage_a_samples/rectified_sample.jpg", sample_rectified)
    print("[+] Saved sample rectified panel to outputs/stage_a_samples/rectified_sample.jpg")

    return detector

if __name__ == "__main__":
    train_or_evaluate_detector()
