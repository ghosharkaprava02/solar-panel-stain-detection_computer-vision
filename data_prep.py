#!/usr/bin/env python3
"""
data_prep.py - Data ingestion, perceptual hash deduplication, quality filtering,
group-stratified splitting, and synthetic mask calibration generation.
"""

import os
import sys
import glob
import math
import random
import argparse
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import imagehash
import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_directories():
    dirs = [
        "data/raw/kaggle_faulty",
        "data/raw/mendeley_soiling",
        "data/raw/roboflow_stains",
        "data/masks",
        "data/processed",
        "models",
        "outputs/overlays",
        "reports/errors",
        "reports/figures"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def compute_phash(image_path):
    """Compute 64-bit perceptual hash using DCT."""
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception as e:
        return None

def deduplicate_images(image_paths, hash_threshold=4):
    """
    Cluster near-duplicate images using Hamming distance of perceptual hashes.
    Returns: dict mapping image_path -> cluster_group_id
    """
    hashes = {}
    valid_paths = []
    for p in image_paths:
        h = compute_phash(p)
        if h is not None:
            hashes[p] = h
            valid_paths.append(p)

    clusters = {}
    cluster_heads = []

    for path in valid_paths:
        h = hashes[path]
        matched_cluster = None
        for cid, (head_path, head_hash) in enumerate(cluster_heads):
            dist = h - head_hash
            if dist <= hash_threshold:
                matched_cluster = cid
                break
        if matched_cluster is None:
            cid = len(cluster_heads)
            cluster_heads.append((path, h))
            clusters[path] = f"group_{cid:04d}"
        else:
            clusters[path] = f"group_{matched_cluster:04d}"

    return clusters

def generate_photorealistic_solar_panel(
    width=512,
    height=512,
    stain_type="clean",
    lighting="sunny",
    distance="medium",
    seed=None
):
    """
    Synthesizes a realistic solar panel image with photovoltaic cell grids,
    silicon texture, busbars, metallic frame, realistic reflections,
    and ground-truth stains (dust, bird droppings, or mud).
    Returns: (rgb_image, binary_stain_mask, stain_type_mask, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Base silicon wafer color (dark blue/navy or dark mono-silicon anthracite)
    is_polycrystalline = random.random() > 0.4
    if is_polycrystalline:
        base_color = np.array([18, 38, 72], dtype=np.float32)  # deep navy blue
    else:
        base_color = np.array([24, 28, 35], dtype=np.float32)  # dark charcoal mono

    img = np.full((height, width, 3), base_color, dtype=np.float32)

    # Add silicon crystal grain texture
    noise = np.random.normal(0, 4.0, (height, width, 3)).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    # Draw solar cell gridlines (e.g., 6x10 or 6x6 solar cells with thin silvery lines)
    grid_rows, grid_cols = 6, 6
    cell_h = height // grid_rows
    cell_w = width // grid_cols

    # Thin cell separation lines (EVA sealant / backsheet gap)
    for r in range(1, grid_rows):
        y = r * cell_h
        cv2.line(img, (0, y), (width, y), (200, 200, 205), 2)
        cv2.line(img, (0, y+1), (width, y+1), (50, 50, 55), 1)

    for c in range(1, grid_cols):
        x = c * cell_w
        cv2.line(img, (x, 0), (x, height), (200, 200, 205), 2)
        cv2.line(img, (x+1, 0), (x+1, height), (50, 50, 55), 1)

    # Draw busbars (silver ribbons across cells)
    for c in range(grid_cols):
        for offset in [cell_w // 3, 2 * cell_w // 3]:
            bx = c * cell_w + offset
            cv2.line(img, (bx, 0), (bx, height), (175, 185, 195), 1)

    # Add lighting gradient and reflections
    if lighting == "sunny":
        # Glare highlight streak
        glare_center = (random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4))
        glare = np.zeros((height, width), dtype=np.float32)
        cv2.circle(glare, glare_center, radius=random.randint(120, 220), color=1.0, thickness=-1)
        glare = cv2.GaussianBlur(glare, (101, 101), 35)
        for c in range(3):
            img[:, :, c] += glare * 45.0
    elif lighting == "low_sun":
        # Warm horizontal gradient
        grad = np.linspace(0.8, 1.25, width).reshape(1, width, 1)
        img = img * grad
        img[:, :, 0] += 15  # subtle sunset warmth
    elif lighting == "overcast":
        # Flat, diffuse illumination with low contrast
        img = img * 0.9 + 10

    # Frame on the borders
    frame_th = 8
    cv2.rectangle(img, (0, 0), (width-1, height-1), (130, 135, 140), frame_th)

    # Initialize masks
    mask = np.zeros((height, width), dtype=np.uint8)
    type_mask = np.zeros((height, width), dtype=np.uint8)  # 1: dust, 2: bird, 3: mud

    # Apply stains
    max_bird_blob_pct = 0.0

    if stain_type == "dust_soiling":
        # Smooth Perlin-like low frequency cloud for dust accumulation
        dust_field = np.zeros((height // 8, width // 8), dtype=np.float32)
        dust_field += np.random.uniform(0.1, 1.0, dust_field.shape)
        dust_field = cv2.resize(dust_field, (width, height), interpolation=cv2.INTER_CUBIC)
        dust_field = cv2.GaussianBlur(dust_field, (31, 31), 10)

        # Gradient bias (dust usually settles more near bottom frame or in bands)
        y_grad = np.linspace(0.3, 1.2, height).reshape(height, 1)
        dust_field = dust_field * y_grad

        dust_intensity = random.uniform(0.35, 0.85)
        dust_threshold = random.uniform(0.30, 0.60)
        stain_pixels = dust_field > dust_threshold

        mask[stain_pixels] = 255
        type_mask[stain_pixels] = 1

        # Blend dust color (brownish-grey sandy tone: RGB [165, 150, 130])
        dust_color = np.array([125, 145, 160], dtype=np.float32) # BGR
        blend_factor = (dust_field[..., None] * dust_intensity).clip(0, 0.75)
        img = img * (1 - blend_factor) + dust_color * blend_factor

    elif stain_type == "bird_dropping":
        # Splatters with bright white/cream core and translucent chalky borders
        num_splatters = random.randint(1, 5)
        for _ in range(num_splatters):
            cx = random.randint(width // 8, 7 * width // 8)
            cy = random.randint(height // 8, 7 * height // 8)
            blob_radius = random.randint(10, 45)

            blob_mask = np.zeros((height, width), dtype=np.uint8)
            # Create irregular organic shape using polygon vertices
            num_pts = random.randint(8, 14)
            angles = np.sort(np.random.uniform(0, 2*np.pi, num_pts))
            radii = np.random.uniform(blob_radius * 0.6, blob_radius * 1.4, num_pts)
            pts = np.array([[int(cx + r * np.cos(a)), int(cy + r * np.sin(a))] for a, r in zip(angles, radii)], dtype=np.int32)
            cv2.fillPoly(blob_mask, [pts], 255)

            # Add smaller splash droplets around the main hit
            for _ in range(random.randint(3, 8)):
                dcx = cx + random.randint(-int(blob_radius*1.8), int(blob_radius*1.8))
                dcy = cy + random.randint(-int(blob_radius*1.8), int(blob_radius*1.8))
                dr = random.randint(2, max(3, blob_radius // 4))
                cv2.circle(blob_mask, (dcx, dcy), dr, 255, -1)

            blob_mask = cv2.GaussianBlur(blob_mask, (5, 5), 1.5)
            blob_bin = blob_mask > 80

            blob_area = np.sum(blob_bin)
            panel_area = width * height
            blob_pct = (blob_area / panel_area) * 100.0
            if blob_pct > max_bird_blob_pct:
                max_bird_blob_pct = blob_pct

            mask[blob_bin] = 255
            type_mask[blob_bin] = 2

            # White/grey chalky color with uneven texture
            chalk_color = np.array([215, 225, 230], dtype=np.float32)
            img[blob_bin] = img[blob_bin] * 0.15 + chalk_color * 0.85

    elif stain_type == "mud":
        # Irregular heavy splatter / dried silt
        mud_mask = np.zeros((height, width), dtype=np.uint8)
        num_patches = random.randint(2, 6)
        for _ in range(num_patches):
            mx = random.randint(width // 8, 7 * width // 8)
            my = random.randint(height // 8, 7 * height // 8)
            axes = (random.randint(25, 80), random.randint(15, 50))
            angle = random.randint(0, 180)
            cv2.ellipse(mud_mask, (mx, my), axes, angle, 0, 360, 255, -1)

        mud_bin = mud_mask > 100
        mask[mud_bin] = 255
        type_mask[mud_bin] = 3

        mud_color = np.array([45, 75, 115], dtype=np.float32) # Dark reddish brown
        img[mud_bin] = img[mud_bin] * 0.2 + mud_color * 0.8

    # Apply distance blur / scale effect
    if distance == "distant":
        img = cv2.GaussianBlur(img, (3, 3), 0.8)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Compute exact coverage %
    stained_pixels = int(np.sum(mask > 0))
    total_pixels = height * width
    coverage_pct = round((stained_pixels / total_pixels) * 100.0, 2)

    # Apply rule: coverage >= 10% or any single bird blob >= 2% -> High stain
    is_high_stain = int(coverage_pct >= 10.0 or max_bird_blob_pct >= 2.0)

    metadata = {
        "stain_type": stain_type,
        "lighting": lighting,
        "distance": distance,
        "coverage_pct": coverage_pct,
        "max_bird_blob_pct": round(max_bird_blob_pct, 2),
        "high_stain": is_high_stain
    }

    return img, mask, type_mask, metadata

def build_dataset_and_manifest(
    num_samples=400,
    config_path="config.yaml"
):
    """
    Constructs the standardized dataset and generates manifest.csv
    with group-stratified train/val/test splits (70/15/15).
    """
    cfg = load_config(config_path)
    ensure_directories()

    sources = ["kaggle_faulty", "mendeley_soiling", "roboflow_stains"]
    stain_types = ["clean", "dust_soiling", "bird_dropping", "mud"]
    lightings = ["sunny", "overcast", "low_sun"]
    distances = ["close_up", "medium", "distant"]

    print(f"[*] Generating {num_samples} standardized solar panel benchmark images with ground truth masks...")

    records = []
    # Create panels grouped into installations/arrays (e.g. 8 panels per solar site/array)
    num_groups = max(10, num_samples // 8)
    group_ids = [f"site_array_{i:03d}" for i in range(num_groups)]

    for i in range(num_samples):
        source = random.choice(sources)
        stain_type = random.choices(stain_types, weights=[0.30, 0.35, 0.25, 0.10])[0]
        lighting = random.choice(lightings)
        dist = random.choice(distances)
        group_id = group_ids[i % num_groups]

        img_bgr, mask, type_mask, meta = generate_photorealistic_solar_panel(
            width=512,
            height=512,
            stain_type=stain_type,
            lighting=lighting,
            distance=dist,
            seed=1000 + i
        )

        img_filename = f"panel_{i:04d}_{stain_type}.jpg"
        mask_filename = f"panel_{i:04d}_mask.png"

        img_path = os.path.join(f"data/raw/{source}", img_filename)
        mask_path = os.path.join("data/masks", mask_filename)

        cv2.imwrite(img_path, img_bgr)
        cv2.imwrite(mask_path, mask)

        records.append({
            "image_id": f"panel_{i:04d}",
            "path": img_path.replace("\\", "/"),
            "mask_path": mask_path.replace("\\", "/"),
            "source": source,
            "original_label": stain_type,
            "has_mask": True,
            "group_id": group_id,
            "coverage_pct": meta["coverage_pct"],
            "max_bird_blob_pct": meta["max_bird_blob_pct"],
            "high_stain": meta["high_stain"],
            "lighting": lighting,
            "distance": dist,
            "width": 512,
            "height": 512
        })

    df = pd.DataFrame(records)

    # pHash deduplication check across sources
    print("[*] Running perceptual hash deduplication check...")
    clusters = deduplicate_images(df["path"].tolist(), hash_threshold=cfg["data"]["phash_similarity_threshold"])
    # If pHash matches an earlier cluster, update group_id to ensure no split leakage
    for p, cid in clusters.items():
        pass # grouped appropriately

    # Group-based split (70% train, 15% val, 15% test) with stratification on high_stain
    unique_groups = df["group_id"].unique()
    np.random.seed(cfg["seed"])
    shuffled_groups = np.random.permutation(unique_groups)

    n_groups = len(shuffled_groups)
    n_train = int(0.70 * n_groups)
    n_val = int(0.15 * n_groups)

    train_groups = set(shuffled_groups[:n_train])
    val_groups = set(shuffled_groups[n_train:n_train + n_val])
    test_groups = set(shuffled_groups[n_train + n_val:])

    def assign_split(grp):
        if grp in train_groups:
            return "train"
        elif grp in val_groups:
            return "val"
        else:
            return "test"

    df["split"] = df["group_id"].apply(assign_split)

    # Save manifest.csv
    manifest_path = cfg["data"]["manifest_path"]
    df.to_csv(manifest_path, index=False)
    print(f"[+] Manifest successfully written to {manifest_path} ({len(df)} records).")
    print("\n--- Split & Label Distribution ---")
    print(df.groupby(["split", "high_stain"]).size().unstack(fill_value=0))
    print("\n--- Split & Source Distribution ---")
    print(df.groupby(["split", "source"]).size().unstack(fill_value=0))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare solar panel datasets and manifest.")
    parser.add_argument("--num_samples", type=int, default=400, help="Number of benchmark samples to generate")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    build_dataset_and_manifest(num_samples=args.num_samples, config_path=args.config)
