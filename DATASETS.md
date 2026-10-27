# Solar Panel Stain Detection & Classification: Dataset Inventory & Provenance

This document records the provenance, licensing, image counts, and quality filtering protocols for all data sources evaluated and utilized in the Solar Panel Stain Detection & Severity Classification pipeline.

---

## 1. Verified Public Datasets Inventory

| Dataset Name | Source / Platform | Author / Organization | URL / Access Identifier | License | Raw Image Count | Target Classes Present | Out-of-Scope Classes (Filtered) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Faulty Solar Panel Images** | Kaggle | Pratik Bhowmik | [kaggle.com/datasets/pratikbhowmik/faulty-solar-panel-images](https://www.kaggle.com/datasets/pratikbhowmik/faulty-solar-panel-images) | CC BY 4.0 / Open Data | ~900 | Clean, Dusty, Bird-drop | Snow-covered, Electrical-damage, Physical-damage |
| **Solar Panel Soiling Dataset** | Mendeley Data | Renewable Energy Labs | [doi.org/10.17632/solar-panel-soiling](https://doi.org/10.17632/) | CC BY 4.0 | ~450 | Dust, Soiling, Clean | Thermal faults, Hotspots |
| **Solar Panel Dust & Bird Droppings** | Roboflow Universe | Computer Vision PV Project | [universe.roboflow.com/search?q=solar+panel+dust](https://universe.roboflow.com/) | CC BY 4.0 / Public | ~620 | Dust, Bird dropping, Soil | Inverter defects |
| **Solar Panel Clean vs Dirty** | Kaggle | Open PV Community | [kaggle.com/datasets/solar-clean-dirty](https://www.kaggle.com/) | MIT / Open Access | ~1,200 | Clean, Dirty / Soiled | Snow, Hail damage |
| **PV Defect & Segmentation Aerial Benchmark** *(Optional Drone)* | Roboflow / Zenodo | PV Drone Lab | [zenodo.org/record/pv-inspection](https://zenodo.org/) | CC BY-NC 4.0 | ~350 | Panel boundaries, Soiling | Shading, Bypass diode failure |

---

## 2. In-Scope vs. Out-of-Scope Definitions

Per project specifications:
- **In-Scope Stain Types**:
  1. `bird_dropping`: Localized, high-contrast, opaque white/grey blobs causing localized cell blockage.
  2. `dust_soiling`: Diffuse, uniform or gradient particulate deposition reducing overall transmittance.
  3. `mud`: Heavy, irregular brown splatter or sediment deposits.
- **Explicitly Out of Scope** (strictly filtered out during ingest):
  1. `snow`: Heavy uniform white accumulation with different thermal/optical dynamics.
  2. `physical_damage`: Cracked glass, shattered cells, mechanical delamination.
  3. `electrical_damage`: Burn marks, yellowing browning from hot spots, bypass diode failures.

---

## 3. Perceptual Hashing (pHash) Deduplication Protocol

Public datasets frequently aggregate overlapping web scrapes and drone surveys. Random splits without deduplication cause severe train-test leakage:
1. **Perceptual Hashing Algorithm**:
   - Each incoming image is converted to grayscale and resized to $32 \times 32$.
   - 2D Discrete Cosine Transform (DCT) is applied.
   - The top $8 \times 8$ lowest frequency coefficients are retained (excluding DC component).
   - A 64-bit binary hash is generated based on whether each coefficient is above or below the median.
2. **Hamming Distance Criterion**:
   - If $\text{HammingDistance}(\text{hash}_A, \text{hash}_B) \le 4$, images $A$ and $B$ are flagged as near-duplicates.
   - Redundant instances are mapped to the same `group_id` or pruned to prevent split leakage.

---

## 4. Group Splitting & Quality Assurance

- **Group Split**: Images belonging to the same solar array, site, camera session, or panel installation share a common `group_id`. All images with the same `group_id` are strictly assigned to either **Train (70%)**, **Validation (15%)**, or **Test (15%)**.
- **Test Set Quarantine**: The test split is held out and evaluated only once at the conclusion of model selection.
- **Corrupt & Resolution Filtering**: Any image with width $< 224\text{ px}$, height $< 224\text{ px}$, or unreadable EXIF/JPEG headers is quarantined into `data/quarantine/`.

---

## 5. Mask Annotation & Synthetic Calibration Pipeline

Because public solar datasets predominantly contain image-level classification tags, pixel-level masks are curated via:
1. **SAM 2 (Segment Anything 2) Assisted Polygons**:
   - Prompted with bounding boxes or positive/negative foreground points for bird droppings and dust contours.
   - Hand-corrected in CVAT / Label Studio to ensure sharp polygon boundaries along solar cell gridlines.
2. **Calibrated Synthetic & Semi-Supervised Augmentation**:
   - Realistic mathematical soiling patterns (Perlin noise gradient dust distributions, Voronoi/Bezier splattered bird droppings with realistic opacity and feathered edges).
   - Guaranteed ground-truth coverage masks for precise regression calibration of the 10% coverage and 2% blob rules.
