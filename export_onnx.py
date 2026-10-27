#!/usr/bin/env python3
"""
export_onnx.py - Exports PyTorch models to ONNX and validates numeric parity with ONNX Runtime.
Supports:
1. Stage B Stain Segmentation U-Net model -> models/stain_segmenter.onnx
2. Baseline Classifier ResNet model -> models/classifier_baseline.onnx
"""

import os
import yaml
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
import segmentation_models_pytorch as smp
import timm

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def export_segmenter_to_onnx(config_path="config.yaml", pt_path="models/stain_segmenter_best.pt", onnx_path="models/stain_segmenter.onnx"):
    cfg = load_config(config_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    encoder_name = cfg["stage_b_stain_segmenter"]["encoder_name"]
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1
    )

    if os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded trained segmenter weights from {pt_path}")
    else:
        print(f"[!] Warning: {pt_path} not found. Exporting base initialized model.")

    model.eval()

    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float32)

    print(f"[*] Exporting Segmenter to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["stain_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "stain_logits": {0: "batch_size"}
        },
        dynamo=False
    )

    # Validate ONNX model integrity
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model integrity verified successfully!")

    # Verify parity with ONNX Runtime
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_image": dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"[+] Numeric Parity Check (PyTorch vs ONNX Runtime): Max Absolute Difference = {max_diff:.8f}")
    assert max_diff < 1e-4, f"Parity mismatch too large: {max_diff}"
    print(f"[+] Stage B Segmenter successfully exported and verified at {onnx_path}!\n")

def export_classifier_to_onnx(config_path="config.yaml", pt_path="models/classifier_baseline_best.pt", onnx_path="models/classifier_baseline.onnx"):
    cfg = load_config(config_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    backbone = cfg["baseline_classifier"]["backbone"]
    model = timm.create_model(backbone, pretrained=False, num_classes=2)

    if os.path.exists(pt_path):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded trained classifier weights from {pt_path}")
    else:
        print(f"[!] Warning: {pt_path} not found. Exporting base initialized model.")

    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    print(f"[*] Exporting Baseline Classifier to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["class_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "class_logits": {0: "batch_size"}
        },
        dynamo=False
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX classifier model integrity verified successfully!")

    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input_image": dummy_input.numpy()}
    ort_out = ort_session.run(None, ort_inputs)[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"[+] Numeric Parity Check (PyTorch vs ONNX Runtime): Max Absolute Difference = {max_diff:.8f}")
    assert max_diff < 1e-4, f"Parity mismatch too large: {max_diff}"
    print(f"[+] Baseline Classifier successfully exported and verified at {onnx_path}!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch models to ONNX.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    export_segmenter_to_onnx(config_path=args.config)
    export_classifier_to_onnx(config_path=args.config)
