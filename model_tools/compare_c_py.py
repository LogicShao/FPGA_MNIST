#!/usr/bin/env python
import argparse
import os
import re
import subprocess
from glob import glob

import numpy as np
import torch
from torchvision import datasets, transforms

from models import get_model


def parse_indices(spec):
    items = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            items.extend(range(int(a), int(b) + 1))
        else:
            items.append(int(part))
    return items


def find_latest_model(models_dir):
    paths = glob(os.path.join(models_dir, "*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth files found in {models_dir}")
    return max(paths, key=os.path.getctime)


def load_model(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    model_name = ckpt.get("model_name", "SimpleMLP")
    model_class = get_model(model_name)
    model = model_class()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_name


def write_test_image_bin(output_path, img_tensor):
    img_int8 = (img_tensor.numpy().squeeze() * 127).astype(np.int8)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(img_int8.tobytes())


def compile_c(source_path, exe_path, opt_level):
    cmd = ["g++", source_path, "-o", exe_path]
    if opt_level:
        cmd.append(f"-{opt_level}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compile failed:\n{result.stdout}\n{result.stderr}")


def run_c(exe_path, img_path=None, label=None):
    cmd = [exe_path]
    if img_path:
        cmd.append(img_path)
    if label is not None:
        cmd.append(str(label))
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(f"C run failed:\n{result.stdout}\n{result.stderr}")
    return result.stdout


def parse_prediction(output_text):
    for line in output_text.splitlines():
        if "Prediction" in line or "Pred" in line:
            nums = re.findall(r"-?\d+", line)
            if nums:
                return int(nums[-1])
    nums = re.findall(r"-?\d+", output_text)
    if nums:
        return int(nums[-1])
    raise RuntimeError("Could not parse prediction from C output")


def main():
    parser = argparse.ArgumentParser(description="Compare C inference with Python for MNIST indices")
    parser.add_argument("--indices", type=str, default="0", help="Comma list or range, e.g. 0,1,2 or 0-9")
    parser.add_argument("--opt", type=str, default="O2", help="Compiler optimization level (e.g. O0,O2,O3)")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--models-dir", type=str, default=os.path.join(base_dir, "trained_models"),
                        help="Directory with .pth models")
    parser.add_argument("--source", type=str, default=os.path.normpath(os.path.join(base_dir, "..", "model_tests", "v2", "main.c")),
                        help="Path to C source")
    parser.add_argument("--exe", type=str, default=os.path.normpath(os.path.join(base_dir, "..", "model_tests", "v2", "main.exe")),
                        help="Path to C executable")
    parser.add_argument("--bin", type=str, default=os.path.normpath(os.path.join(base_dir, "..", "model_tests", "v2", "test_image.bin")),
                        help="Path to test_image.bin")
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    if not indices:
        print("No indices provided")
        return 1

    model_path = find_latest_model(args.models_dir)
    model, model_name = load_model(model_path)
    print(f"Using model: {model_name} ({model_path})")

    data_dir = os.path.join(base_dir, "data")
    raw_ds = datasets.MNIST(data_dir, train=False, download=True,
                            transform=transforms.Compose([transforms.ToTensor()]))
    norm_ds = datasets.MNIST(data_dir, train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))

    c_correct = 0
    py_correct = 0
    agree = 0

    compile_c(args.source, args.exe, args.opt)

    for idx in indices:
        raw_img, raw_label = raw_ds[idx]
        norm_img, norm_label = norm_ds[idx]
        if raw_label != norm_label:
            raise RuntimeError("Label mismatch between datasets")

        write_test_image_bin(args.bin, raw_img)
        output = run_c(args.exe, args.bin, raw_label)
        c_pred = parse_prediction(output)

        with torch.no_grad():
            logits = model(norm_img.unsqueeze(0))
            py_pred = int(logits.argmax(dim=1).item())

        c_correct += int(c_pred == raw_label)
        py_correct += int(py_pred == raw_label)
        agree += int(c_pred == py_pred)

        print(f"idx {idx}: label={raw_label} c_pred={c_pred} py_pred={py_pred}")

    total = len(indices)
    print("---")
    print(f"C accuracy: {c_correct}/{total} = {c_correct/total:.3f}")
    print(f"Py accuracy: {py_correct}/{total} = {py_correct/total:.3f}")
    print(f"C vs Py agree: {agree}/{total} = {agree/total:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
