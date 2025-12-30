#!/usr/bin/env python
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from models import TinyLeNet


def load_model(model_path):
    model = TinyLeNet()
    checkpoint = torch.load(model_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


def find_latest_model():
    candidates = glob.glob(os.path.join("model_tools", "trained_models", "*.pth"))
    if not candidates:
        raise FileNotFoundError("No .pth found under model_tools/trained_models")
    return max(candidates, key=os.path.getctime)


def compute_scale_from_max(abs_max):
    if abs_max <= 0:
        return 1.0
    return 127.0 / abs_max


def quantize_int8_tensor(acc, mult, shift):
    acc64 = acc.to(torch.int64)
    prod = acc64 * int(mult)
    prod = prod + (1 << (shift - 1))
    shifted = prod >> shift
    shifted = torch.clamp(shifted, -128, 127)
    return shifted.to(torch.int32)


def main():
    parser = argparse.ArgumentParser(description="Compute activation scales and fixed-point multipliers.")
    parser.add_argument("--model", default=None, help="Path to TinyLeNet .pth (default: latest).")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to scan.")
    parser.add_argument("--shift", type=int, default=16, help="Fixed-point shift (default: 16).")
    parser.add_argument("--normalize", action="store_true", help="Use training normalization (mean=0.1307, std=0.3081).")
    parser.add_argument("--out-json", default="model_tools/quant_params.json", help="Output JSON path.")
    parser.add_argument("--out-vh", default="hardware/src/v1.1/rtl/quant_params.vh", help="Output Verilog header path.")
    parser.add_argument("--int-calib", action="store_true",
                        help="Calibrate activation ranges with integer pipeline (pre-pool quantization).")
    args = parser.parse_args()

    model_path = args.model or find_latest_model()
    model = load_model(model_path)

    tfm = [transforms.ToTensor()]
    if args.normalize:
        tfm.append(transforms.Normalize((0.1307,), (0.3081,)))
    dataset = datasets.MNIST("model_tools/data", train=False, download=False, transform=transforms.Compose(tfm))

    max_in = 0.0
    with torch.no_grad():
        for idx in range(min(args.samples, len(dataset))):
            x, _ = dataset[idx]
            max_in = max(max_in, float(x.abs().max()))

    s_in = compute_scale_from_max(max_in)

    w1 = model.conv1.weight.detach().cpu().numpy()
    w2 = model.conv2.weight.detach().cpu().numpy()
    w3 = model.fc1.weight.detach().cpu().numpy()
    w4 = model.fc2.weight.detach().cpu().numpy()

    s_w1 = compute_scale_from_max(np.max(np.abs(w1)))
    s_w2 = compute_scale_from_max(np.max(np.abs(w2)))
    s_w3 = compute_scale_from_max(np.max(np.abs(w3)))
    s_w4 = compute_scale_from_max(np.max(np.abs(w4)))

    def mult_for(s_in_layer, s_w, s_out):
        # y_q = (acc * (s_out / (s_in * s_w))) with fixed-point multiplier.
        eff = s_out / (s_in_layer * s_w)
        mult = int(np.round(eff * (2 ** args.shift)))
        return eff, mult

    if args.int_calib:
        w1_q = torch.from_numpy(np.clip(np.round(w1 * s_w1), -127, 127).astype(np.int8))
        w2_q = torch.from_numpy(np.clip(np.round(w2 * s_w2), -127, 127).astype(np.int8))
        w3_q = torch.from_numpy(np.clip(np.round(w3 * s_w3), -127, 127).astype(np.int8))
        w4_q = torch.from_numpy(np.clip(np.round(w4 * s_w4), -127, 127).astype(np.int8))

        b1 = model.conv1.bias.detach().cpu().numpy()
        b2 = model.conv2.bias.detach().cpu().numpy()
        b3 = model.fc1.bias.detach().cpu().numpy()
        b4 = model.fc2.bias.detach().cpu().numpy()

        b1_q = torch.from_numpy(np.round(b1 * (s_in * s_w1)).astype(np.int32))

        max_conv1_int = 0.0
        with torch.no_grad():
            for idx in range(min(args.samples, len(dataset))):
                x, _ = dataset[idx]
                x_q = torch.clamp(torch.round(x * s_in), -128, 127).to(torch.int32)
                conv1 = F.conv2d(x_q.float().unsqueeze(0), w1_q.float())
                conv1 = conv1 + b1_q.view(1, 6, 1, 1).float()
                conv1 = torch.relu(conv1)
                max_conv1_int = max(max_conv1_int, float(conv1.max()))

        s_pool1 = compute_scale_from_max(max_conv1_int / (s_in * s_w1))
        eff1, m1 = mult_for(s_in, s_w1, s_pool1)

        b2_q = torch.from_numpy(np.round(b2 * (s_pool1 * s_w2)).astype(np.int32))
        max_conv2_int = 0.0
        with torch.no_grad():
            for idx in range(min(args.samples, len(dataset))):
                x, _ = dataset[idx]
                x_q = torch.clamp(torch.round(x * s_in), -128, 127).to(torch.int32)
                conv1 = F.conv2d(x_q.float().unsqueeze(0), w1_q.float())
                conv1 = conv1 + b1_q.view(1, 6, 1, 1).float()
                conv1 = torch.relu(conv1)
                conv1_q = quantize_int8_tensor(conv1, m1, args.shift)
                pool1 = F.max_pool2d(conv1_q.float(), 2)
                conv2 = F.conv2d(pool1, w2_q.float())
                conv2 = conv2 + b2_q.view(1, 16, 1, 1).float()
                conv2 = torch.relu(conv2)
                max_conv2_int = max(max_conv2_int, float(conv2.max()))

        s_pool2 = compute_scale_from_max(max_conv2_int / (s_pool1 * s_w2))
        eff2, m2 = mult_for(s_pool1, s_w2, s_pool2)

        b3_q = torch.from_numpy(np.round(b3 * (s_pool2 * s_w3)).astype(np.int32))
        max_fc1_int = 0.0
        with torch.no_grad():
            for idx in range(min(args.samples, len(dataset))):
                x, _ = dataset[idx]
                x_q = torch.clamp(torch.round(x * s_in), -128, 127).to(torch.int32)
                conv1 = F.conv2d(x_q.float().unsqueeze(0), w1_q.float())
                conv1 = conv1 + b1_q.view(1, 6, 1, 1).float()
                conv1 = torch.relu(conv1)
                conv1_q = quantize_int8_tensor(conv1, m1, args.shift)
                pool1 = F.max_pool2d(conv1_q.float(), 2)
                conv2 = F.conv2d(pool1, w2_q.float())
                conv2 = conv2 + b2_q.view(1, 16, 1, 1).float()
                conv2 = torch.relu(conv2)
                conv2_q = quantize_int8_tensor(conv2, m2, args.shift)
                pool2 = F.max_pool2d(conv2_q.float(), 2)
                flat = pool2.view(1, -1).to(torch.int64)
                fc1 = torch.matmul(flat, w3_q.view(32, -1).t().to(torch.int64))
                fc1 = fc1 + b3_q.view(1, 32).to(torch.int64)
                fc1 = torch.relu(fc1)
                max_fc1_int = max(max_fc1_int, float(fc1.max()))

        s_fc1 = compute_scale_from_max(max_fc1_int / (s_pool2 * s_w3))
        eff3, m3 = mult_for(s_pool2, s_w3, s_fc1)

        b4_q = torch.from_numpy(np.round(b4 * (s_fc1 * s_w4)).astype(np.int32))
        max_fc2_int = 0.0
        with torch.no_grad():
            for idx in range(min(args.samples, len(dataset))):
                x, _ = dataset[idx]
                x_q = torch.clamp(torch.round(x * s_in), -128, 127).to(torch.int32)
                conv1 = F.conv2d(x_q.float().unsqueeze(0), w1_q.float())
                conv1 = conv1 + b1_q.view(1, 6, 1, 1).float()
                conv1 = torch.relu(conv1)
                conv1_q = quantize_int8_tensor(conv1, m1, args.shift)
                pool1 = F.max_pool2d(conv1_q.float(), 2)
                conv2 = F.conv2d(pool1, w2_q.float())
                conv2 = conv2 + b2_q.view(1, 16, 1, 1).float()
                conv2 = torch.relu(conv2)
                conv2_q = quantize_int8_tensor(conv2, m2, args.shift)
                pool2 = F.max_pool2d(conv2_q.float(), 2)
                flat = pool2.view(1, -1).to(torch.int64)
                fc1 = torch.matmul(flat, w3_q.view(32, -1).t().to(torch.int64))
                fc1 = fc1 + b3_q.view(1, 32).to(torch.int64)
                fc1 = torch.relu(fc1)
                fc1_q = quantize_int8_tensor(fc1, m3, args.shift)
                fc2 = torch.matmul(fc1_q.to(torch.int64), w4_q.view(10, -1).t().to(torch.int64))
                fc2 = fc2 + b4_q.view(1, 10).to(torch.int64)
                max_fc2_int = max(max_fc2_int, float(fc2.max()))

        s_fc2 = compute_scale_from_max(max_fc2_int / (s_fc1 * s_w4))
        eff4, m4 = mult_for(s_fc1, s_w4, s_fc2)
    else:
        max_conv1 = 0.0
        max_conv2 = 0.0
        max_fc1 = 0.0
        max_fc2 = 0.0

        with torch.no_grad():
            for idx in range(min(args.samples, len(dataset))):
                x, _ = dataset[idx]
                x = x.unsqueeze(0)

                x = F.relu(model.conv1(x))
                max_conv1 = max(max_conv1, float(x.abs().max()))
                x = F.max_pool2d(x, 2)

                x = F.relu(model.conv2(x))
                max_conv2 = max(max_conv2, float(x.abs().max()))
                x = F.max_pool2d(x, 2)

                x = x.view(-1, 16 * 4 * 4)
                x = F.relu(model.fc1(x))
                max_fc1 = max(max_fc1, float(x.abs().max()))

                x = model.fc2(x)
                max_fc2 = max(max_fc2, float(x.abs().max()))

        s_pool1 = compute_scale_from_max(max_conv1)
        s_pool2 = compute_scale_from_max(max_conv2)
        s_fc1 = compute_scale_from_max(max_fc1)
        s_fc2 = compute_scale_from_max(max_fc2)

        eff1, m1 = mult_for(s_in, s_w1, s_pool1)
        eff2, m2 = mult_for(s_pool1, s_w2, s_pool2)
        eff3, m3 = mult_for(s_pool2, s_w3, s_fc1)
        eff4, m4 = mult_for(s_fc1, s_w4, s_fc2)

    data = {
        "model_path": model_path,
        "samples": args.samples,
        "shift": args.shift,
        "normalize": bool(args.normalize),
        "scales": {
            "s_in": float(s_in),
            "s_pool1": float(s_pool1),
            "s_pool2": float(s_pool2),
            "s_fc1": float(s_fc1),
            "s_fc2": float(s_fc2),
            "s_w1": float(s_w1),
            "s_w2": float(s_w2),
            "s_w3": float(s_w3),
            "s_w4": float(s_w4),
        },
        "effective": {
            "conv1": {"eff": float(eff1), "mult": int(m1)},
            "conv2": {"eff": float(eff2), "mult": int(m2)},
            "fc1": {"eff": float(eff3), "mult": int(m3)},
            "fc2": {"eff": float(eff4), "mult": int(m4)},
        },
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    os.makedirs(os.path.dirname(args.out_vh) or ".", exist_ok=True)
    with open(args.out_vh, "w", newline="\n") as f:
        f.write("// Auto-generated by calc_quant_params.py\n")
        f.write(f"// model: {model_path}\n")
        f.write(f"// normalize: {int(args.normalize)} samples: {args.samples}\n")
        f.write(f"`define Q_SHIFT {args.shift}\n")
        f.write(f"`define Q_MULT_CONV1 {m1}\n")
        f.write(f"`define Q_MULT_CONV2 {m2}\n")
        f.write(f"`define Q_MULT_FC1 {m3}\n")
        f.write(f"`define Q_MULT_FC2 {m4}\n")

    print("Wrote:", args.out_json)
    print("Wrote:", args.out_vh)
    print("Effective scales:")
    print(f"  conv1: {eff1:.8f} mult={m1}")
    print(f"  conv2: {eff2:.8f} mult={m2}")
    print(f"  fc1  : {eff3:.8f} mult={m3}")
    print(f"  fc2  : {eff4:.8f} mult={m4}")


if __name__ == "__main__":
    main()
