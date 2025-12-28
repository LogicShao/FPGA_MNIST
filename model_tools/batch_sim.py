#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import subprocess
import sys

import numpy as np
from torchvision import datasets, transforms


RE_PRED = re.compile(r"PRED\s*=\s*(\d+)")


def run_cmd(cmd, cwd):
    result = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def write_mem_file(mem_path, img_f, normalize, quant_scale):
    if normalize:
        img_f = (img_f - 0.1307) / 0.3081
    if quant_scale is not None:
        scale = quant_scale
    else:
        abs_max = float(np.max(np.abs(img_f)))
        scale = 127.0 / abs_max if abs_max > 0 else 1.0
    img_int8 = np.clip(np.round(img_f * scale), -128, 127).astype(np.int8)
    flat_img = img_int8.flatten()

    os.makedirs(os.path.dirname(mem_path) or ".", exist_ok=True)
    with open(mem_path, "w") as f:
        for value in flat_img:
            f.write(f"{(int(value) & 0xFF):02x}\n")
    return flat_img


def write_header_file(header_path, flat_img, label):
    content = "#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n"
    content += "#include <stdint.h>\n\n"
    content += f"static const int TEST_IMAGE_LABEL = {label};\n\n"
    content += "static const int8_t test_image[784] = {\n    "
    content += ", ".join(map(str, flat_img))
    content += "\n};\n\n#endif\n"

    os.makedirs(os.path.dirname(header_path) or ".", exist_ok=True)
    with open(header_path, "w") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Batch RTL sim on MNIST test set.")
    parser.add_argument("--count", type=int, default=20, help="Number of images to simulate.")
    parser.add_argument("--start", type=int, default=0, help="Start index in MNIST test set.")
    parser.add_argument("--out", default="model_tools/batch_sim_results.csv", help="CSV output path.")
    parser.add_argument("--normalize", action="store_true", help="Use MNIST normalization.")
    parser.add_argument("--quant-params", default="model_tools/quant_params.json", help="Quant params JSON.")
    parser.add_argument("--fast", action="store_true", help="Enable FAST_SIM shortcuts.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose TB outputs.")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    tfm = [transforms.ToTensor()]
    dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transforms.Compose(tfm))

    quant_scale = None
    if args.quant_params:
        with open(args.quant_params, "r", encoding="utf-8") as f:
            params = json.load(f)
        quant_scale = float(params["scales"]["s_in"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sim_base = os.path.join(repo_root, "hardware", "src", "v1.1")
    os.makedirs(os.path.join(sim_base, "sim"), exist_ok=True)

    out_file = os.path.join(sim_base, "sim", "tb_mnist_network_core.out")
    compiled = False
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "pred", "match"])

        total = 0
        correct = 0
        for idx in range(args.start, args.start + args.count):
            img, label = dataset[idx]
            img_f = img.numpy().squeeze()

            mem_path = os.path.join(repo_root, "hardware", "src", "v1.1", "tb", "test_image.mem")
            header_path = os.path.join(repo_root, "model_tests", "v1", "test_image.h")
            flat_img = write_mem_file(mem_path, img_f, args.normalize, quant_scale)
            write_header_file(header_path, flat_img, int(label))

            if not compiled:
                fast_flag = " --fast" if args.fast else ""
                quiet_flag = " --quiet" if args.quiet else ""
                cmd_compile = (
                    f"python hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core --no-wave"
                    f"{fast_flag}{quiet_flag}"
                )
                code, out = run_cmd(cmd_compile, cwd=repo_root)
                if code != 0:
                    print(out)
                    print("Compile failed")
                    break
                compiled = True

            code, out = run_cmd(f'vvp -n "{out_file}"', cwd=sim_base)
            if code != 0:
                print(out)
                print(f"Sim failed at idx {idx}")
                break

            m = RE_PRED.search(out)
            if not m:
                print(out)
                print(f"No PRED found at idx {idx}")
                break

            pred = int(m.group(1))
            match = int(pred == int(label))
            writer.writerow([idx, int(label), pred, match])
            total += 1
            correct += match
            print(f"[{idx}] label={label} pred={pred} match={match}")

        if total:
            acc = correct / total * 100.0
            print(f"Accuracy: {correct}/{total} = {acc:.2f}%")


if __name__ == "__main__":
    main()
