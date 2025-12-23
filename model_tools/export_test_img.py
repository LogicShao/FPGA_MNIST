# model_tools/export_test_img.py
import argparse
import os

import torch
from torchvision import datasets, transforms
import numpy as np


def export_image_header(output_path):
    # Load test set
    dataset = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))

    # Pick a sample image (default: idx 0)
    idx = 0
    img, label = dataset[idx]  # img is (1, 28, 28) float 0~1

    # Quantize to int8 (0~127), keep consistent with training/export
    img_int8 = (img.numpy().squeeze() * 127).astype(np.int8)
    flat_img = img_int8.flatten()

    print(f"Exporting image idx {idx}, label: {label}")

    # Generate C header content
    content = "#ifndef TEST_IMAGE_H
#define TEST_IMAGE_H

"
    content += "#include <stdint.h>

"
    content += f"static const int TEST_IMAGE_LABEL = {label};

"
    content += "static const int8_t test_image[784] = {
    "
    content += ", ".join(map(str, flat_img))
    content += "
};

#endif
"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a test MNIST image to C header")
    parser.add_argument(
        "--output",
        type=str,
        default="../software/app/test_image.h",
        help="Output path for test_image.h",
    )
    args = parser.parse_args()

    export_image_header(args.output)
