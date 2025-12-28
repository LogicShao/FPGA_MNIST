# model_tools/export_test_img.py
import argparse
import json
import os

from torchvision import datasets, transforms
import numpy as np


def export_image_header(output_path, mem_path=None, normalize=False, quant_params=None, index=0):
    # Load test set
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Pick a sample image
    img, label = dataset[index]  # img is (1, 28, 28) float 0~1

    img_f = img.numpy().squeeze()
    if normalize:
        img_f = (img_f - 0.1307) / 0.3081

    if quant_params:
        with open(quant_params, "r", encoding="utf-8") as f:
            params = json.load(f)
        scale = float(params["scales"]["s_in"])
    else:
        abs_max = float(np.max(np.abs(img_f)))
        scale = 127.0 / abs_max if abs_max > 0 else 1.0

    img_int8 = np.clip(np.round(img_f * scale), -128, 127).astype(np.int8)
    flat_img = img_int8.flatten()

    print(f"Exporting image idx {index}, label: {label}")

    # Generate C header content
    content = "#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n"
    content += "#include <stdint.h>\n\n"
    content += f"static const int TEST_IMAGE_LABEL = {label};\n\n"
    content += "static const int8_t test_image[784] = {\n    "
    content += ", ".join(map(str, flat_img))
    content += "\n};\n\n#endif\n"

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Wrote {output_path}")

    if mem_path:
        os.makedirs(os.path.dirname(mem_path) or ".", exist_ok=True)
        with open(mem_path, "w") as f:
            for value in flat_img:
                f.write(f"{(int(value) & 0xFF):02x}\n")
        print(f"Wrote {mem_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a test MNIST image to C header")
    parser.add_argument(
        "--output",
        type=str,
        default="../model_tests/v1/test_image.h",
        help="Output path for test_image.h",
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="../hardware/src/v1.1/tb/test_image.mem",
        help="Optional output path for test_image.mem",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply MNIST normalization (mean=0.1307, std=0.3081).",
    )
    parser.add_argument(
        "--quant-params",
        type=str,
        default=None,
        help="Use s_in from quant_params.json for input scaling.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="MNIST test image index (0-9999).",
    )
    args = parser.parse_args()

    export_image_header(args.output, args.mem, args.normalize, args.quant_params, args.index)
