# model_tools/export_test_img.py
import torch
from torchvision import datasets, transforms
import numpy as np


def export_image_header():
    # 载入测试集
    dataset = datasets.MNIST('./data', train=False, download=True, 
                           transform=transforms.Compose([transforms.ToTensor()]))
    
    # 随便找一张比较清晰的图，比如第 7 张（通常是 9）或第 0 张（通常是 7）
    idx = 0 
    img, label = dataset[idx] # img is (1, 28, 28) float 0~1
    
    # 量化到 int8 (0~127)，要和训练时一致
    img_int8 = (img.numpy().squeeze() * 127).astype(np.int8)
    flat_img = img_int8.flatten()

    print(f"导出第 {idx} 张图，真实标签是: {label}")

    # 生成 C 内容
    content = f"#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n"
    content += "#include <stdint.h>\n\n"
    content += f"static const int TEST_IMAGE_LABEL = {label};\n\n"
    content += "static const int8_t test_image[784] = {\n    "
    content += ", ".join(map(str, flat_img))
    content += "\n};\n\n#endif\n"

    with open("../software/app/test_image.h", "w") as f:
        f.write(content)
    print("已生成 ../software/app/test_image.h")


if __name__ == "__main__":
    export_image_header()
