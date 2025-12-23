"""
模型导出脚本 - 支持多模型多格式导出
解耦版：专注于导出功能，训练功能在train.py
"""

import torch
import numpy as np
import argparse
import os
import glob

# 导入模型注册系统
from models import get_model, get_model_info, MODEL_REGISTRY


def load_trained_model(model_path):
    """加载训练好的模型"""
    print(f"正在加载模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path)
    model_name = checkpoint.get('model_name', 'SimpleMLP')
    model_type = checkpoint.get('model_type', 'mlp')
    test_accuracy = checkpoint.get('test_accuracy', None)

    # 获取模型类并加载权重
    model_class = get_model(model_name)
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"模型加载完成: {model_name} ({model_type})")
    if test_accuracy:
        print(f"训练时测试准确率: {test_accuracy:.2f}%")

    return model, model_name, model_type


def quantize_symmetric(arr, name=""):
    """对称量化：动态计算最优缩放因子"""
    abs_max = np.max(np.abs(arr))

    if abs_max < 1e-8:
        print(f"  警告: {name} 接近零，使用默认缩放")
        return np.zeros_like(arr, dtype=np.int8), 1.0

    scale = 127.0 / abs_max
    quantized = np.clip(np.round(arr * scale), -127, 127).astype(np.int8)

    # 计算量化误差
    dequantized = quantized / scale
    error = np.mean(np.abs(arr - dequantized))

    print(f"  {name}: scale={scale:.2f}, abs_max={abs_max:.6f}, avg_error={error:.6f}")

    return quantized, scale


def export_mlp_to_c(model, filename):
    """导出MLP模型到C头文件"""
    print(f"\n正在导出MLP模型到 {filename} ...")

    # 提取权重
    weights1 = model.fc1.weight.detach().cpu().numpy()
    bias1 = model.fc1.bias.detach().cpu().numpy()
    weights2 = model.fc2.weight.detach().cpu().numpy()
    bias2 = model.fc2.bias.detach().cpu().numpy()

    hidden_size = weights1.shape[0]

    # 量化
    print("正在量化权重...")
    q_w1, scale_w1 = quantize_symmetric(weights1, "W1 (Layer1 Weights)")
    q_b1, scale_b1 = quantize_symmetric(bias1, "B1 (Layer1 Biases)")
    q_w2, scale_w2 = quantize_symmetric(weights2, "W2 (Layer2 Weights)")
    q_b2, scale_b2 = quantize_symmetric(bias2, "B2 (Layer2 Biases)")

    # 生成C代码
    c_content = f"""#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

// Network Config
#define INPUT_SIZE 784
#define HIDDEN_SIZE {hidden_size}
#define OUTPUT_SIZE 10

// Quantization Scales
#define SCALE_W1 {scale_w1:.6f}f
#define SCALE_B1 {scale_b1:.6f}f
#define SCALE_W2 {scale_w2:.6f}f
#define SCALE_B2 {scale_b2:.6f}f

// Layer 1 Weights ({hidden_size} x 784)
static const int8_t W1[{hidden_size}][784] = {{
"""

    for i in range(hidden_size):
        c_content += "    {" + ", ".join(map(str, q_w1[i])) + "},\n"

    c_content += "};\n\n// Layer 1 Biases\nstatic const int8_t B1[] = {"
    c_content += ", ".join(map(str, q_b1)) + "};\n"

    c_content += f"\n// Layer 2 Weights (10 x {hidden_size})\nstatic const int8_t W2[10][{hidden_size}] = {{\n"

    for i in range(10):
        c_content += "    {" + ", ".join(map(str, q_w2[i])) + "},\n"

    c_content += "};\n\n// Layer 2 Biases\nstatic const int8_t B2[] = {"
    c_content += ", ".join(map(str, q_b2)) + "};\n"

    c_content += "\n#endif // MODEL_WEIGHTS_H\n"

    # 写入文件
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write(c_content)

    print(f"导出完成: {filename}")


def export_cnn_to_c(model, filename):
    """导出CNN模型到C头文件（适用于TinyLeNet）"""
    print(f"\n正在导出CNN模型到 {filename} ...")

    # 提取卷积层权重
    conv1_weight = model.conv1.weight.detach().cpu().numpy()  # (6, 1, 5, 5)
    conv1_bias = model.conv1.bias.detach().cpu().numpy()      # (6,)
    conv2_weight = model.conv2.weight.detach().cpu().numpy()  # (16, 6, 5, 5)
    conv2_bias = model.conv2.bias.detach().cpu().numpy()      # (16,)

    # 提取全连接层权重
    fc1_weight = model.fc1.weight.detach().cpu().numpy()      # (32, 256)
    fc1_bias = model.fc1.bias.detach().cpu().numpy()          # (32,)
    fc2_weight = model.fc2.weight.detach().cpu().numpy()      # (10, 32)
    fc2_bias = model.fc2.bias.detach().cpu().numpy()          # (10,)

    print("正在量化权重...")
    # 量化卷积层
    q_conv1_w, scale_conv1_w = quantize_symmetric(conv1_weight, "Conv1 Weights")
    q_conv1_b, scale_conv1_b = quantize_symmetric(conv1_bias, "Conv1 Biases")
    q_conv2_w, scale_conv2_w = quantize_symmetric(conv2_weight, "Conv2 Weights")
    q_conv2_b, scale_conv2_b = quantize_symmetric(conv2_bias, "Conv2 Biases")

    # 量化全连接层
    q_fc1_w, scale_fc1_w = quantize_symmetric(fc1_weight, "FC1 Weights")
    q_fc1_b, scale_fc1_b = quantize_symmetric(fc1_bias, "FC1 Biases")
    q_fc2_w, scale_fc2_w = quantize_symmetric(fc2_weight, "FC2 Weights")
    q_fc2_b, scale_fc2_b = quantize_symmetric(fc2_bias, "FC2 Biases")

    # 生成C代码
    c_content = f"""#ifndef TINYLENET_WEIGHTS_H
#define TINYLENET_WEIGHTS_H

#include <stdint.h>

// Network Architecture
// Conv1: 1@28x28 -> 6@24x24 (5x5 kernel)
// Pool1: 6@24x24 -> 6@12x12 (2x2 max pool)
// Conv2: 6@12x12 -> 16@8x8 (5x5 kernel)
// Pool2: 16@8x8 -> 16@4x4 (2x2 max pool)
// FC1: 256 -> 32
// FC2: 32 -> 10

// Quantization Scales
#define SCALE_CONV1_W {scale_conv1_w:.6f}f
#define SCALE_CONV1_B {scale_conv1_b:.6f}f
#define SCALE_CONV2_W {scale_conv2_w:.6f}f
#define SCALE_CONV2_B {scale_conv2_b:.6f}f
#define SCALE_FC1_W {scale_fc1_w:.6f}f
#define SCALE_FC1_B {scale_fc1_b:.6f}f
#define SCALE_FC2_W {scale_fc2_w:.6f}f
#define SCALE_FC2_B {scale_fc2_b:.6f}f

// Conv1: 6 kernels, each 1x5x5
// Shape: [6][1][5][5]
static const int8_t CONV1_WEIGHTS[6][25] = {{
"""

    # 展平每个5x5卷积核
    for i in range(6):
        kernel_flat = q_conv1_w[i, 0, :, :].flatten()
        c_content += "    {" + ", ".join(map(str, kernel_flat)) + "},\n"

    c_content += "};\n\nstatic const int8_t CONV1_BIASES[6] = {"
    c_content += ", ".join(map(str, q_conv1_b)) + "};\n\n"

    # Conv2: 16 kernels, each 6x5x5
    c_content += "// Conv2: 16 kernels, each 6x5x5\n"
    c_content += "// Shape: [16][6][5][5]\n"
    c_content += "static const int8_t CONV2_WEIGHTS[16][150] = {\n"

    for i in range(16):
        kernel_flat = q_conv2_w[i, :, :, :].flatten()
        c_content += "    {" + ", ".join(map(str, kernel_flat)) + "},\n"

    c_content += "};\n\nstatic const int8_t CONV2_BIASES[16] = {"
    c_content += ", ".join(map(str, q_conv2_b)) + "};\n\n"

    # 全连接层
    c_content += f"// FC1: 256 -> 32\nstatic const int8_t FC1_WEIGHTS[32][256] = {{\n"
    for i in range(32):
        c_content += "    {" + ", ".join(map(str, q_fc1_w[i])) + "},\n"

    c_content += "};\n\nstatic const int8_t FC1_BIASES[32] = {"
    c_content += ", ".join(map(str, q_fc1_b)) + "};\n\n"

    c_content += f"// FC2: 32 -> 10\nstatic const int8_t FC2_WEIGHTS[10][32] = {{\n"
    for i in range(10):
        c_content += "    {" + ", ".join(map(str, q_fc2_w[i])) + "},\n"

    c_content += "};\n\nstatic const int8_t FC2_BIASES[10] = {"
    c_content += ", ".join(map(str, q_fc2_b)) + "};\n"

    c_content += "\n#endif // TINYLENET_WEIGHTS_H\n"

    # 写入文件
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, 'w') as f:
        f.write(c_content)

    print(f"导出完成: {filename}")
    print("\n提示: TinyLeNet需要实现硬件加速器，请参考 docs/TinyLeNet_fpga.md")


def export_model(model_path, output_path=None):
    """自动检测模型类型并导出"""
    # 加载模型
    model, model_name, model_type = load_trained_model(model_path)

    # 确定输出路径
    if output_path is None:
        if model_type == 'mlp':
            output_path = "../software/app/model_weights.h"
        else:
            output_path = f"../software/app/{model_name.lower()}_weights.h"

    # 根据模型类型选择导出方法
    if model_type == 'mlp':
        export_mlp_to_c(model, output_path)
    elif model_type == 'cnn':
        export_cnn_to_c(model, output_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST 模型导出脚本")

    parser.add_argument('--model-path', type=str, required=False,
                        help='训练好的模型路径 (.pth文件)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出C头文件路径')
    parser.add_argument('--latest', action='store_true',
                        help='自动使用最新训练的模型')
    parser.add_argument('--list', action='store_true',
                        help='列出所有已训练的模型')

    args = parser.parse_args()

    # 列出所有模型
    if args.list:
        models_dir = "./trained_models"
        if os.path.exists(models_dir):
            models = glob.glob(os.path.join(models_dir, "*.pth"))
            if models:
                print(f"\n已训练的模型 ({len(models)}个):")
                print("-" * 70)
                for i, m in enumerate(sorted(models, reverse=True)[:10], 1):
                    print(f"  {i}. {os.path.basename(m)}")
                print("-" * 70)
            else:
                print("没有找到已训练的模型")
        else:
            print("trained_models目录不存在")
        exit(0)

    # 使用最新模型
    if args.latest:
        models_dir = "./trained_models"
        models = glob.glob(os.path.join(models_dir, "*.pth"))
        if not models:
            print("错误: 没有找到已训练的模型")
            exit(1)
        model_path = max(models, key=os.path.getctime)
        print(f"使用最新模型: {model_path}")
    elif args.model_path:
        model_path = args.model_path
    else:
        print("错误: 请指定 --model-path 或使用 --latest")
        parser.print_help()
        exit(1)

    # 导出模型
    export_model(model_path, args.output)

    print("\n任务完成！")
