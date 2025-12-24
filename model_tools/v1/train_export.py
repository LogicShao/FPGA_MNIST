import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import csv
from datetime import datetime
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")

# 配置
BATCH_SIZE = 1024
EPOCHS = 3
HIDDEN_SIZE = 16  # 隐层节点数，FPGA资源有限，32或16比较合适
MODEL_PATH = "./mnist_model.pth"  # 模型保存路径
EXPORT_PATH = "../model_tests/v1/model_weights.h"  # 导出到 C 代码目录
DATA_DIR = "./data"  # 数据集存储目录（相对于当前脚本）
LOG_DIR = "./logs"  # 训练日志目录


# Early Stopping 实现
class EarlyStopping:
    """早停机制：监控验证指标，连续N个epoch未改善则停止训练"""
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        """
        Args:
            patience: 容忍的epoch数量
            min_delta: 最小改善阈值
            mode: 'max'表示指标越大越好（如准确率），'min'表示越小越好（如损失）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


# 1. 定义简单的 MLP 网络
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # Layer 1: 784 -> 32
        self.fc1 = nn.Linear(28 * 28, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        # Layer 2: 32 -> 10
        self.fc2 = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 2. 数据加载（支持数据增强）
def get_data_loaders(augmentation=True):
    """
    获取训练集和测试集的DataLoader
    Args:
        augmentation: 是否对训练集应用数据增强
    """
    # 测试集transform：仅标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 训练集transform：可选数据增强
    if augmentation:
        print("数据增强: 已启用 (随机旋转±10°, 随机平移±10%, 随机缩放0.9-1.1)")
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,           # 随机旋转±10度
                translate=(0.1, 0.1), # 随机平移±10%
                scale=(0.9, 1.1),     # 随机缩放90%-110%
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("数据增强: 已禁用")
        train_transform = test_transform

    # 训练集
    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 测试集
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# 3. 模型评估
def evaluate(model, test_loader, device='cpu'):
    """在测试集上评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy


# 4. 训练流程（学习率调度 + Early Stopping + 进度条 + 日志 + 数据增强）
def train(save_best=True, model_path=MODEL_PATH, use_scheduler=True,
          early_stopping_patience=7, log_enabled=True, augmentation=True):
    print("正在准备数据...")
    train_loader, test_loader = get_data_loaders(augmentation=augmentation)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器：验证准确率停滞时降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    ) if use_scheduler else None

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, min_delta=0.01, mode='max'
    ) if early_stopping_patience > 0 else None

    best_accuracy = 0.0
    best_model_state = None
    history = []

    # 准备日志文件
    log_file = None
    if log_enabled:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"training_{timestamp}.csv")
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_accuracy', 'learning_rate'])

    print("开始训练...")
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0

        # 训练进度条
        if TQDM_AVAILABLE:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        else:
            train_iter = train_loader

        for batch_idx, (data, target) in enumerate(train_iter):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 更新进度条显示
            if TQDM_AVAILABLE:
                train_iter.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        accuracy = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'test_accuracy': accuracy,
            'learning_rate': current_lr
        })

        # 写入日志
        if log_file:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, accuracy, current_lr])

        # 显示训练信息
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_train_loss:.4f}, "
              f"Test Acc: {accuracy:.2f}%, LR: {current_lr:.6f}")

        # 学习率调度
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(accuracy)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  -> 学习率调整: {old_lr:.6f} -> {new_lr:.6f}")

        # 保存最优模型
        if save_best and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            print(f"  -> 发现更优模型 (准确率: {accuracy:.2f}%), 已记录")

        # Early Stopping 检查
        if early_stopping:
            if early_stopping(accuracy):
                print(f"\nEarly Stopping触发！连续{early_stopping_patience}个epoch未改善")
                break

    # 恢复最优模型
    if save_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n训练完成! 最佳测试准确率: {best_accuracy:.2f}%")
        if log_file:
            print(f"训练日志已保存到: {log_file}")

        # 保存最优模型
        save_model_with_accuracy(model, model_path, best_accuracy)
    else:
        print(f"\n训练完成! 最终测试准确率: {accuracy:.2f}%")

    return model


# 5. 模型保存与加载
def save_model(model, path):
    """保存模型到文件（不含准确率信息）"""
    print(f"正在保存模型到 {path} ...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': HIDDEN_SIZE,
    }, path)
    print("模型保存完成！")


def save_model_with_accuracy(model, path, accuracy):
    """保存模型到文件（包含准确率信息）"""
    print(f"正在保存最优模型到 {path} ...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_size': HIDDEN_SIZE,
        'test_accuracy': accuracy,
    }, path)
    print(f"最优模型保存完成！测试准确率: {accuracy:.2f}%")


def load_model(path):
    """从文件加载模型"""
    print(f"正在从 {path} 加载模型...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"模型文件不存在: {path}")

    checkpoint = torch.load(path)
    model = SimpleMLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 显示准确率信息（如果有）
    if 'test_accuracy' in checkpoint:
        print(f"模型加载完成！测试准确率: {checkpoint['test_accuracy']:.2f}%")
    else:
        print("模型加载完成！")

    return model


# 4. 量化并导出为 C 头文件
def export_weights_to_c(model, filename):
    print(f"正在导出权重到 {filename} ...")

    # 提取权重（确保先移到CPU再转换为numpy）
    weights1 = model.fc1.weight.detach().cpu().numpy()  # shape (32, 784)
    bias1 = model.fc1.bias.detach().cpu().numpy()      # shape (32,)
    weights2 = model.fc2.weight.detach().cpu().numpy()  # shape (10, 32)
    bias2 = model.fc2.bias.detach().cpu().numpy()      # shape (10,)

    # --- 改进的量化策略 ---
    # 使用对称量化：自动计算每层的最优缩放因子
    # 确保最大程度利用INT8范围 [-127, 127]，同时避免溢出

    def quantize_symmetric(arr, name=""):
        """
        对称量化：基于实际权重范围动态计算缩放因子
        Args:
            arr: 待量化的浮点数组
            name: 参数名称（用于日志）
        Returns:
            quantized: INT8量化后的数组
            scale: 缩放因子（记录在注释中，用于推理时反量化）
        """
        # 计算绝对值最大值
        abs_max = np.max(np.abs(arr))

        if abs_max < 1e-8:  # 避免除零
            print(f"  警告: {name} 接近零，使用默认缩放")
            return np.zeros_like(arr, dtype=np.int8), 1.0

        # 计算缩放因子：使最大值映射到127
        scale = 127.0 / abs_max

        # 量化并截断到INT8范围
        quantized = np.clip(np.round(arr * scale), -127, 127).astype(np.int8)

        # 计算量化误差
        dequantized = quantized / scale
        error = np.mean(np.abs(arr - dequantized))

        print(f"  {name}: scale={scale:.2f}, abs_max={abs_max:.6f}, avg_error={error:.6f}")

        return quantized, scale

    print("正在量化权重...")
    q_w1, scale_w1 = quantize_symmetric(weights1, "W1 (Layer1 Weights)")
    q_b1, scale_b1 = quantize_symmetric(bias1, "B1 (Layer1 Biases)")
    q_w2, scale_w2 = quantize_symmetric(weights2, "W2 (Layer2 Weights)")
    q_b2, scale_b2 = quantize_symmetric(bias2, "B2 (Layer2 Biases)")

    # 生成 C 代码内容
    c_content = f"""
#ifndef MODEL_WEIGHTS_H
#define MODEL_WEIGHTS_H

#include <stdint.h>

// Network Config
#define INPUT_SIZE 784
#define HIDDEN_SIZE {HIDDEN_SIZE}
#define OUTPUT_SIZE 10

// Quantization Scales (for reference and dequantization)
// NOTE: INT8_value = float_value * scale
//       float_value = INT8_value / scale
#define SCALE_W1 {scale_w1:.6f}f
#define SCALE_B1 {scale_b1:.6f}f
#define SCALE_W2 {scale_w2:.6f}f
#define SCALE_B2 {scale_b2:.6f}f

// Layer 1 Weights ({HIDDEN_SIZE} x 784)
static const int8_t W1[{HIDDEN_SIZE}][784] = {{
"""
    # 写入 W1
    for i in range(HIDDEN_SIZE):
        c_content += "    {" + ", ".join(map(str, q_w1[i])) + "},\n"

    c_content += "};\n\n// Layer 1 Biases\nstatic const int8_t B1[] = {"
    c_content += ", ".join(map(str, q_b1)) + "};\n"

    c_content += f"\n// Layer 2 Weights (10 x {HIDDEN_SIZE})\nstatic const int8_t W2[10][{HIDDEN_SIZE}] = {{\n"

    # 写入 W2
    for i in range(10):
        c_content += "    {" + ", ".join(map(str, q_w2[i])) + "},\n"

    c_content += "};\n\n// Layer 2 Biases\nstatic const int8_t B2[] = {"
    c_content += ", ".join(map(str, q_b2)) + "};\n"

    c_content += "\n#endif // MODEL_WEIGHTS_H\n"

    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        f.write(c_content)

    print("导出完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNIST MLP 训练与导出工具 (优化版)")
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'export', 'test', 'all'],
                        help='执行模式: train(仅训练), export(仅导出), test(测试模型), all(完整流程)')
    parser.add_argument('--model-path', type=str, default=MODEL_PATH,
                        help=f'模型保存/加载路径 (默认: {MODEL_PATH})')
    parser.add_argument('--export-path', type=str, default=EXPORT_PATH,
                        help=f'C头文件导出路径 (默认: {EXPORT_PATH})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'训练轮数 (默认: {EPOCHS})')
    parser.add_argument('--no-save-best', action='store_true',
                        help='训练时不自动保存最优模型（默认会保存）')
    parser.add_argument('--no-scheduler', action='store_true',
                        help='禁用学习率调度器（默认启用）')
    parser.add_argument('--early-stop', type=int, default=7,
                        help='Early Stopping容忍epoch数，0表示禁用 (默认: 7)')
    parser.add_argument('--no-log', action='store_true',
                        help='禁用训练日志记录（默认启用）')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='禁用训练数据增强（默认启用）')

    args = parser.parse_args()

    # 更新全局配置
    EPOCHS = args.epochs

    if args.mode == 'train':
        # 仅训练模式
        print("=" * 50)
        print("模式: 仅训练")
        print("=" * 50)
        trained_model = train(
            save_best=not args.no_save_best,
            model_path=args.model_path,
            use_scheduler=not args.no_scheduler,
            early_stopping_patience=args.early_stop,
            log_enabled=not args.no_log,
            augmentation=not args.no_augmentation
        )

    elif args.mode == 'export':
        # 仅导出模式
        print("=" * 50)
        print("模式: 仅导出")
        print("=" * 50)
        model = load_model(args.model_path)
        export_weights_to_c(model, args.export_path)

    elif args.mode == 'test':
        # 仅测试模式
        print("=" * 50)
        print("模式: 测试模型")
        print("=" * 50)
        model = load_model(args.model_path)
        _, test_loader = get_data_loaders(augmentation=False)  # 测试时不使用数据增强
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        accuracy = evaluate(model, test_loader, device)
        print(f"测试准确率: {accuracy:.2f}%")

    else:  # args.mode == 'all'
        # 完整流程（默认）
        print("=" * 50)
        print("模式: 完整流程 (训练 + 导出)")
        print("=" * 50)
        trained_model = train(
            save_best=not args.no_save_best,
            model_path=args.model_path,
            use_scheduler=not args.no_scheduler,
            early_stopping_patience=args.early_stop,
            log_enabled=not args.no_log,
            augmentation=not args.no_augmentation
        )
        export_weights_to_c(trained_model, args.export_path)

    print("\n任务完成！")
