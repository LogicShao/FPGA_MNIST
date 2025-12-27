"""
通用训练脚本 - 支持多模型训练
解耦版：专注于训练功能，导出功能移至export.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import os
import csv
from datetime import datetime

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# 导入模型注册系统
from models import get_model, get_model_info, list_models, MODEL_REGISTRY

# 配置
BATCH_SIZE = 1024
EPOCHS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOG_DIR = "./logs"
MODELS_DIR = "./trained_models"  # 保存训练好的模型


# Early Stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
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


# 数据加载
def get_data_loaders(augmentation=True, batch_size=BATCH_SIZE):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if augmentation:
        print("数据增强: 已启用 (随机旋转±10°, 随机平移±10%, 随机缩放0.9-1.1)")
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("数据增强: 已禁用")
        train_transform = test_transform

    train_dataset = datasets.MNIST(DATA_DIR, train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 模型评估
def evaluate(model, test_loader, device='cpu'):
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


# 训练流程
def train(model_name, epochs=EPOCHS, batch_size=BATCH_SIZE,
          use_scheduler=True, early_stopping_patience=7,
          log_enabled=True, augmentation=True):

    print(f"\n{'='*60}")
    print(f"开始训练模型: {model_name}")
    print(f"{'='*60}\n")

    # 获取模型
    model_class = get_model(model_name)
    model_info = get_model_info(model_name)

    print("正在准备数据...")
    train_loader, test_loader = get_data_loaders(augmentation=augmentation, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"模型类型: {model_info['type'].upper()}")
    print(f"模型描述: {model_info['description']}\n")

    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    ) if use_scheduler else None

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, min_delta=0.01, mode='max'
    ) if early_stopping_patience > 0 else None

    best_accuracy = 0.0
    best_model_state = None

    # 日志文件
    log_file = None
    if log_enabled:
        os.makedirs(LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(LOG_DIR, f"{model_name}_{timestamp}.csv")
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_accuracy', 'learning_rate'])

    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0

        if TQDM_AVAILABLE:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        else:
            train_iter = train_loader

        for data, target in train_iter:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if TQDM_AVAILABLE:
                train_iter.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        accuracy = evaluate(model, test_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # 写入日志
        if log_file:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, accuracy, current_lr])

        # 显示训练信息
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, "
              f"Test Acc: {accuracy:.2f}%, LR: {current_lr:.6f}")

        # 学习率调度
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(accuracy)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  -> 学习率调整: {old_lr:.6f} -> {new_lr:.6f}")

        # 保存最优模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict().copy()
            print(f"  -> 发现更优模型 (准确率: {accuracy:.2f}%), 已记录")

        # Early Stopping 检查
        if early_stopping:
            if early_stopping(accuracy):
                print(f"\nEarly Stopping触发！连续{early_stopping_patience}个epoch未改善")
                break

    # 恢复最优模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n训练完成! 最佳测试准确率: {best_accuracy:.2f}%")
        if log_file:
            print(f"训练日志已保存到: {log_file}")

        # 保存最优模型
        os.makedirs(MODELS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}_acc{best_accuracy:.2f}.pth")

        torch.save({
            'model_name': model_name,
            'model_state_dict': model.state_dict(),
            'test_accuracy': best_accuracy,
            'model_type': model_info['type'],
        }, model_path)
        print(f"最优模型已保存到: {model_path}")

        return model, model_path, best_accuracy

    return model, None, best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST 模型训练脚本 (多模型支持)")

    parser.add_argument('--model', type=str, default='SimpleMLP',
                        help='模型名称 (SimpleMLP | TinyLeNet)')
    parser.add_argument('--list-models', action='store_true',
                        help='列出所有可用模型')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'训练轮数 (默认: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'批次大小 (默认: {BATCH_SIZE})')
    parser.add_argument('--no-scheduler', action='store_true',
                        help='禁用学习率调度器')
    parser.add_argument('--early-stop', type=int, default=7,
                        help='Early Stopping容忍度 (默认: 7, 0表示禁用)')
    parser.add_argument('--no-log', action='store_true',
                        help='禁用训练日志记录')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='禁用数据增强')

    args = parser.parse_args()

    if args.list_models:
        list_models()
        exit(0)

    # 开始训练
    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_scheduler=not args.no_scheduler,
        early_stopping_patience=args.early_stop,
        log_enabled=not args.no_log,
        augmentation=not args.no_augmentation
    )

    print("\n任务完成！")
