# MNIST 多模型训练与导出工具（重构版 v2.0）

> ⚠️ **过时文档警告**
>
> 本文档描述的是 Nios II 时代的训练与导出架构，已不适用于当前 **v1.1 硬件加速器**版本。
>
> **当前主线工作流请参考**：
> - [QUICKSTART.md](QUICKSTART.md) - 快速入门指南
> - [README.md](README.md) - 完整工具链文档
>
> 本文档仅作历史参考保留。
>
> **最后适用版本**：v1 (Nios II + SimpleMLP)
> **文档状态**：🔒 Legacy / Read-only

---

## 架构说明

本工具已完全重构为**模块化多模型架构**，支持轻松切换和扩展不同的神经网络模型。

```
model_tools/
├── train.py              # 通用训练脚本（解耦）
├── export.py             # 通用导出脚本（解耦）
├── v1/
│   └── train_export.py   # 旧版脚本（保留用于兼容）
├── models/               # 模型定义目录
│   ├── __init__.py       # 模型注册系统
│   ├── SimpleMLP.py      # 2层MLP模型
│   └── TinyLeNet.py      # Tiny-LeNet CNN模型
├── trained_models/       # 保存训练好的模型
├── data/                 # MNIST数据集
└── logs/                 # 训练日志
```

## 快速开始

### 1. 查看可用模型

```bash
python train.py --list-models
```

输出：
```
Available models:
----------------------------------------------------------------------
  SimpleMLP       | mlp  | 简单的2层MLP (784->32->10)
                  | Params: ~25K
  TinyLeNet       | cnn  | Tiny-LeNet CNN (C1->S2->C3->S4->FC)
                  | Params: ~11K
----------------------------------------------------------------------
```

### 2. 训练模型

**训练MLP：**
```bash
python train.py --model SimpleMLP --epochs 10
```

**训练TinyLeNet：**
```bash
python train.py --model TinyLeNet --epochs 20
```

**完整参数示例：**
```bash
python train.py \
    --model TinyLeNet \
    --epochs 20 \
    --batch-size 512 \
    --early-stop 7 \
    --no-augmentation
```

### 3. 导出模型

**自动导出最新模型：**
```bash
python export.py --latest
```

**导出指定模型：**
```bash
python export.py --model-path trained_models/TinyLeNet_20250124_143022_acc98.45.pth
```

**列出所有已训练模型：**
```bash
python export.py --list
```

## 模型说明

### SimpleMLP

**结构：**
- 输入：28x28 = 784
- 隐藏层：32神经元 + ReLU
- 输出：10类别
- 参数量：~25K

**特点：**
- 简单快速，适合EP4CE10资源
- 准确率：~95-97%
- 推理速度快

**导出格式：**
- C头文件：`model_weights.h`
- 包含：W1, B1, W2, B2（INT8量化）

### TinyLeNet

**结构（参考 TinyLeNet_fpga.md）：**
```
输入 (1@28x28)
  ↓
Conv1 (6@24x24, 5x5 kernel)
  ↓
MaxPool (6@12x12, 2x2 pool)
  ↓
Conv2 (16@8x8, 5x5 kernel)
  ↓
MaxPool (16@4x4, 2x2 pool)
  ↓
FC1 (256->32)
  ↓
FC2 (32->10)
```

**特点：**
- 卷积神经网络，准确率可达97-99%
- 需要FPGA硬件加速器（Line Buffer + MAC Array）
- 参数量：~11K（完全可以放在片上ROM）

**导出格式：**
- C头文件：`tinylenet_weights.h`
- 包含：CONV1/2权重，FC1/2权重（INT8量化）
- 卷积核展平为一维数组，便于硬件读取

## 命令行参数

### train.py

```
--model <name>          模型名称 (SimpleMLP | TinyLeNet)
--list-models           列出所有可用模型
--epochs <N>            训练轮数（默认：10）
--batch-size <N>        批次大小（默认：1024）
--no-scheduler          禁用学习率调度器
--early-stop <N>        Early Stopping容忍度（默认：7）
--no-log                禁用训练日志
--no-augmentation       禁用数据增强
```

### export.py

```
--model-path <path>     指定模型路径
--output <path>         输出文件路径
--latest                自动使用最新训练的模型
--list                  列出所有已训练的模型
```

## 扩展新模型

### 步骤1：创建模型文件

在 `models/` 目录下创建新的模型文件，例如 `MyModel.py`：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义网络结构
        ...

    def forward(self, x):
        # 定义前向传播
        ...
```

### 步骤2：注册模型

编辑 `models/__init__.py`，添加：

```python
from .MyModel import MyModel

MODEL_REGISTRY = {
    ...
    'MyModel': {
        'class': MyModel,
        'type': 'custom',  # mlp | cnn | custom
        'description': '我的自定义模型',
        'input_shape': (1, 28, 28),
        'params': '~XXK',
    },
}
```

### 步骤3：（可选）自定义导出器

如果需要特殊导出格式，在 `export.py` 中添加新的导出函数。

## TinyLeNet FPGA实现路径

参考 `docs/TinyLeNet_fpga.md`，TinyLeNet在FPGA上的实现包括：

1. **训练和导出（已完成）**：
   ```bash
   python train.py --model TinyLeNet --epochs 20
   python export.py --latest --output ../software/app/tinylenet_weights.h
   ```

2. **硬件加速器设计（Verilog）**：
   - Line Buffer（行缓存）：存储5行像素，支持5x5卷积窗口
   - MAC Array：并行乘累加阵列
   - Avalon-MM Slave接口：与Nios II通信

3. **软件调度（C代码）**：
   - Nios II负责数据搬运和流程控制
   - 调用硬件加速器完成卷积计算
   - 软件实现Pooling和全连接层

## 训练优化特性（继承v2.0）

所有v2.0优化特性在新架构中完全保留：

- 学习率自适应调度（ReduceLROnPlateau）
- Early Stopping
- 数据增强（RandomAffine）
- 训练进度条（tqdm）
- CSV日志记录
- 对称量化算法
- GPU自动检测

## 与旧版兼容性

- 旧版 `v1/train_export.py` 保留，可继续使用
- 新版架构与旧版模型文件格式完全兼容
- 可以使用 `export.py` 导出旧版训练的模型

## 常见工作流

### 场景1：MLP快速验证

```bash
# 训练MLP
python train.py --model SimpleMLP --epochs 5

# 导出最新模型
python export.py --latest

# 在PC上测试（假设你有test程序）
cd ../software/app
gcc main.c -o mnist_test && ./mnist_test
```

### 场景2：TinyLeNet完整训练

```bash
# 训练20个epoch
python train.py --model TinyLeNet --epochs 20

# 查看所有模型并选择最佳
python export.py --list

# 导出最佳模型
python export.py --model-path trained_models/TinyLeNet_xxx_acc98.45.pth \
    --output ../software/app/tinylenet_weights.h
```

### 场景3：对比不同模型

```bash
# 训练多个模型
python train.py --model SimpleMLP --epochs 10
python train.py --model TinyLeNet --epochs 20

# 对比准确率（查看日志或模型文件名）
python export.py --list
```

## 技术细节

### 模型保存格式

训练后的模型保存为 `.pth` 文件，包含：
- `model_name`: 模型名称
- `model_state_dict`: 模型参数
- `test_accuracy`: 测试准确率
- `model_type`: 模型类型（mlp/cnn）

### 量化算法

使用**对称量化**：
```
scale = 127 / max(abs(weights))
quantized = round(weights * scale).clip(-127, 127)
```

每层权重和偏置独立计算缩放因子，量化误差<0.5%。

### 导出格式对比

| 模型类型 | 输出文件 | 主要内容 |
|---------|---------|---------|
| MLP | model_weights.h | W1, B1, W2, B2 |
| CNN | tinylenet_weights.h | CONV1/2权重, FC1/2权重 |

## 依赖安装

```bash
pip install torch torchvision numpy tqdm
```

## 故障排查

**问题1：找不到模型**
```
解决：运行 python train.py --list-models 查看可用模型
```

**问题2：导出失败**
```
解决：确保先训练模型，使用 python export.py --list 检查
```

**问题3：训练速度慢**
```
解决：增大batch-size或使用GPU（自动检测CUDA）
```

## 下一步

1. 训练TinyLeNet并达到98%+准确率
2. 参考 `docs/TinyLeNet_fpga.md` 实现硬件加速器
3. 在Qsys中集成加速器
4. 编写Nios II调度代码
5. 在EP4CE10上验证

---

**版本**：v3.0（模块化重构版）
**作者**：ZCF
**日期**：2025-01-24
