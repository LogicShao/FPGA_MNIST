# 快速开始指南

## 新架构快速上手（5分钟）

### 1. 查看可用模型

```bash
cd model_tools
python train.py --list-models
```

### 2. 训练SimpleMLP（推荐初学者）

```bash
# 快速训练（3个epoch，约1分钟）
python train.py --model SimpleMLP --epochs 3

# 完整训练（10个epoch，约3分钟）
python train.py --model SimpleMLP --epochs 10
```

### 3. 导出模型到C代码

```bash
# 自动导出最新训练的模型
python export.py --latest
```

生成的文件：`../software/app/model_weights.h`

### 4. 在PC上测试推理

```bash
cd ../software/app
gcc main.c -o mnist_test
./mnist_test
```

---

## 进阶：训练TinyLeNet CNN

TinyLeNet是为FPGA设计的轻量级CNN，准确率可达98%+

### 1. 训练TinyLeNet

```bash
cd model_tools
python train.py --model TinyLeNet --epochs 20
```

训练约5-10分钟（GPU）或15-30分钟（CPU）

### 2. 导出CNN模型

```bash
python export.py --latest --output ../software/app/tinylenet_weights.h
```

### 3. 实现FPGA加速器

参考文档：`docs/TinyLeNet_fpga.md`

需要实现：
- Line Buffer（行缓存）
- MAC Array（乘累加阵列）
- Avalon-MM接口

---

## 常用命令速查

```bash
# 列出所有可用模型
python train.py --list-models

# 训练指定模型
python train.py --model <ModelName> --epochs <N>

# 列出所有已训练模型
python export.py --list

# 导出指定模型
python export.py --model-path trained_models/xxx.pth

# 导出最新模型
python export.py --latest
```

---

## 新旧版本对比

| 功能 | 旧版 (v1/train_export.py) | 新版 (train.py + export.py) |
|------|------------------------|----------------------------|
| 模型切换 | 需修改代码 | 命令行参数 --model |
| 训练导出 | 耦合在一起 | 完全分离 |
| 模型管理 | 覆盖式保存 | 保留所有版本 |
| 扩展性 | 困难 | 模块化注册系统 |

---

## 故障排查

**Q: 提示找不到模型**

A: 确保在 `model_tools/` 目录下运行命令

**Q: 导出时提示没有模型**

A: 先运行 `python train.py --model xxx` 训练模型

**Q: 训练速度很慢**

A:
1. 增大batch-size：`--batch-size 2048`
2. 使用GPU（自动检测CUDA）
3. 减少epoch数

---

## 下一步学习

1. 阅读 `README_v3.md` 了解完整架构
2. 阅读 `docs/TinyLeNet_fpga.md` 了解FPGA实现
3. 尝试修改 `models/SimpleMLP.py` 创建自定义模型
4. 查看 `logs/` 目录下的训练日志分析训练过程

---

**祝你成功！如有问题，请参考 README_v3.md 完整文档。**
