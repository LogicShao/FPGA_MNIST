# model_tools - FPGA_MNIST 工具链

> Python 工具集，用于模型训练、量化参数导出、硬件权重生成、仿真验证与串口通信

**🚀 新手入门**：如果你是第一次使用，建议先阅读 [QUICKSTART.md](QUICKSTART.md)（5 分钟上手）

---

## 📋 目录

1. [工具概览](#工具概览)
2. [量化策略详解](#量化策略详解)
3. [常用命令](#常用命令)
4. [工作流程](#工作流程)
5. [文件说明](#文件说明)

---

## 工具概览

本目录包含完整的软硬协同工作流工具链：

| 工具脚本 | 功能 | 输出文件 |
|---------|------|---------|
| `calc_quant_params.py` | 计算 INT8 量化参数 | `quant_params.json` |
| `quantize_bias.py` | 生成 INT32 偏置 ROM | `rom_FC*_BIASES_int32.mif` |
| `export_test_img.py` | 导出量化后的测试图像 | `test_image.mem` |
| `hw_ref.py` | 硬件等效参考推理 | 终端输出（准确率） |
| `batch_sim.py` | 批量 RTL 仿真 | `batch_sim_results.csv` |
| `send_image.py` | 串口发送图像到 FPGA | UART 通信 |
| `train_log_plot.py` | 绘制训练曲线 | PNG 图表 |

---

## 量化策略详解

### 基本原理

本项目采用 **对称量化（Symmetric Quantization）** 策略，将浮点数映射到 INT8 范围 `[-128, 127]`：

```
量化公式:   q = clamp(round(r / scale), -128, 127)
反量化公式: r = q * scale
```

其中：
- `r`：浮点数（Real Number）
- `q`：量化整数（Quantized Integer）
- `scale`：缩放因子（Scale Factor）

### 多层级量化

神经网络中每一层都有独立的量化参数：

```
输入数据 (x)      --[s_in]-->   INT8
权重 (w)         --[s_w]-->    INT8
激活值 (act)     --[s_out]-->  INT8
```

**核心挑战**：卷积/全连接运算后的累加结果需要重新量化回 INT8：

```
acc = Σ(x_q[i] * w_q[i])        # 累加器（64-bit）
acc_real ≈ acc * (s_in * s_w)   # 反量化到浮点域
y_q = acc_real / s_out          # 量化到输出域
```

### 定点近似（Fixed-Point Approximation）

为避免浮点除法，采用整数移位近似：

```verilog
eff = s_out / (s_in * s_w)
mult = round(eff * 2^shift)     # shift 通常为 20-24

// 硬件实现
acc_scaled = (acc * mult + round_bias) >> shift
y_q = clamp(acc_scaled, -128, 127)
```

**示例**（Layer1 量化参数）：
```json
{
  "s_in": 0.003921568859368563,
  "s_w": 0.004626067914068699,
  "s_out": 0.0069256257265806,
  "mult": 381851,              # eff * 2^20 ≈ 381851
  "shift": 20
}
```

### 权重与激活值量化

#### 权重量化（Weight Quantization）
```python
def quantize_weights(W_float, s_w):
    W_int8 = np.clip(np.round(W_float / s_w), -128, 127).astype(np.int8)
    return W_int8
```

#### 激活值量化（Activation Quantization）
- **输入层**：MNIST 图像已归一化至 `[0, 1]`，`s_in = 1/255 ≈ 0.003922`
- **中间层**：通过统计训练集的激活值范围动态计算 `s_out`
- **ReLU 影响**：激活后范围变为 `[0, 127]`（负值被截断）

### 偏置（Bias）处理

偏置需要匹配累加器的尺度，因此量化为 **INT32**：

```python
bias_q = round(bias_float / (s_in * s_w))  # INT32
```

在硬件中，偏置直接加到 64-bit 累加器上：
```verilog
acc = Σ(x[i] * w[i]) + bias_q
```

---

## 常用命令

### 1. 量化参数计算（normalize on）

**作用**：分析模型权重与激活值分布，生成 `quant_params.json`

```bash
python calc_quant_params.py --normalize
```

**输出示例**（`quant_params.json`）：
```json
{
  "input": {"scale": 0.003921568859368563},
  "conv1": {
    "s_in": 0.003921568859368563,
    "s_w": 0.004626067914068699,
    "s_out": 0.0069256257265806,
    "mult": 381851,
    "shift": 20
  },
  ...
}
```

### 2. 生成 INT32 偏置 ROM

**作用**：将浮点偏置量化为 INT32 并生成 `.mif` 初始化文件

```bash
python quantize_bias.py \
    --quant-params quant_params.json \
    --out-dir ../hardware/src/v1.1/rtl/weights
```

**输出文件**：
- `rom_FC1_BIASES_int32.mif`（32 个 INT32 值）
- `rom_FC2_BIASES_int32.mif`（10 个 INT32 值）

### 3. 导出测试图像

**作用**：从 MNIST 测试集中提取一张图像，应用 normalize 和量化，生成 `.mem` 文件

```bash
python export_test_img.py \
    --normalize \
    --quant-params quant_params.json
```

**输出**：`../hardware/src/v1.1/tb/test_image.mem`（784 行十六进制）

### 4. 硬件等效参考推理

**作用**：使用与 FPGA 完全相同的量化逻辑执行推理，用于验证 RTL 实现

```bash
# 单张图像推理
python hw_ref.py \
    --image ../hardware/src/v1.1/tb/test_image.mem \
    --weights ../hardware/src/v1.1/rtl/weights \
    --quant-params quant_params.json

# 批量测试（评估准确率）
python hw_ref.py \
    --batch \
    --count 200 \
    --normalize \
    --quant-params quant_params.json \
    --data-dir data
```

**输出示例**：
```
Conv1 q[0]: -22  # 第一个输出通道的第一个像素
...
Predicted: 7, Label: 7, Match: True
Accuracy: 199/200 = 99.50%
```

### 5. 批量 RTL 仿真

**作用**：自动化运行大量 RTL 仿真，生成准确率报告

```bash
# 快速验证（20 张）
python batch_sim.py \
    --count 20 \
    --normalize \
    --quant-params quant_params.json \
    --quiet

# 完整测试集（10000 张，约需数小时）
python batch_sim.py \
    --count 10000 \
    --normalize \
    --quant-params quant_params.json \
    --quiet
```

**输出**：`batch_sim_results.csv`
```csv
index,label,pred,match
0,7,7,True
1,2,2,True
...
```

### 6. 串口发送图像（上板验证）

**作用**：通过 UART 将 MNIST 图像发送到 FPGA，接收推理结果

```bash
python send_image.py
```

**交互流程**：
```
1) MNIST image (选择测试集图像)
2) Custom file (选择自定义文件)
> 1
Enter image index (0-9999): 42
Sending image #42 (label: 3)...
FPGA Response: [10 bytes] + [4 bytes cycle count]
Predicted: 3, Inference time: 10.031 ms
```

---

## 工作流程

### 完整开发流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: 模型训练与量化参数生成                                    │
└─────────────────────────────────────────────────────────────────┘
   1. train_pytorch.py          # 训练模型 (FP32)
   2. calc_quant_params.py      # 计算量化参数 → quant_params.json
   3. quantize_bias.py          # 生成 INT32 偏置 ROM

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: 权重导出与 RTL 集成                                       │
└─────────────────────────────────────────────────────────────────┘
   4. export_weights.py         # 生成权重 .mif 文件
   5. 将 .mif 文件拷贝到 hardware/src/v1.1/rtl/weights/
   6. 使用 Quartus 综合 RTL 代码

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: 仿真验证（软硬对齐）                                      │
└─────────────────────────────────────────────────────────────────┘
   7. export_test_img.py        # 生成测试图像
   8. hw_ref.py --batch         # Python 参考推理（Golden）
   9. batch_sim.py              # RTL 批量仿真
  10. 对比 Python 与 RTL 输出，确保层级一致

┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4: 上板验证                                                  │
└─────────────────────────────────────────────────────────────────┘
  11. 下载 .sof 到 FPGA
  12. send_image.py             # 通过 UART 测试真实硬件
```

---

## 文件说明

### 量化与权重工具
- `calc_quant_params.py`：核心量化参数计算器
  - 支持 `--normalize` 标志（MNIST 图像归一化）
  - 输出 `quant_params.json`

- `quantize_bias.py`：偏置量化工具
  - 读取 `quant_params.json` 中的 `s_in * s_w`
  - 生成 INT32 格式的 `.mif` 文件

- `export_weights.py`：权重导出工具
  - 将 PyTorch 权重量化为 INT8
  - 生成 Quartus 可识别的 `.mif` 格式

### 测试与验证工具
- `export_test_img.py`：测试图像生成器
  - 从 MNIST 数据集中提取图像
  - 应用与硬件一致的 normalize + 量化

- `hw_ref.py`：硬件等效参考模型
  - **关键作用**：提供 Golden Reference
  - 与 FPGA 使用完全相同的量化逻辑
  - 用于验证 RTL 实现的正确性

- `batch_sim.py`：批量仿真自动化脚本
  - 循环调用 Icarus Verilog 仿真
  - 收集预测结果并统计准确率
  - 支持 `--debug-mismatch` 自动保存错误样本

### 板级工具
- `send_image.py`：UART 串口通信工具
  - 发送量化后的图像数据（784 字节）
  - 接收 FPGA 回传的结果（14 字节）
  - 解析推理时间（周期数 → 毫秒）

### 可视化工具
- `train_log_plot.py`：训练日志可视化
  - 绘制 Loss / Accuracy / Learning Rate 曲线
  - 输出 PNG 图表用于报告

---

## 数据目录

MNIST 数据集位于：`model_tools/data/`

目录结构：
```
data/
├── t10k-images-idx3-ubyte    # 测试集图像（10000 张）
├── t10k-labels-idx1-ubyte    # 测试集标签
├── train-images-idx3-ubyte   # 训练集图像（60000 张）
└── train-labels-idx1-ubyte   # 训练集标签
```

**数据集下载**：
- 官方来源：[MNIST Database](http://yann.lecun.com/exdb/mnist/)
- 镜像站点：通过 PyTorch `torchvision.datasets.MNIST` 自动下载

---

## 常见问题排查

### Q1: `quant_params.json` 不存在
**原因**：未运行量化参数计算
**解决**：
```bash
python calc_quant_params.py --normalize
```

### Q2: RTL 仿真结果与 `hw_ref.py` 不一致
**排查步骤**：
1. 确认使用相同的 `quant_params.json`
2. 检查权重文件是否最新（重新运行 `quantize_bias.py`）
3. 使用 `--debug-mismatch` 保存失败样本：
   ```bash
   python batch_sim.py --count 10 --debug-mismatch
   ```
4. 手动运行单张图像，对比每层输出：
   ```bash
   # Python 参考
   python hw_ref.py --image test_image.mem --weights ...

   # RTL 仿真（查看日志）
   python ../hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core
   ```

### Q3: 串口无响应
**检查清单**：
- [ ] 串口号正确（`send_image.py` 中的 `SERIAL_PORT`）
- [ ] 波特率匹配（115200）
- [ ] FPGA 已下载 `.sof` 文件
- [ ] USB-UART 驱动已安装

### Q4: 准确率低于预期
**可能原因**：
- 量化参数计算时未启用 `--normalize`
- 权重文件版本不匹配
- RTL 代码中的量化参数未更新（检查 `quant_params.vh`）

---

## 高级用法

### 批量仿真加速模式

**FAST_SIM 模式**（跳过真实计算，仅验证控制流）：
```bash
python batch_sim.py --count 100 --fast --quiet
```
⚠️ **警告**：此模式输出结果不准确，仅用于波形检查！

### 静默模式（关闭日志与波形）
```bash
python batch_sim.py --count 10000 --quiet
```
- 关闭 VCD 波形生成
- 减少终端输出
- 适合大规模准确率评估

### 自定义测试图像
```bash
# 导出指定索引的图像
python export_test_img.py --index 1234 --normalize --quant-params quant_params.json

# 通过串口发送
python send_image.py
> 1  # 选择 MNIST image
> 1234
```

---

## 技术参考

### 量化精度分析

| 模型版本 | 测试集准确率 | 与浮点差异 |
|---------|------------|-----------|
| PyTorch FP32 | 99.00% | - |
| INT8 量化（Python） | 98.85% | -0.15% |
| INT8 硬件（FPGA） | 98.71% | -0.29% |

**结论**：INT8 量化在 MNIST 任务上几乎无损，硬件实现的额外损失主要来自定点近似误差。

### 量化参数推荐值

| 参数 | 推荐范围 | 说明 |
|------|---------|------|
| `shift` | 20-24 | 越大精度越高，但需确保不溢出 |
| `s_in` | 1/255 ≈ 0.00392 | MNIST 归一化标准 |
| `s_w` | 根据权重分布 | 由 `max(abs(W))` 计算 |
| `s_out` | 根据激活值分布 | 统计训练集激活值范围 |

---

**更新时间**：2026-01-04
