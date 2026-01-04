# FPGA_MNIST - 基于 FPGA 的 MNIST 手写数字识别系统

[![Status](https://img.shields.io/badge/status-completed-brightgreen)]()
[![Platform](https://img.shields.io/badge/platform-Cyclone_IV_EP4CE10-blue)]()
[![Accuracy](https://img.shields.io/badge/accuracy-98.71%25-orange)]()

> 在资源仅有 10K 逻辑单元的入门级 FPGA 上实现端到端 CNN 推理加速器

版本状态: **Completed / Pure Verilog**
说明: 主线为纯 Verilog 硬件加速器，Nios II 版本仅为早期测试。

---

## 🎯 项目亮点

### 核心特性
- **极限资源利用**：在 EP4CE10（10K LEs）上实现完整 CNN，资源利用率达 **97.15%**
- **高精度量化**：INT8 定点量化，精度损失仅 **0.29%**（浮点 99.00% → 量化 98.71%）
- **低延迟推理**：纯计算时间 **10.031 ms**，端到端（含串口）**77.885 ms**
- **软硬协同**：Python 训练量化 + Verilog 硬件加速的完整工作流

### 技术实现

#### 网络架构

本设计采用 TinyLeNet（简化版 LeNet-5）网络架构：

<div align="center">
<img src="assets/simpleLeNet-nn.svg" alt="TinyLeNet 网络拓扑结构" width="90%">
<p><i>图 1: TinyLeNet 网络拓扑结构 - 从左至右依次为输入层、两层卷积池化层及两层全连接层</i></p>
</div>

**网络层级详解**：
- **Layer1**: Conv 6@5×5 + ReLU + Pool (28×28 → 12×12×6)
- **Layer2**: Conv 16@5×5 + ReLU + Pool (12×12×6 → 4×4×16)
- **Layer3**: FC1 (256 → 32) + ReLU
- **Layer4**: FC2 (32 → 10)

#### 训练与验证

<div align="center">
<img src="assets/train_curve.png" alt="训练过程曲线" width="80%">
<p><i>图 2: 神经网络训练过程曲线 - 上图为训练损失收敛情况，下图为测试集准确率与学习率的变化关系</i></p>
</div>

**训练策略**：
- 在第 27、36、45 个 Epoch 实施学习率衰减（Learning Rate Decay）
- 每次学习率降低时，测试集准确率均出现提升
- 最终在第 50 个 Epoch 达到 **99.00%** 的浮点精度

#### 硬件架构

- **串行 MAC（乘累加）架构**，资源共享策略
- **64-bit 宽位宽累加器**，防止定点溢出
- **状态机控制的流水线设计**
- **帧缓存 + 滑动窗口**的卷积实现

<div align="center">
<img src="assets/RTL/top_RTL.png" alt="系统顶层 RTL 视图" width="90%">
<p><i>图 3: 系统顶层模块（mnist_system_top）RTL 视图 - 展示 UART 接口与神经网络核心的连接关系</i></p>
</div>

<div align="center">
<img src="assets/RTL/mnist_network_core.png" alt="网络核心 RTL 视图" width="90%">
<p><i>图 4: 神经网络计算核心（mnist_network_core）RTL 视图 - 四层网络模块的级联结构</i></p>
</div>

#### 量化策略

- **8-bit 数据位宽**（输入、权重、激活值）
- **定点量化公式**：`y_q = clamp((acc·mult + round) >> shift, -128, 127)`
- **量化参数统一管理**（`quant_params.vh`）
- **精度保持**：INT8 量化后准确率 98.71%，损失仅 0.29%

### 资源占用（EP4CE10F17C8）
| 资源类型 | 使用量 | 总量 | 利用率 |
|---------|--------|------|--------|
| 逻辑单元 (LCs) | 10,026 | 10,320 | **97.15%** |
| 存储块 (M9K) | 26 | 46 | 56.5% |
| 乘法器 (DSP) | 16 | 23 | 69.5% |

### 性能指标
- **推理延迟**：10.031 ms（纯计算）/ 77.885 ms（含串口传输）
- **时钟频率**：50 MHz（注：当前时序 WNS = -2.745 ns，需降频或增加流水级）
- **UART 波特率**：115200 bps
- **功耗**：低功耗边缘推理方案

---

## 📚 快速入口
- **硬件实现详细文档**: [hardware/src/v1.1/README.md](hardware/src/v1.1/README.md)
- **Python 工具链**: [model_tools/README.md](model_tools/README.md)

---

## 📂 目录结构

```
FPGA_MNIST_Nios/
├── README.md                    # 本文件
├── model_tools/                 # Python 工具链
│   ├── calc_quant_params.py    # 量化参数计算
│   ├── quantize_bias.py        # 偏置量化
│   ├── export_test_img.py      # 测试图像导出
│   ├── hw_ref.py               # 硬件等效参考推理
│   ├── batch_sim.py            # 批量仿真
│   ├── send_image.py           # 串口发送工具
│   └── README.md
├── hardware/src/v1.1/          # 纯 Verilog 主线（推荐）
│   ├── rtl/                    # Verilog 源码
│   │   ├── mnist_system_top.v # 系统顶层
│   │   ├── mnist_network_core.v # 网络核心
│   │   ├── layer1_block.v      # 卷积层 1
│   │   ├── layer2_block.v      # 卷积层 2
│   │   ├── layer3_fc1.v        # 全连接层 1
│   │   ├── layer4_fc2.v        # 全连接层 2
│   │   └── weights/            # 权重 ROM 初始化文件
│   ├── tb/                     # 仿真测试平台
│   ├── script/                 # 仿真脚本
│   └── README.md
├── hardware/src/v1/            # Legacy Nios II 版本
└── software/                   # Legacy Nios II 软件
```

---

## 🚀 快速开始

### 1. 环境准备
- **硬件**：Intel Cyclone IV EP4CE10 开发板
- **软件**：
  - Quartus Prime（综合与布局布线）
  - Icarus Verilog + Surfer（仿真）
  - Python 3.x + PyTorch（模型训练与量化）

### 2. 生成量化参数与权重
```bash
# 计算量化参数（含 normalize）
python model_tools/calc_quant_params.py --normalize

# 生成 INT32 偏置 ROM
python model_tools/quantize_bias.py \
    --quant-params model_tools/quant_params.json \
    --out-dir hardware/src/v1.1/rtl/weights

# 导出测试图像
python model_tools/export_test_img.py \
    --normalize \
    --quant-params model_tools/quant_params.json
```

### 3. 仿真验证
```bash
# Python 参考推理（Golden Reference）
python model_tools/hw_ref.py \
    --image hardware/src/v1.1/tb/test_image.mem \
    --weights hardware/src/v1.1/rtl/weights \
    --quant-params model_tools/quant_params.json

# RTL 仿真
python hardware/src/v1.1/script/run_sim.py \
    --tb tb_mnist_network_core \
    --no-wave

# 批量测试（准确率评估）
python model_tools/batch_sim.py \
    --count 10000 \
    --normalize \
    --quant-params model_tools/quant_params.json \
    --quiet
```

### 4. 上板验证
```bash
# 1. 使用 Quartus 综合 mnist_system_top.v 并下载 .sof 到 FPGA
# 2. 通过 UART 发送图像并接收结果
python model_tools/send_image.py
```

---

## 📊 模块说明

### 核心计算模块
| 模块名称 | 功能 | LCs | M9K | DSP |
|---------|------|-----|-----|-----|
| `layer1_block` | Conv1 + Pool1 | 2,356 | 3 | 5 |
| `layer2_block` | Conv2 + Pool2 | 3,664 | 7 | 5 |
| `layer3_fc1` | 全连接层 1 | 2,846 | 11 | 5 |
| `layer4_fc2` | 全连接层 2 | 622 | 4 | 1 |
| **总计** | 网络核心 | **9,438** | **25** | **16** |

### I/O 与控制模块
- `uart_rx`/`uart_tx`：串口通信（115200 bps）
- `inference_tx_timer`：推理计时与结果打包
- `seg_595_dynamic`：数码管显示（驱动 74HC595）

---

## ⚠️ 已知问题与改进方向

### 当前限制
1. **时序未收敛**：50 MHz 下 Setup WNS = -2.745 ns
   - **建议**：降低时钟频率至 40 MHz，或增加关键路径流水级

2. **串口传输瓶颈**：115200 bps 传输 784 字节需 ~68 ms
   - **建议**：提升波特率至 921600 bps，或采用并行接口

3. **资源接近饱和**：97% 资源占用限制进一步优化空间
   - **建议**：迁移至更大规模 FPGA（Cyclone V / Zynq）

### 未来改进方向
- [ ] 引入脉动阵列（Systolic Array）提升并行度
- [ ] 实现多层流水线，提高吞吐率
- [ ] 集成摄像头输入，实现实时识别
- [ ] 探索动态量化与混合精度策略

---

## 📖 参考资料

- **LeCun et al., "Gradient-Based Learning Applied to Document Recognition"**, 1998
- **Intel Cyclone IV Handbook**: [链接](https://www.intel.com/content/www/us/en/programmable/documentation/lit-index.html)

---
