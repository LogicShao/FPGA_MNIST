# FPGA MNIST Handwritten Digit Recognition (Nios II)

基于 FPGA (Nios II 软核) 的 MNIST 手写数字识别系统。本项目采用 **软硬协同 (Hardware-Software Co-design)** 的设计思路，实现了从 PC 端发送图片到 FPGA 端完成推理显示的完整链路。

## 📖 项目简介

本项目旨在 FPGA 开发板上部署一个轻量级的神经网络（MLP），用于识别 28x28 像素的手写数字。

*   **输入**：PC 端 Python 脚本读取 MNIST 测试集，通过 **UART 串口** 发送量化后的 int8 像素数据。
*   **处理**：FPGA 内部运行 Nios II/f 软核处理器，接收数据并执行神经网络前向传播（推理）。
*   **输出**：
    *   **串口回传**：识别结果及置信度。
    *   **板载显示**：LCD 屏幕显示识别到的数字（可选）。
*   **进阶特性**：支持将计算密集型的矩阵乘法（MAC）卸载到 **Verilog 硬件加速器**（Custom IP）以提升性能。

## 🛠️ 技术栈

*   **硬件平台**：Intel (Altera) Cyclone IV / 10 (野火/正点原子开发板)
*   **开发环境**：Quartus Prime 18.1+, Platform Designer (Qsys)
*   **处理器架构**：Nios II/f (Fast Core)
*   **模型训练**：Python 3.11, PyTorch 2.8.0+cu129
*   **编程语言**：C (嵌入式软件), Verilog HDL (硬件逻辑), Python (上位机工具)

## 📂 目录结构

```text
FPGA_MNIST_Nios/
├── README.md                # 项目说明文档
├── doc/                     # 文档与开发计划
│   └── PLAN.md              # 开发进度追踪
├── model_tools/             # Python 工具集
├── software/                # Nios II 软件工程
│   └── app/                 # C 源代码 (main.c, model_weights.h)
└── hardware/                # Quartus 硬件工程
    ├── src/                 # Verilog 源码 (加速器)
    └── quartus_prj/         # 工程文件 (.qpf)
```

## 🚀 快速开始

### 1. 环境准备 (Python)
本项目依赖特定版本的 PyTorch 环境。推荐使用 Conda 克隆环境：
```bash
# 方式 A: 克隆现有环境 (推荐)
conda create -n fpga_mnist --clone your_conda_env_name
conda activate fpga_mnist

# 方式 B: 手动安装依赖
pip install -r model_tools/requirements.txt
```

### 2. 模型训练与部署
在 PC 上训练 MLP 模型，并将其量化导出为 C 语言头文件：

### 3. 硬件综合与烧录
1.  使用 Quartus 打开 `hardware/quartus_prj/` 下的工程。
2.  进入 Platform Designer (Qsys) 生成 Nios 系统。
3.  全编译工程，下载 `.sof` 文件到 FPGA。

### 4. 软件编译与运行
1.  使用 Nios II SBT for Eclipse 创建工程（基于 `software/app` 模板）。
2.  编译并 `Run as Hardware`。
3.  运行 PC 端发送脚本进行测试

## 📡 通信协议

PC 与 FPGA 之间通过 UART 通信，波特率 **115200**。

| 字节偏移 | 内容 | 说明 |
| :--- | :--- | :--- |
| 0 | `0xAA` | 帧头 (Start Byte) |
| 1 ~ 784 | Data | 28x28 像素数据 (int8, Row-Major) |
| 785 | `0x55` | 帧尾 (End Byte) |

## 📝 许可证

MIT License / Academic Use Only
