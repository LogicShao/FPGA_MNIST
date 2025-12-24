# 📄 项目文档：FPGA MNIST 手写数字识别加速器 (v1.1)

**版本状态**：`Draft / In-Progress`
**目标**：使用纯 Verilog 实现端到端推理路径（UART -> 卷积加速 -> 结果输出），不依赖 Nios II。
**预期提升**：解决 v1.0 中 14ms 的推理延迟瓶颈，利用流水线技术大幅提升计算吞吐率。

---

## 1. 系统架构概览 (System Architecture)

v1.1 采用 **纯硬件 (Pure RTL)** 架构，不使用 Nios II。数据从 PC 通过 UART 进入 FPGA，由顶层逻辑驱动窗口生成与卷积 PE，再把结果回传并显示。

* **数据输入 (UART RX)**:
* PC 发送 28x28 像素流，每个字节对应一个像素。
* `uart_rx` 输出 `po_data` 与 `po_flag`，作为像素与 valid 输入。

* **计算层 (Verilog Accelerator)**:
* **Line Buffer (行缓存)**：实时缓存 4 行图像数据，构建 5x5 卷积窗口。
* **5x5 PE**：并行乘加 + bias + ReLU。

* **结果输出 (UART TX / 数码管)**:
* `uart_tx` 发送结果低 8 位到 PC。
* 数码管显示最近一次识别结果。



---

## 2. 模块清单与文件结构

请确保你的工程目录 `hardware/src/v1.1/rtl/` 下包含以下文件：

### A. RTL 核心硬件 (Verilog)

| 文件名 | 状态 | 功能描述 |
| --- | --- | --- |
| `layer1_window_gen.v` | **已完成** | 核心模块。输入像素流，输出 5x5 并行窗口数据（W00-W44）。 |
| `conv_accelerator.v` | **进行中** | 当前包含 `conv_pe_5x5`（5x5 卷积 + bias + ReLU），待封装完整加速器接口。 |
| `vector_dot_product.v` | **保留** | 旧版向量点乘（MAC）模块，用于对照验证或简化测试。 |
| `mnist_system_top.v` | **待对接** | 纯 Verilog 顶层，UART/数码管/加速器连接。 |

### B. 通讯与显示 (UART / 7-Segment)

| 文件名 | 状态 | 功能描述 |
| --- | --- | --- |
| `uart/uart_rx.v` | **已完成** | UART 接收器，输出 `po_data` / `po_flag`。 |
| `uart/uart_tx.v` | **已完成** | UART 发送器，发送结果字节。 |
| `seg/seg_dynamic.v` | **已完成** | 动态数码管扫描显示。 |
| `seg/seg_595_dynamic.v` | **已完成** | 数码管 + 595 驱动封装。 |
| `seg/hc595_ctrl.v` | **已完成** | 74HC595 串行控制。 |
| `seg/bcd_8421.v` | **已完成** | 二进制转 BCD。 |

---

## 3. 开发路线图 (Action Plan)

当你回来工作时，请按此顺序执行：

### 第一阶段：纯 Verilog 数据通路

1. 在 `conv_accelerator.v` 中完成加速器封装：
* 实例化 `layer1_window_gen`。
* 实例化 `conv_pe_5x5`，连入 25 像素与卷积核/偏置。
* 按 28x28 帧长做计数，必要时在新帧开始时清状态。

### 第二阶段：顶层对接与外设

1. 在 `mnist_system_top.v` 对接 UART 与显示：
* `uart_rx.po_data` -> 像素输入。
* `uart_rx.po_flag` -> `valid_in`。
* `result/result_valid` -> `uart_tx` 与数码管显示。

### 第三阶段：综合与比特流生成

1. 在 Quartus 中将 `mnist_system_top` 设为 Top-Level Entity。
2. 运行 Analysis & Synthesis 检查语法错误。
3. 分配引脚（如果用到新的调试引脚）。
4. 全编译 (Compile)，生成 `.sof` 文件并下载。

### 第四阶段：联调与验证

1. 使用 `python model_tools/send_image.py` 发送 MNIST 图像。
2. 读取 UART 回传结果，并与 Python 仿真结果对比。

---

## 4. 关键数据备忘 (Memo)

* **时钟频率**: 50 MHz (周期 20ns)。
* **UART 波特率**: 115200 (与 `uart_rx/uart_tx` 参数一致)。
* **输入图像**: 28 x 28 = 784 像素。
* **Valid 信号**: 高电平有效。
* **通信协议**:
* PC 通过 UART 连续发送 784 个字节，每个字节对应一个像素。
* `uart_rx.po_flag` 为单周期脉冲，可直接作为 `valid_in`。
