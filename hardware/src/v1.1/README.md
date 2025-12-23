# 📄 项目文档：FPGA MNIST 手写数字识别加速器 (v1.1)

**版本状态**：`Draft / In-Progress`
**目标**：将推理核心从 Nios II 纯软件迁移至 Verilog 硬件加速器。
**预期提升**：解决 v1.0 中 14ms 的推理延迟瓶颈，利用流水线技术大幅提升计算吞吐率。

---

## 1. 系统架构概览 (System Architecture)

v1.1 采用 **软硬协同 (Co-Design)** 架构。Nios II 不再负责繁重的矩阵乘法，转型为“数据调度员”。

* **控制层 (Software - Nios II)**:
* 通过 UART 接收 PC 发送的图像数据。
* 负责流程控制、结果显示（数码管/串口）。
* **核心任务**：通过 PIO 总线，将像素逐个“喂”给硬件加速器。


* **计算层 (Hardware - Verilog Accelerator)**:
* **Line Buffer (行缓存)**：实时缓存 4 行图像数据，构建 5x5 卷积窗口。
* **PE (Processing Element)**：并行/流水线式执行卷积核的点乘累加运算。



---

## 2. 模块清单与文件结构

请确保你的工程目录 `hardware/src/v2/` 下包含以下文件：

### A. RTL 核心硬件 (Verilog)

| 文件名 | 状态 | 功能描述 |
| --- | --- | --- |
| `layer1_window_gen.v` | **已完成** | 核心模块。输入像素流，输出 5x5 并行窗口数据（W00-W44）。 |
| `vector_dot_product.v` | **已完成** | 计算核心。执行向量点乘（MAC操作）。(注：需根据并行度需求调整实例化数量)。 |
| `conv_accelerator.v` | **待编写** | **中间层**。将 `window_gen` 和 `dot_product` 封装起来，对接 PIO 接口。 |
| `mnist_system_top.v` | **待编写** | **顶层模块**。连接 PLL、Qsys 系统、加速器和外设引脚。 |

### B. 系统集成 (Qsys/Platform Designer)

| 组件名 | 类型 | 设置 | 功能 |
| --- | --- | --- | --- |
| `pio_img_data` | **新建** | Output, 32-bit | Nios 发送数据 (Bit0-7:像素, Bit8:Valid, Bit9:Rst)。 |
| `pio_result` | **新建** | Input, 32-bit | Nios 读取硬件计算出的卷积结果。 |
| `pio_seg_ctrl` | 沿用 | Output, 32-bit | 数码管控制 (沿用 v1.0)。 |

### C. 驱动软件 (C Code)

| 功能 | 状态 | 描述 |
| --- | --- | --- |
| `hardware_inference()` | **待编写** | 替代原来的双层 `for` 循环乘法。负责向 PIO 写入数据脉冲。 |

---

## 3. 开发路线图 (Action Plan)

当你回来工作时，请按此顺序执行：

### 第一阶段：Qsys 硬件配置

1. 打开 Qsys (Platform Designer)。
2. 添加 `pio_img_data` (Output, 32位) 和 `pio_result` (Input, 32位)。
3. **Export** 这两个接口（命名为 `external_connection`）。
4. 点击 "Generate HDL" 更新系统网表。

### 第二阶段：Verilog 组装

1. 创建 `conv_accelerator.v`：
* 实例化 `layer1_window_gen`。
* 实例化 `vector_dot_product`（目前先做一个通道的计算验证）。


2. 创建 `mnist_system_top.v`：
* 实例化 Qsys 系统。
* 解析 `pio_img_data` 信号 (拆分为 pixel, valid, rst)。
* 连接 `conv_accelerator`。



### 第三阶段：综合与比特流生成

1. 在 Quartus 中将 `mnist_system_top` 设为 Top-Level Entity。
2. 运行 Analysis & Synthesis 检查语法错误。
3. 分配引脚（如果用到新的调试引脚）。
4. 全编译 (Compile)，生成 `.sof` 文件并下载。

### 第四阶段：软件驱动开发 (Nios II Eclipse)

1. 并在 Eclipse 中 "Generate BSP" 以更新 `system.h`（获取新 PIO 的基地址）。
2. 编写 C 代码，使用 `IOWR` 指令向加速器发送 784 个像素数据。
3. 读取结果并打印，验证与 Python 仿真结果是否一致。

---

## 4. 关键数据备忘 (Memo)

* **时钟频率**: 50 MHz (周期 20ns)。
* **输入图像**: 28 x 28 = 784 像素。
* **Valid 信号**: 高电平有效。
* **通信协议**:
* CPU 发送 `Pixel + Valid(1)` -> 保持至少 1 个时钟周期。
* CPU 发送 `Pixel + Valid(0)` -> 完成握手。
