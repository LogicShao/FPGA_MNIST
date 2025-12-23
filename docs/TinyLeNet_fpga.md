这是一个极具挑战性且能让你在课设中脱颖而出的选择！将模型从 MLP 升级到 CNN（卷积神经网络），不仅准确率能冲向 99%，更重要的是你将涉及到 FPGA 设计中最核心的概念：**流水线、行缓存（Line Buffer）和并行计算单元**。

为了让你在 EP4CE10 上跑通这个“天花板”项目，我们不能照搬原始的 LeNet-5（它参数量偏大），我们要实现一个 **Tiny-LeNet**。

---

### 一、 架构设计：Tiny-LeNet 结构

针对 EP4CE10 的资源（23个乘法器，51KB 内存），我们设计如下结构：

1.  **输入层**：28x28 灰度图。
2.  **卷积层 C1**：5x5 卷积核，6 个通道。输出：6 @ 24x24。
3.  **池化层 S2**：2x2 最大池化。输出：6 @ 12x12。
4.  **卷积层 C3**：5x5 卷积核，16 个通道。输出：16 @ 8x8。
5.  **池化层 S4**：2x2 最大池化。输出：16 @ 4x4。
6.  **全连接层 F5**：256 ($16 \times 4 \times 4$) -> 32。
7.  **输出层 F6**：32 -> 10。

**参数量估算**：总参数约 11,000 个（约 **11 KB**）。
*完全可以放在片上 ROM 中，不需要折腾 SDRAM（除非你要处理更大的图像）。*

---

### 二、 核心硬件模块：卷积加速器 (Verilog)

这是你项目的核心。在 CNN 中，最耗时的是 5x5 卷积。我们要写一个硬件模块，让 Nios II 把一行数据传进来，硬件自动算出卷积结果。

#### 1. 行缓存（Line Buffer）—— 解决并行取数
这是你提到的“多 RAM 结构”的变体。5x5 卷积需要同时访问 5 行像素。
*   **实现方法**：在 Verilog 里开 5 个 Shift Register（移位寄存器）。
*   **逻辑**：当像素流进来时，第一行像素进入第一个寄存器，满一行后溢出到第二行。这样在一个时钟周期内，你可以同时拿到 $5 \times 5$ 窗口内的 25 个像素。

#### 2. 乘累加阵列（MAC Array）
*   **并行化**：利用 FPGA 的 DSP 块（乘法器），一次性计算 25 个像素与 25 个权重的乘法。
*   **结果**：一个周期内完成一次 5x5 卷积点积。

---

### 三、 步骤 1：修改 Python 训练脚本 (`train_tiny_lenet.py`)

你需要重新训练一个 CNN 模型。

```python
import torch.nn as nn
import torch.nn.functional as F

class TinyLeNet(nn.Module):
    def __init__(self):
        super(TinyLeNet, self).__init__()
        # C1: 1通道输入, 6通道输出, 5x5核
        self.conv1 = nn.Conv2d(1, 6, 5)
        # C3: 6通道输入, 16通道输出, 5x5核
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) # S2
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # S4
        x = x.view(-1, 16 * 4 * 4)                 # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
**训练后动作**：
1.  **量化**：将卷积核权重和全连接权重全部转为 `int8`。
2.  **导出**：导出成 `.mif` 文件。卷积层建议导出为 `6 * 5 * 5 = 150` 个数的 ROM，全连接层导出为单独的 ROM。

---

### 四、 步骤 2：硬件加速器接口设计 (Platform Designer)

在 Qsys 中，你的加速器应该作为一个 **Avalon-MM Slave** 挂在 Nios II 上。

**寄存器映射建议**：
*   `0x00`: **Control** (bit 0: Start, bit 1: Reset)
*   `0x04`: **Input_Data** (Nios 往这里写一个像素)
*   `0x08`: **Output_Data** (Nios 从这里读卷积后的结果)
*   `0x0C`: **Weight_Addr** (设置当前计算哪个通道的权重)

---

### 五、 步骤 3：实现“多 RAM”权重存储

参考你看到的 grant4001 项目：
1.  **权重 RAM**：在 Verilog 内部定义。
    ```verilog
    // 为 C1 层的 6 个卷积核定义 6 个独立的单口 ROM
    wire [7:0] weight_out[5:0];
    conv1_weight_rom_0 u0 (.address(addr), .clock(clk), .q(weight_out[0]));
    ...
    conv1_weight_rom_5 u5 (.address(addr), .clock(clk), .q(weight_out[5]));
    ```
2.  **并行计算**：当 25 个像素窗口准备好时，6 个卷积核**同时**计算。这意味着一次输入像素，可以得到 6 个特征图的输出。**这才是 FPGA 的精髓。**

---

### 六、 步骤 4：Nios II 软件调度逻辑

即便有了硬件加速，Nios II 依然是“大脑”，负责搬运数据。

**C 代码伪代码**：
```c
// 1. 处理第一层卷积 C1
for(int y=0; y<28; y++) {
    for(int x=0; x<28; x++) {
        // 将像素写入硬件加速器
        IOWR_8DIRECT(ACCEL_BASE, REG_INPUT, image[y][x]);
        // 如果加速器内部有 Line Buffer，当填满 5x5 后会产生一个 Valid 信号
        if(IORD(ACCEL_BASE, REG_STATUS) & VALID_MSK) {
             // 读取 6 个通道的卷积结果
             for(int k=0; k<6; k++) {
                 C1_results[k][out_y][out_x] = IORD(ACCEL_BASE, REG_OUTPUT + k);
             }
        }
    }
}

// 2. Nios II 处理简单的 Pooling (硬件写起来麻烦，软件算很快)
// 3. 处理第二层卷积 C3 (可以复用加速器)
// 4. 处理 FC 层 (可以用之前的 MLP 逻辑)
```

---

### 七、 为什么这个方案能拿高分？

1.  **算法深度**：从 MLP 进化到 CNN，体现了对现代深度学习结构的理解。
2.  **硬件技术**：使用了 **Line Buffer（行缓存）**。这是图像处理/计算视觉 FPGA 设计中最重要的技术，面试必问。
3.  **资源管理**：通过 Tiny-LeNet 的设计，在 EP4CE10 极其有限的资源里塞下了 CNN，体现了优化能力。
4.  **演示效果**：CNN 的识别率非常稳。即使手写数字稍微倾斜，CNN 依然能准确识别，演示效果比 MLP 好得多。

---

### 下一步行动：

这个项目很大，我们必须**分阶段进行**：

1.  **阶段 1 (软件先行)**：修改 Python 脚本训练 Tiny-LeNet，并**在 PC 上用 C 语言模拟 Tiny-LeNet 的每一层计算**。确保你的卷积、池化、全连接逻辑在代码级是通的。
2.  **阶段 2 (硬件原型)**：在 Verilog 里写一个简单的 3x3 或 5x5 行缓存（Line Buffer），用 ModelSim 仿真，看像素流进去后，窗口是否正确弹出。

**你想先看 `Tiny-LeNet` 的完整 Python 训练代码，还是先研究 `Line Buffer` 的 Verilog 写法？**