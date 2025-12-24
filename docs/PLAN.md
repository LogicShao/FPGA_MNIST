
### 文件 2：开发计划文档 (`doc/PLAN.md`)

这个文件用来给你自己打钩（`[x]`）用的，放在 `doc` 文件夹下。

```markdown
# 📅 开发计划与进度追踪 (Development Plan)

**目标**：在 FPGA 课设截止日期前，完成从模型训练到板级验证的完整流程。

## 阶段 0：环境与项目初始化 (Done)
- [x] 创建 Git 仓库与目录结构
- [x] 配置 Python 环境 (Conda/PyTorch 2.8.0+cu129)
- [x] 编写 `.gitignore` 规则

## 阶段 1：模型准备 (Model Preparation)
**目标**：获得能在 C 语言环境中运行的 int8 权重数据。
- [ ] **1.1 编写训练脚本** (`model_tools/v1/train_export.py`)：搭建 784->32->10 MLP 网络。
- [ ] **1.2 实现量化逻辑**：将 float32 权重转换为 int8 并在 Python 中验证精度损失。
- [ ] **1.3 导出 C 头文件**：自动生成 `model_weights.h`。
- [ ] **1.4 编写串口测试脚本** (`send_image.py`)：实现图片读取与协议封装发送。

## 阶段 2：硬件系统搭建 (Hardware Setup)
**目标**：让 Nios II 跑起来，串口通起来。
- [ ] **2.1 Platform Designer 配置**：
    - 添加 Nios II/f Core。
    - 添加 On-Chip Memory (建议 64KB+)。
    - 添加 UART (RS232) 与 JTAG UART。
    - 添加 PIO (LED) 或 LCD 接口。
- [ ] **2.2 Quartus 工程集成**：实例化 Qsys 系统，分配引脚，约束时钟。
- [ ] **2.3 硬件编译**：生成 `.sof` 文件并成功下载到开发板。

## 阶段 3：软件实现 (Software Implementation)
**目标**：在 Nios II 上实现接收与纯软推理。
- [ ] **3.1 串口驱动开发**：实现 `uart_receive_frame()`，能够正确解析 `0xAA ... 0x55` 包。
- [ ] **3.2 推理逻辑移植**：用 C 语言实现全连接层与 ReLU 计算（使用 int8/int32 运算）。
- [ ] **3.3 联调测试**：
    - PC 发送图片 -> Nios 接收 -> 计算 -> Nios `printf` 结果。
    - 验证 FPGA 算出的结果与 Python 端是否一致。

## 阶段 4：人机交互优化 (Display & UI)
**目标**：脱离电脑终端，直观展示结果。
- [ ] **4.1 LCD/数码管显示**：将推理结果（0-9）显示在板载屏幕上。
- [ ] **4.2 (可选) 推理时间统计**：利用 timestamp timer 统计计算耗时。

## 阶段 5：硬件加速 (Hardware Acceleration - Bonus)
**目标**：使用 Verilog 提升计算速度（加分项）。
- [ ] **5.1 设计加速器 IP**：编写 Verilog 模块实现向量点积（MAC）。
- [ ] **5.2 挂载 Avalon 总线**：封装为 Avalon-MM Slave 接口。
- [ ] **5.3 驱动修改**：C 代码中用 `IOWR/IORD` 替代第一层的 `for` 循环。
- [ ] **5.4 性能对比**：对比软硬实现的耗时差异。

---

## 🐞 问题记录 (Log)

*   *(在此处记录开发过程中遇到的 Bug 和解决方案)*
*   [2025-12-23] 解决 PyTorch 版本依赖问题，采用 Conda Clone 方案。
```

---

### 现在的操作建议

1.  在根目录创建 `README.md`，把上面的内容复制进去。
2.  在 `doc` 目录创建 `PLAN.md`，把计划书复制进去。
3.  **Git 提交**：
    ```bash
    git add README.md doc/PLAN.md
    git commit -m "Docs: Add README and Development Plan"
    ```

做完这些，你的项目看起来就已经非常专业了！接下来你可以放心地去跑 `model_tools/v1/train_export.py` 了。
