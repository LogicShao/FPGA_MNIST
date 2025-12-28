# 项目文档：FPGA MNIST 手写数字识别加速器 (v1.1)

**版本状态**：`Draft / In-Progress`  
**目标**：纯 Verilog 端到端推理路径（UART -> 推理 -> 结果输出），不依赖 Nios II  
**预期提升**：通过流水线提升吞吐率，降低推理延迟

---

## 1. 系统架构概览 (System Architecture)

v1.1 采用纯 RTL 架构，数据从 PC 通过 UART 进入 FPGA，推理结果回传并显示。

- **数据输入 (UART RX)**：PC 发送 28x28 像素流（每字节 1 像素）
- **计算层 (RTL)**：line buffer 生成 5x5 窗口，PE 并行计算，池化/全连接等
- **结果输出 (UART TX / 数码管)**：UART 回传预测值，数码管显示最新结果

---

## 2. 模块清单与文件结构

核心路径位于 `hardware/src/v1.1/rtl/`：

- `layer1_window_gen.v`：5x5 窗口生成
- `conv_pe_5x5.v` / `conv_pe_group.v`：卷积 PE
- `conv1_core.v` / `conv2_core.sv`：卷积核心
- `max_pool_2x2.v` / `max_pool_2x2_16ch.v`：池化
- `layer3_fc1.v` / `layer4_fc2.v`：全连接
- `mnist_network_core.v`：四层串联
- `mnist_system_top.v`：UART + 推理 + 顶层 IO

权重/偏置 ROM：`hardware/src/v1.1/rtl/weights/`

---

## 3. 仿真验证 (Simulation Verification)

### 3.1 生成量化参数 / bias / 测试图像

1) 计算 fixed-point 量化参数（normalize on）：
```
python model_tools/calc_quant_params.py --normalize
```

2) 生成 int32 bias ROM：
```
python model_tools/quantize_bias.py --quant-params model_tools/quant_params.json --out-dir hardware/src/v1.1/rtl/weights
```

3) 生成与量化流程一致的 `test_image.mem`：
```
python model_tools/export_test_img.py --normalize --quant-params model_tools/quant_params.json
```

4) Python 参考推理（结果应与 RTL 一致）：
```
python model_tools/hw_ref.py --image hardware/src/v1.1/tb/test_image.mem --weights hardware/src/v1.1/rtl/weights --quant-params model_tools/quant_params.json
```

### 3.2 运行 RTL 仿真

标准仿真（不打开波形）：
```
python hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core --no-wave
```

加速选项：
- `--fast`：启用 `FAST_SIM`，跳过真实乘加，仅用于冒烟/波形检查，**不可用于准确率评估**。
- `--quiet`：启用 `QUIET_SIM`，关闭 VCD 与大部分日志，适合批量测试。

示例：
```
python hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core --no-wave --quiet
```

---

## 4. 批量仿真（测试集）

脚本：`model_tools/batch_sim.py`

常用参数：
- `--count N`：跑 N 张图
- `--start N`：起始 index
- `--normalize`：使用 MNIST normalize
- `--quant-params`：指定量化参数文件
- `--fast`：启用 FAST_SIM（**不准确，只用于冒烟**）
- `--quiet`：关闭大部分日志与波形

全测试集（准确率评估）：
```
python model_tools/batch_sim.py --count 10000 --normalize --quant-params model_tools/quant_params.json --quiet
```

快速冒烟（不代表准确率）：
```
python model_tools/batch_sim.py --count 10000 --normalize --quant-params model_tools/quant_params.json --fast --quiet
```

输出 CSV：`model_tools/batch_sim_results.csv`（index/label/pred/match）

---

## 5. 上板验证 (On-Board Verification)

### 5.1 准备

1) 下载 `mnist_system_top` 的 `.sof` 到 FPGA  
2) 确保 `model_tools/quant_params.json` 是最新（normalize on）  
3) 校验 `model_tools/send_image.py` 的 `SERIAL_PORT` / `BAUD_RATE`

### 5.2 发送 MNIST 图像
```
python model_tools/send_image.py
```

- 选择 1（MNIST image）
- 脚本会按 `quant_params.json` 进行 normalize + s_in 量化
- FPGA UART 回传结果与 `hw_ref.py` 对照

### 5.3 常见问题排查

- UART 无返回：检查 `SERIAL_PORT` / `BAUD_RATE` / 接线
- 预测不一致：确认 `quant_params.json` 与权重匹配
- 输入不一致：确保 `export_test_img.py` / `send_image.py` 使用 normalize + quant_params

---

## 6. 关键参数

- 时钟：50 MHz（20ns）
- UART：115200
- 输入：28x28 = 784 像素
- valid：高电平有效
- 协议：UART 连续发送 784 字节（1 字节=1 像素）
