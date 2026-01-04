# 快速开始指南（v1.1 硬件加速器）

> 5 分钟完成从量化参数生成到仿真验证的完整流程

---

## ⚡ 极速上手（零基础）

### 前提条件

- ✅ Python 3.x 已安装
- ✅ 已安装依赖：`pip install torch torchvision numpy tqdm`
- ✅ Icarus Verilog 已安装（仿真需要）

### 三步完成验证

```bash
cd model_tools

# 1. 生成量化参数（20秒）
python calc_quant_params.py --normalize

# 2. 生成 INT32 偏置 ROM（5秒）
python quantize_bias.py \
    --quant-params quant_params.json \
    --out-dir ../hardware/src/v1.1/rtl/weights

# 3. 导出测试图像并运行 Python 参考推理（10秒）
python export_test_img.py --normalize --quant-params quant_params.json
python hw_ref.py \
    --image ../hardware/src/v1.1/tb/test_image.mem \
    --weights ../hardware/src/v1.1/rtl/weights \
    --quant-params quant_params.json
```

**预期输出**：
```
Conv1 q[0]: -22
Pool1 q[0]: -22
...
Predicted: 7, Label: 7, Match: True
```

---

## 📊 进阶：RTL 仿真验证

### 单张图像仿真（30秒）

```bash
cd ../hardware/src/v1.1
python script/run_sim.py --tb tb_mnist_network_core --no-wave
```

**预期**：RTL 输出与 Python 参考一致

### 批量测试（快速验证 20 张）

```bash
cd model_tools
python batch_sim.py \
    --count 20 \
    --normalize \
    --quant-params quant_params.json \
    --quiet
```

**预期准确率**：20/20 = 100%

---

## 🚀 完整测试集评估（10,000 张）

**警告**：需要数小时完成

```bash
python batch_sim.py \
    --count 10000 \
    --normalize \
    --quant-params quant_params.json \
    --quiet
```

**预期准确率**：~98.71%

---

## 🔌 上板验证（FPGA）

### 1. 综合并下载

1. 使用 Quartus 打开 `hardware/src/v1.1/rtl/mnist_system_top.v`
2. 综合项目（约 5 分钟）
3. 下载 `.sof` 到 FPGA

### 2. 串口测试

```bash
cd model_tools
python send_image.py
```

**交互示例**：
```
1) MNIST image
2) Custom file
> 1
Enter image index (0-9999): 42
Sending image #42 (label: 3)...
FPGA Response: Predicted: 3, Inference time: 10.031 ms
```

---

## 🛠️ 故障排查

### 问题 1：量化参数不存在

```bash
# 解决：重新生成
python calc_quant_params.py --normalize
```

### 问题 2：RTL 仿真结果不一致

```bash
# 解决：重新生成所有文件
python calc_quant_params.py --normalize
python quantize_bias.py --quant-params quant_params.json --out-dir ../hardware/src/v1.1/rtl/weights
python export_test_img.py --normalize --quant-params quant_params.json
```

### 问题 3：串口无响应

**检查清单**：
- [ ] FPGA 已下载 `.sof` 文件
- [ ] 串口号正确（修改 `send_image.py` 中的 `SERIAL_PORT`）
- [ ] 波特率为 115200
- [ ] USB-UART 驱动已安装

---

## 📖 深入学习

| 文档 | 内容 |
|------|------|
| [model_tools/README.md](README.md) | 完整工具链文档 |
| [hardware/src/v1.1/README.md](../hardware/src/v1.1/README.md) | 硬件实现详解 |
| [README.md](../README.md) | 项目总览 |
| [README_v1.md](../README_v1.md) | Nios II 实现路线 |

---

## 🎯 常用命令速查

```bash
# ========== 量化与权重 ==========
# 计算量化参数
python calc_quant_params.py --normalize

# 生成 INT32 偏置
python quantize_bias.py --quant-params quant_params.json --out-dir ../hardware/src/v1.1/rtl/weights

# 导出测试图像
python export_test_img.py --normalize --quant-params quant_params.json

# ========== 验证 ==========
# Python 参考推理（单张）
python hw_ref.py --image ../hardware/src/v1.1/tb/test_image.mem --weights ../hardware/src/v1.1/rtl/weights --quant-params quant_params.json

# Python 参考推理（批量）
python hw_ref.py --batch --count 200 --normalize --quant-params quant_params.json

# RTL 仿真（单张）
cd ../hardware/src/v1.1
python script/run_sim.py --tb tb_mnist_network_core --no-wave

# RTL 批量仿真
cd model_tools
python batch_sim.py --count 20 --normalize --quant-params quant_params.json --quiet

# ========== 上板 ==========
# 串口发送图像
python send_image.py

# ========== 可视化 ==========
# 绘制训练曲线
python train_log_plot.py
```

---

## ⚙️ 高级选项

### 调试 Mismatch

```bash
# 自动保存失败样本
python batch_sim.py \
    --count 20 \
    --debug-mismatch \
    --normalize \
    --quant-params quant_params.json
```

生成文件位于 `batch_sim_debug/`：
- `idx_<n>_sim.log` - RTL 仿真日志
- `idx_<n>_hw_ref.log` - Python 参考日志
- `idx_<n>_test_image.mem` - 失败样本

### 加速仿真（不准确）

```bash
# FAST_SIM 模式（仅用于波形检查）
python batch_sim.py --count 100 --fast --quiet
```

⚠️ **警告**：跳过真实计算，输出不准确！

---

**最后更新**：2026-01-04
**版本**：v1.1 硬件加速器
**状态**：✅ Completed
