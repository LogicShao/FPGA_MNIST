# v1.1 仿真测试说明

本目录用于存放 RTL 仿真测试文件（testbench）和波形相关资源。以下内容面向课程报告，重点说明“该看哪些波形”和“正确波形应该长什么样”。

## 总体观察要点

- `clk`：50 MHz（20 ns 周期），所有有效数据在上升沿采样。
- `rst_n`：低有效复位，复位后再启动输入。
- `valid_in`：输入有效脉冲，表示本拍输入被采样。
- `*_valid`：各层输出有效脉冲；输出数据只有在对应 `*_valid=1` 时有效。
- 观察原则：输出应只在 valid=1 的周期变化，其余时间保持不变。

## 各测试文件的波形说明

### 1) `tb_mnist_network_core.v`（全网络功能仿真）

关注信号：
`valid_in`，`result_valid`，`result`，`dut.u_layer1.result_valid`，`dut.u_layer2.out_valid`，`dut.u_layer3.out_valid`，
`dut.u_layer1.load_state`~`dut.u_layer4.load_state`。

期望波形（关键节奏）：
- `load_state` 全部变为 1 后，开始输入。
- `valid_in` 连续为 1 784 拍（28x28）。
- `u_layer1.result_valid`：144 次脉冲（12x12 池化输出）。
- `u_layer2.out_valid`：16 次脉冲（4x4 池化输出）。
- `u_layer3.out_valid`：32 次脉冲（FC1 32 维）。
- `result_valid`：10 次脉冲（FC2 10 类）。
- `result` 只在 `result_valid=1` 时更新，顺序为类 0~9。

用于报告的结论：
- 有效信号链条正确（输入→L1→L2→FC1→FC2），输出数量与网络结构一致。

### 2) `tb_mnist_network_core_wave.v`（全网络波形版）

关注信号：
`valid_in`，`dut.u_layer1.result_valid`，`dut.u_layer2.out_valid`，`dut.u_layer3.out_valid`，`result_valid`。

期望波形：
- `valid_in` 一次连续脉冲串（784 拍）。
- 四个 `valid` 信号呈“阶梯式延迟”出现：先 L1，再 L2，再 FC1，最后 `result_valid`。
- `result_valid` 仅在末尾出现 10 次。

用途：报告中展示“流水线时序和有效信号传播路径”。

### 3) `tb_layer1_block_wave.v`（Layer1：conv1 + relu + pool1）

关注信号：
`valid_in`，`pixel_in`，`result_valid`，`result_ch0..5`，
`dut.load_state`，`dut.c1_state`，`dut.pos_x`，`dut.pos_y`，
`dut.mac_phase`，`dut.img_wr_en`，`dut.img_wr_addr`，`dut.img_rd_addr`，
`dut.conv1_weight_addr`，`dut.conv1_weight_q`。

期望波形：
- `load_state` 从 0→1（权重/偏置加载完成）。
- `img_wr_en` 连续高 784 拍，`img_wr_addr` 递增 0→783。
- `c1_state`：IDLE→LOAD→MAC→OUT 循环。
- `mac_phase` 在 MAC 期间 0/1 交替（同步 ROM 读延迟）。
- `result_valid` 输出 144 次脉冲（12x12）。
- `pos_x` 0→11，`pos_y` 0→11，逐点扫描输出。
- `result_ch0..5` 为 8-bit 量化值，仅在 `result_valid=1` 的周期变化。

用途：报告中展示“LineBuffer+权重读写+MAC 时序”。

### 4) `tb_layer2_block_wave.v`（Layer2：conv2 + relu + pool2）

关注信号：
`valid_in`，`in_ch0..5`，`out_valid`，`out_ch0..15`，
`dut.load_state`，`dut.c_state`，`dut.pos_x`，`dut.pos_y`，
`dut.k_ch`，`dut.k_y`，`dut.k_x`，`dut.mac_phase`，`dut.q_idx`，
`dut.feat_wr_en`，`dut.feat_wr_addr`，`dut.feat_rd_addr`，
`dut.conv2_weight_addr`，`dut.conv2_weight_q`。

期望波形：
- `valid_in` 连续为 1 144 拍（12x12 输入特征）。
- `feat_wr_en` 跟随 `valid_in` 写入特征 RAM。
- `c_state`：IDLE→LOAD→MAC→OUT 循环。
- `mac_phase` 在 MAC 期间 0/1 交替。
- `out_valid` 输出 16 次脉冲（4x4 池化）。
- `pos_x` 0→3，`pos_y` 0→3；`q_idx` 0→15（依次量化 16 通道）。

用途：报告中展示“多通道输入 + 顺序量化 + 池化输出”。

### 5) `tb_conv1_accelerator.v`（旧版 conv1 单元）

关注信号：
`valid_in`，`pixel_in`，`result_valid`，`result_ch0`。

期望波形：
- `valid_in` 连续为 1 784 拍。
- `result_valid` 输出 576 次脉冲（24x24 卷积输出）。
- `result_ch0` 与 testbench 内 `expected` 完全一致（无 mismatch）。

用途：报告中展示“单层卷积正确性验证”。

### 6) `tb_vector_dot_product.v`（点积模块）

关注信号：
`valid_in`，`sop`，`eop`，`data_a`，`data_b`，`result_valid`，`result`。

期望波形：
- 发送 3 个点：[2, -3, 4] · [5, 2, 1]。
- `result_valid` 在 `eop` 之后 1 拍拉高。
- `result = 8`（10 - 6 + 4）。

用途：报告中展示“MAC 流水线与 SOP/EOP 控制正确性”。

## 运行方法

使用脚本 `hardware/src/v1.1/script/run_sim.py`：

```
python hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core
python hardware/src/v1.1/script/run_sim.py --tb tb_mnist_network_core_wave
python hardware/src/v1.1/script/run_sim.py --tb tb_layer1_block_wave
python hardware/src/v1.1/script/run_sim.py --tb tb_layer2_block_wave
```

常用参数：
- `--no-wave`：不打开 GTKWave（只跑仿真）。
- `--quiet`：减少打印并关闭波形（TB 内部识别 `QUIET_SIM`）。
- `--fast`：启用 `FAST_SIM` 宏（由 TB 决定加速策略）。

## 波形文件

波形默认生成在仓库根目录（或 `hardware/src/v1.1/sim`）：
- `tb_layer1_block_wave.vcd`
- `tb_layer2_block_wave.vcd`
- `tb_mnist_network_core_wave.vcd`

如需更小文件，可使用 `vcd2fst` 转换为 `.fst`。

## 波形查看工具（Surfer）

Surfer 是轻量的波形查看器，打开大波形比 GTKWave 更流畅，建议优先使用 `.fst` 格式。

下载与使用步骤（Windows）：
1. 打开浏览器，搜索 `Surfer waveform viewer GitHub releases`。
2. 下载 Windows 版压缩包（如 `surfer-<version>-windows.zip`），解压到任意目录。
3. 运行 `Surfer.exe`。
4. 在 Surfer 内 `File -> Open` 选择 `.vcd` 或 `.fst` 波形文件。

建议流程：
- 先用 `vcd2fst` 转换，再在 Surfer 打开 `.fst`。
- 只加载需要的信号，减少波形渲染负担。
