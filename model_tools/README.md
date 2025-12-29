# model_tools (FPGA_MNIST)

用途: 量化/导出/仿真/串口发送脚本。

## 常用命令

1) 量化参数（normalize on）:
```
python calc_quant_params.py --normalize
```

2) int32 bias ROM:
```
python quantize_bias.py --quant-params quant_params.json --out-dir ../hardware/src/v1.1/rtl/weights
```

3) 导出测试图像:
```
python export_test_img.py --normalize --quant-params quant_params.json
```

4) 硬件等效参考推理:
```
python hw_ref.py --image ../hardware/src/v1.1/tb/test_image.mem --weights ../hardware/src/v1.1/rtl/weights --quant-params quant_params.json
```

5) 批量仿真（RTL）:
```
python batch_sim.py --count 10000 --normalize --quant-params quant_params.json --quiet
```

## 数据目录

MNIST 数据集位于: `model_tools/data`
