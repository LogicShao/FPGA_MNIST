@echo off

python model_tools/calc_quant_params.py --model model_tools\trained_models\TinyLeNet_20251223_095246_acc99.00.pth --normalize --samples 2000 --int-calib               
python model_tools/export.py --model-path model_tools\trained_models\TinyLeNet_20251223_095246_acc99.00.pth --output model_tests/v2/tinylenet_weights.h                
python model_tools/weights_to_mem.py --input model_tests/v2/tinylenet_weights.h --out-dir hardware/src/v1.1/rtl/weights --format mem,mif --verilog                     
python model_tools/quantize_bias.py --quant-params model_tools/quant_params.json --out-dir hardware/src/v1.1/rtl/weights                                               
python model_tools/hw_ref.py --batch --count 200 --normalize --quant-params model_tools/quant_params.json --quiet
python model_tools/batch_sim.py --count 20 --normalize --quant-params model_tools/quant_params.json --quiet
