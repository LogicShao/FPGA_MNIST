import serial
import time
import numpy as np
from torchvision import datasets, transforms
import torch

# --- 串口配置 (请根据实际情况修改 COM 口) ---
SERIAL_PORT = 'COM3'  # Windows 示例，Linux/Mac 用 /dev/ttyUSB0
BAUD_RATE = 115200


def get_test_image(index=0):
    # 获取一张 MNIST 测试图
    dataset = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.Compose([transforms.ToTensor()]))
    img, label = dataset[index]
    # img 是 (1, 28, 28) float 0~1
    # 量化成 int8 (0~127)，注意这里要和训练时的量化逻辑匹配，保持正数方便
    img_np = (img.numpy().squeeze() * 127).astype(np.int8)
    return img_np.flatten(), label


def send_via_uart(data_bytes):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"打开串口 {SERIAL_PORT} 成功")

        # 协议：0xAA (头) + 784字节 + 0x55 (尾)
        packet = bytearray()
        packet.append(0xAA)
        packet.extend(data_bytes.tobytes())
        packet.append(0x55)

        print(f"正在发送 {len(packet)} 字节数据...")
        ser.write(packet)
        print("发送完毕！等待板子响应...")

        # 简单的接收打印循环（用于看 Nios 的 printf）
        start_time = time.time()
        while time.time() - start_time < 5:  # 听5秒
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore')
                print(f"[FPGA]: {line.strip()}")

        ser.close()
    except Exception as e:
        print(f"串口错误: {e}")


if __name__ == "__main__":
    # 选第 5 张测试图
    idx = 7
    pixels, label = get_test_image(idx)

    print(f"图片真实数字是: {label}")
    print(f"像素数据预览: {pixels[300:310]} ...")  # 打印中间几个像素看看

    # 提示用户插入串口
    input("请确认 FPGA 串口已连接，按回车开始发送...")
    send_via_uart(pixels)
