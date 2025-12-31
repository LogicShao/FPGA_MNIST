import json
import os
import serial
import time
import numpy as np

# --- Serial config (update COM port as needed) ---
SERIAL_PORT = "COM7"  # Windows example; Linux/Mac: /dev/ttyUSB0
BAUD_RATE = 115200
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
QUANT_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "quant_params.json")


def _load_quant_params():
    if not os.path.exists(QUANT_PARAMS_PATH):
        return None, False
    with open(QUANT_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)
    s_in = float(params["scales"]["s_in"])
    normalize = bool(params.get("normalize", False))
    return s_in, normalize


def _quantize_pixels(arr, s_in=None, normalize=False):
    arr = np.asarray(arr)
    if arr.size != 28 * 28:
        raise ValueError(f"Expected 784 pixels, got {arr.size}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
        if normalize:
            arr = (arr - 0.1307) / 0.3081
        if s_in is None:
            abs_max = float(np.max(np.abs(arr))) if arr.size else 0.0
            s_in = 127.0 / abs_max if abs_max > 0 else 1.0
        arr = np.clip(np.round(arr * s_in), -128, 127).astype(np.int8)
    else:
        # Assume already-quantized int8.
        arr = arr.astype(np.int8)
    return arr.flatten()


def get_mnist_image(index=0):
    # Lazy import to avoid slow torch load unless needed
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        DATA_DIR,
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    img, label = dataset[index]
    img_np = img.numpy().squeeze()
    s_in, normalize = _load_quant_params()
    return _quantize_pixels(img_np, s_in, normalize), label


def load_npy_image(path):
    data = np.load(path)
    s_in, normalize = _load_quant_params()
    return _quantize_pixels(data, s_in, normalize), None


def load_bin_image(path):
    raw = np.fromfile(path, dtype=np.int8)
    return _quantize_pixels(raw, None, False), None


def send_via_uart(data_bytes):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Opened serial {SERIAL_PORT}")
        ser.reset_input_buffer()

        # Protocol: 0xAA (head) + 784 bytes + 0x55 (tail)
        packet = bytearray()
        packet.append(0xAA)
        packet.extend(data_bytes.tobytes())
        packet.append(0x55)

        print(f"Sending {len(packet)} bytes...")
        ser.write(packet)
        ser.flush()
        print("Send complete. Waiting for board response...")

        resp = read_uart_response(ser, expected=18, timeout=5.0)
        if resp is None:
            print("[FPGA]: timeout waiting for 18-byte response")
        else:
            results_i8 = np.frombuffer(resp[:10], dtype=np.int8).tolist()
            cycles_total = int.from_bytes(resp[10:14], byteorder="little", signed=False)
            cycles_pure = int.from_bytes(resp[14:18], byteorder="little", signed=False)
            hex_bytes = " ".join(f"{b:02X}" for b in resp)
            print(f"[FPGA]: raw bytes = {hex_bytes}")
            print(f"[FPGA]: fc2_bytes_i8 = {results_i8}")
            print(f"[FPGA]: inf_cycles_total = {cycles_total}")
            print(f"[FPGA]: inf_cycles_pure = {cycles_pure}")

        ser.close()
    except Exception as e:
        print(f"Serial error: {e}")


def read_uart_response(ser, expected=14, timeout=5.0):
    buf = bytearray()
    start = time.time()
    while len(buf) < expected and (time.time() - start) < timeout:
        chunk = ser.read(expected - len(buf))
        if chunk:
            buf.extend(chunk)
        else:
            time.sleep(0.01)
    if len(buf) < expected:
        return None
    return bytes(buf[:expected])


if __name__ == "__main__":
    while True:
        print("\nSelect data source:")
        print("1) MNIST test image (lazy load torch)")
        print("2) Local .npy file (784 values)")
        print("3) Local .bin file (784 int8 bytes)")
        print("q) Quit")
        choice = input("> ").strip().lower()

        if choice == "q":
            break

        try:
            if choice == "1":
                idx_str = input("MNIST index (0-9999, default 0): ").strip()
                idx = int(idx_str) if idx_str else 0
                pixels, label = get_mnist_image(idx)
                print(f"MNIST label: {label}")
            elif choice == "2":
                path = input("Path to .npy: ").strip()
                pixels, _ = load_npy_image(path)
            elif choice == "3":
                path = input("Path to .bin: ").strip()
                pixels, _ = load_bin_image(path)
            else:
                print("Unknown choice.")
                continue

            print(f"Preview: {pixels[300:310]} ...")
            input("Press Enter to send via UART...")
            send_via_uart(pixels)
        except Exception as e:
            print(f"Error: {e}")
