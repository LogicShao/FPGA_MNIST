import serial
import time
import numpy as np

# --- Serial config (update COM port as needed) ---
SERIAL_PORT = "COM5"  # Windows example; Linux/Mac: /dev/ttyUSB0
BAUD_RATE = 115200


def _normalize_pixels(arr):
    arr = np.asarray(arr)
    if arr.size != 28 * 28:
        raise ValueError(f"Expected 784 pixels, got {arr.size}")
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = (arr * 127.0).astype(np.int8)
        else:
            arr = arr.astype(np.int8)
    else:
        arr = arr.astype(np.int8)
    return arr.flatten()


def get_mnist_image(index=0):
    # Lazy import to avoid slow torch load unless needed
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    img, label = dataset[index]
    img_np = (img.numpy().squeeze() * 127).astype(np.int8)
    return img_np.flatten(), label


def load_npy_image(path):
    data = np.load(path)
    return _normalize_pixels(data), None


def load_bin_image(path):
    raw = np.fromfile(path, dtype=np.int8)
    return _normalize_pixels(raw), None


def send_via_uart(data_bytes):
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Opened serial {SERIAL_PORT}")

        # Protocol: 0xAA (head) + 784 bytes + 0x55 (tail)
        packet = bytearray()
        packet.append(0xAA)
        packet.extend(data_bytes.tobytes())
        packet.append(0x55)

        print(f"Sending {len(packet)} bytes...")
        ser.write(packet)
        print("Send complete. Waiting for board response...")

        start_time = time.time()
        while time.time() - start_time < 5:
            if ser.in_waiting:
                line = ser.readline().decode("utf-8", errors="ignore")
                print(f"[FPGA]: {line.strip()}")

        ser.close()
    except Exception as e:
        print(f"Serial error: {e}")


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
