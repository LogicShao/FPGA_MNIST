#!/usr/bin/env python
import argparse
import json
import os
import numpy as np


def load_mnist_raw(data_dir, train=False):
    split = "train" if train else "t10k"
    img_path = os.path.join(data_dir, "MNIST", "raw", f"{split}-images-idx3-ubyte")
    lbl_path = os.path.join(data_dir, "MNIST", "raw", f"{split}-labels-idx1-ubyte")
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        raise FileNotFoundError(f"MNIST raw files not found under {data_dir}")

    with open(img_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Bad MNIST image magic: {magic}")
        count = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")
        data = np.fromfile(f, dtype=np.uint8, count=count * rows * cols)
        images = data.reshape(count, rows, cols)

    with open(lbl_path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Bad MNIST label magic: {magic}")
        count_lbl = int.from_bytes(f.read(4), "big")
        labels = np.fromfile(f, dtype=np.uint8, count=count_lbl)

    if count_lbl != count:
        count = min(count, count_lbl)
        images = images[:count]
        labels = labels[:count]

    return images, labels


def read_mem_int8(path, count):
    values = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            values.append(int(s, 16) & 0xFF)
            if len(values) >= count:
                break
    if len(values) != count:
        raise ValueError(f"Expected {count} values in {path}, got {len(values)}")
    data = np.array(values, dtype=np.uint8)
    return data.view(np.int8)


def read_mem_int32(path, count):
    values = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            values.append(int(s, 16) & 0xFFFFFFFF)
            if len(values) >= count:
                break
    if len(values) != count:
        raise ValueError(f"Expected {count} values in {path}, got {len(values)}")
    data = np.array(values, dtype=np.uint32)
    return data.view(np.int32)


def to_int32(x):
    if np.isscalar(x):
        return int(np.int32(np.uint32(x)))
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim == 0:
        return int(np.int32(np.uint32(arr.item())))
    arr = arr.astype(np.uint32)
    return arr.view(np.int32)


def relu(x):
    return np.maximum(x, 0).astype(np.int32)

def relu_trunc64(acc):
    if acc < 0:
        return np.int32(0)
    return np.int32(np.uint32(acc))

def relu_trunc32(acc):
    tmp = to_int32(acc)
    if tmp < 0:
        return np.int32(0)
    return np.int32(tmp)


def quantize_int8(acc, mult, shift):
    # Fixed-point quantize: (acc * mult + rounding) >> shift, then saturate.
    prod = int(acc) * int(mult)
    prod += (1 << (shift - 1))
    shifted = prod >> shift
    if shifted > 127:
        return np.int8(127)
    if shifted < -128:
        return np.int8(-128)
    return np.int8(shifted)

def load_weights(weights_dir, use_int32_bias):
    w1 = read_mem_int8(f"{weights_dir}/CONV1_WEIGHTS.mem", 150).reshape(6, 25)
    w2 = read_mem_int8(f"{weights_dir}/CONV2_WEIGHTS.mem", 2400).reshape(16, 150)
    w3 = read_mem_int8(f"{weights_dir}/FC1_WEIGHTS.mem", 8192).reshape(32, 256)
    w4 = read_mem_int8(f"{weights_dir}/FC2_WEIGHTS.mem", 320).reshape(10, 32)

    if use_int32_bias:
        b1 = read_mem_int32(os.path.join(weights_dir, "CONV1_BIASES_INT32.mem"), 6)
        b2 = read_mem_int32(os.path.join(weights_dir, "CONV2_BIASES_INT32.mem"), 16)
        b3 = read_mem_int32(os.path.join(weights_dir, "FC1_BIASES_INT32.mem"), 32)
        b4 = read_mem_int32(os.path.join(weights_dir, "FC2_BIASES_INT32.mem"), 10)
    else:
        b1 = read_mem_int8(f"{weights_dir}/CONV1_BIASES.mem", 6)
        b2 = read_mem_int8(f"{weights_dir}/CONV2_BIASES.mem", 16)
        b3 = read_mem_int8(f"{weights_dir}/FC1_BIASES.mem", 32)
        b4 = read_mem_int8(f"{weights_dir}/FC2_BIASES.mem", 10)

    return w1, w2, w3, w4, b1, b2, b3, b4


def infer_one(img, w1, w2, w3, w4, b1, b2, b3, b4, quant=None, dump=False):
    out1 = np.zeros((6, 24, 24), dtype=np.int32)
    for oc in range(6):
        k = w1[oc].reshape(5, 5).astype(np.int64)
        for y in range(24):
            for x in range(24):
                patch = img[y:y + 5, x:x + 5].astype(np.int64)
                acc = np.sum(patch * k) + int(b1[oc])
                out1[oc, y, x] = relu_trunc32(acc)

    out1_q = None
    if quant:
        mult = quant["effective"]["conv1"]["mult"]
        shift = quant["shift"]
        out1_q = np.zeros_like(out1, dtype=np.int8)
        for ch in range(6):
            for y in range(24):
                for x in range(24):
                    out1_q[ch, y, x] = quantize_int8(out1[ch, y, x], mult, shift)
        pool1 = out1_q.reshape(6, 12, 2, 12, 2).max(axis=(2, 4)).astype(np.int32)
    else:
        pool1 = out1.reshape(6, 12, 2, 12, 2).max(axis=(2, 4))

    if dump:
        if out1_q is not None:
            print("Conv1 q (row2 col0..7, ch0..5):")
            for x in range(8):
                vals = [int(out1_q[ch, 2, x]) for ch in range(6)]
                print(f"  pos{48 + x}: " + " ".join(str(v) for v in vals))
            print("Conv1 q (row3 col2..3, ch0..5):")
            for x in range(2, 4):
                vals = [int(out1_q[ch, 3, x]) for ch in range(6)]
                idx = 3 * 24 + x
                print(f"  pos{idx}: " + " ".join(str(v) for v in vals))
            print("Conv1 q (row0/1 col14..19, ch0..5):")
            for y in range(2):
                for x in range(14, 20):
                    vals = [int(out1_q[ch, y, x]) for ch in range(6)]
                    idx = y * 24 + x
                    print(f"  pos{idx}: " + " ".join(str(v) for v in vals))
        print("Pool1 (first 16 positions, ch0..5):")
        pos_idx = 0
        for y in range(12):
            for x in range(12):
                if pos_idx >= 16:
                    break
                vals = [int(pool1[ch, y, x]) for ch in range(6)]
                print(f"  pos{pos_idx}: " + " ".join(str(v) for v in vals))
                pos_idx += 1
            if pos_idx >= 16:
                break
        print("Pool1 window (pos0, 5x5) ch0..5:")
        for ch in range(6):
            print(f"  ch{ch}:")
            for y in range(5):
                row = [int(pool1[ch, y, x]) for x in range(5)]
                print("   " + " ".join(str(v) for v in row))

    out2 = np.zeros((16, 8, 8), dtype=np.int32)
    for oc in range(16):
        k = w2[oc].reshape(6, 5, 5).astype(np.int64)
        for y in range(8):
            for x in range(8):
                patch = pool1[:, y:y + 5, x:x + 5].astype(np.int64)
                acc = np.sum(patch * k) + int(b2[oc])
                out2[oc, y, x] = relu_trunc64(acc)

    if quant:
        mult = quant["effective"]["conv2"]["mult"]
        shift = quant["shift"]
        out2_q = np.zeros_like(out2, dtype=np.int8)
        for ch in range(16):
            for y in range(8):
                for x in range(8):
                    out2_q[ch, y, x] = quantize_int8(out2[ch, y, x], mult, shift)
        pool2 = out2_q.reshape(16, 4, 2, 4, 2).max(axis=(2, 4)).astype(np.int32)
    else:
        pool2 = out2.reshape(16, 4, 2, 4, 2).max(axis=(2, 4))

    if dump:
        print("Conv2 out (pos0, pre-quant):")
        vals = [int(out2[ch, 0, 0]) for ch in range(16)]
        print("  " + " ".join(str(v) for v in vals))
        if quant:
            print("Conv2 q (first 4 positions, ch0..15):")
            pos_idx = 0
            for y in range(8):
                for x in range(8):
                    if pos_idx >= 4:
                        break
                    vals = [int(out2_q[ch, y, x]) for ch in range(16)]
                    print(f"  pos{pos_idx}: " + " ".join(str(v) for v in vals))
                    pos_idx += 1
                if pos_idx >= 4:
                    break
        print("Pool2 (first 4 positions, ch0..15):")
        pos_idx = 0
        for y in range(4):
            for x in range(4):
                if pos_idx >= 4:
                    break
                vals = [int(pool2[ch, y, x]) for ch in range(16)]
                print(f"  pos{pos_idx}: " + " ".join(str(v) for v in vals))
                pos_idx += 1
            if pos_idx >= 4:
                break

    fc_in = np.zeros(256, dtype=np.int32)
    idx = 0
    for ch in range(16):
        for y in range(4):
            for x in range(4):
                fc_in[idx] = pool2[ch, y, x]
                idx += 1

    fc1 = np.zeros(32, dtype=np.int32)
    for oc in range(32):
        acc = np.sum(fc_in.astype(np.int64) * w3[oc].astype(np.int64)) + int(b3[oc])
        fc1[oc] = relu_trunc64(acc)
    if quant:
        mult = quant["effective"]["fc1"]["mult"]
        shift = quant["shift"]
        fc1_q = np.zeros(32, dtype=np.int8)
        for oc in range(32):
            fc1_q[oc] = quantize_int8(fc1[oc], mult, shift)
        fc1 = fc1_q.astype(np.int32)

    if dump:
        print("FC1 outputs:")
        for i, v in enumerate(fc1.tolist()):
            print(f"  [{i}] = {v} (0x{(int(v) & 0xFFFFFFFF):08x})")

    fc2 = np.zeros(10, dtype=np.int32)
    for oc in range(10):
        acc = np.sum(fc1.astype(np.int64) * w4[oc].astype(np.int64)) + int(b4[oc])
        fc2[oc] = to_int32(acc)

    if dump:
        print("FC2:", fc2)
        print("PRED:", int(np.argmax(fc2)))

    return fc2


def quantize_image(img_f, normalize, s_in):
    if normalize:
        img_f = (img_f - 0.1307) / 0.3081
    scale = s_in
    img_int8 = np.clip(np.round(img_f * scale), -128, 127).astype(np.int8)
    return img_int8


def main():
    parser = argparse.ArgumentParser(description="Hardware-equivalent int8 inference")
    parser.add_argument("--image", default="hardware/src/v1.1/tb/test_image.mem")
    parser.add_argument("--weights", default="hardware/src/v1.1/rtl/weights")
    parser.add_argument("--quant-params", default=None, help="Path to quant_params.json")
    parser.add_argument("--batch", action="store_true", help="Run over MNIST test set.")
    parser.add_argument("--count", type=int, default=20, help="Number of images for batch mode.")
    parser.add_argument("--start", type=int, default=0, help="Start index for batch mode.")
    parser.add_argument("--normalize", action="store_true", help="Apply MNIST normalization.")
    parser.add_argument("--data-dir", default="model_tools/data", help="MNIST data directory.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-sample prints.")
    args = parser.parse_args()

    quant = None
    if args.quant_params:
        with open(args.quant_params, "r", encoding="utf-8") as f:
            quant = json.load(f)

    use_int32_bias = bool(quant) and os.path.exists(os.path.join(args.weights, "CONV1_BIASES_INT32.mem"))
    w1, w2, w3, w4, b1, b2, b3, b4 = load_weights(args.weights, use_int32_bias)

    if args.batch:
        if not quant:
            raise ValueError("--batch requires --quant-params")
        dataset = None
        images = None
        labels = None
        try:
            from torchvision import datasets, transforms
            dataset = datasets.MNIST(
                args.data_dir,
                train=False,
                download=False,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        except Exception:
            images, labels = load_mnist_raw(args.data_dir, train=False)
        total = 0
        correct = 0
        for idx in range(args.start, args.start + args.count):
            if dataset is not None:
                img, label = dataset[idx]
                img_f = img.numpy().squeeze()
                label = int(label)
            else:
                img_f = images[idx].astype(np.float32) / 255.0
                label = int(labels[idx])
            img_int8 = quantize_image(img_f, args.normalize, quant["scales"]["s_in"])
            fc2 = infer_one(img_int8, w1, w2, w3, w4, b1, b2, b3, b4, quant, dump=False)
            pred = int(np.argmax(fc2))
            match = int(pred == label)
            if not args.quiet:
                print(f"[{idx}] label={label} pred={pred} match={match}")
            total += 1
            correct += match
        acc = correct / total * 100.0 if total else 0.0
        print(f"Accuracy: {correct}/{total} = {acc:.2f}%")
        return

    img = read_mem_int8(args.image, 784).reshape(28, 28)
    infer_one(img, w1, w2, w3, w4, b1, b2, b3, b4, quant, dump=True)

    # Inference handled by infer_one().


if __name__ == "__main__":
    main()
