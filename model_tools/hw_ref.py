#!/usr/bin/env python
import argparse
import json
import os
import numpy as np


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

def main():
    parser = argparse.ArgumentParser(description="Hardware-equivalent int8 inference")
    parser.add_argument("--image", default="hardware/src/v1.1/tb/test_image.mem")
    parser.add_argument("--weights", default="hardware/src/v1.1/rtl/weights")
    parser.add_argument("--quant-params", default=None, help="Path to quant_params.json")
    args = parser.parse_args()

    quant = None
    if args.quant_params:
        with open(args.quant_params, "r", encoding="utf-8") as f:
            quant = json.load(f)

    img = read_mem_int8(args.image, 784).reshape(28, 28)

    w1 = read_mem_int8(f"{args.weights}/CONV1_WEIGHTS.mem", 150).reshape(6, 25)
    w2 = read_mem_int8(f"{args.weights}/CONV2_WEIGHTS.mem", 2400).reshape(16, 150)
    w3 = read_mem_int8(f"{args.weights}/FC1_WEIGHTS.mem", 8192).reshape(32, 256)
    w4 = read_mem_int8(f"{args.weights}/FC2_WEIGHTS.mem", 320).reshape(10, 32)

    b1_int32 = os.path.join(args.weights, "CONV1_BIASES_INT32.mem")
    b2_int32 = os.path.join(args.weights, "CONV2_BIASES_INT32.mem")
    b3_int32 = os.path.join(args.weights, "FC1_BIASES_INT32.mem")
    b4_int32 = os.path.join(args.weights, "FC2_BIASES_INT32.mem")

    if quant and os.path.exists(b1_int32):
        b1 = read_mem_int32(b1_int32, 6)
        b2 = read_mem_int32(b2_int32, 16)
        b3 = read_mem_int32(b3_int32, 32)
        b4 = read_mem_int32(b4_int32, 10)
    else:
        b1 = read_mem_int8(f"{args.weights}/CONV1_BIASES.mem", 6)
        b2 = read_mem_int8(f"{args.weights}/CONV2_BIASES.mem", 16)
        b3 = read_mem_int8(f"{args.weights}/FC1_BIASES.mem", 32)
        b4 = read_mem_int8(f"{args.weights}/FC2_BIASES.mem", 10)

    out1 = np.zeros((6, 24, 24), dtype=np.int32)
    for oc in range(6):
        k = w1[oc].reshape(5, 5).astype(np.int64)
        for y in range(24):
            for x in range(24):
                patch = img[y:y + 5, x:x + 5].astype(np.int64)
                acc = np.sum(patch * k) + int(b1[oc])
                out1[oc, y, x] = relu_trunc32(acc)

    pool1 = out1.reshape(6, 12, 2, 12, 2).max(axis=(2, 4))
    if quant:
        mult = quant["effective"]["conv1"]["mult"]
        shift = quant["shift"]
        pool1_q = np.zeros_like(pool1, dtype=np.int8)
        for ch in range(6):
            for y in range(12):
                for x in range(12):
                    pool1_q[ch, y, x] = quantize_int8(pool1[ch, y, x], mult, shift)
        pool1 = pool1_q.astype(np.int32)

    print("Pool1 (first 4 positions, ch0..5):")
    pos_idx = 0
    for y in range(12):
        for x in range(12):
            if pos_idx >= 4:
                break
            vals = [int(pool1[ch, y, x]) for ch in range(6)]
            print(f"  pos{pos_idx}: " + " ".join(str(v) for v in vals))
            pos_idx += 1
        if pos_idx >= 4:
            break

    out2 = np.zeros((16, 8, 8), dtype=np.int32)
    for oc in range(16):
        k = w2[oc].reshape(6, 5, 5).astype(np.int64)
        for y in range(8):
            for x in range(8):
                patch = pool1[:, y:y + 5, x:x + 5].astype(np.int64)
                acc = np.sum(patch * k) + int(b2[oc])
                out2[oc, y, x] = relu_trunc64(acc)

    pool2 = out2.reshape(16, 4, 2, 4, 2).max(axis=(2, 4))
    if quant:
        mult = quant["effective"]["conv2"]["mult"]
        shift = quant["shift"]
        pool2_q = np.zeros_like(pool2, dtype=np.int8)
        for ch in range(16):
            for y in range(4):
                for x in range(4):
                    pool2_q[ch, y, x] = quantize_int8(pool2[ch, y, x], mult, shift)
        pool2 = pool2_q.astype(np.int32)

    fc_in = np.zeros(256, dtype=np.int32)
    idx = 0
    for ch in range(16):
        for y in range(4):
            for x in range(4):
                fc_in[idx] = pool2[ch, y, x]
                idx += 1

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

    print("FC1 outputs:")
    for i, v in enumerate(fc1.tolist()):
        print(f"  [{i}] = {v} (0x{(int(v) & 0xFFFFFFFF):08x})")

    fc2 = np.zeros(10, dtype=np.int32)
    for oc in range(10):
        acc = np.sum(fc1.astype(np.int64) * w4[oc].astype(np.int64)) + int(b4[oc])
        fc2[oc] = to_int32(acc)

    print("FC2:", fc2)
    print("PRED:", int(np.argmax(fc2)))


if __name__ == "__main__":
    main()
