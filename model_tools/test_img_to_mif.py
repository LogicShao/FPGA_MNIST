#!/usr/bin/env python3
"""
Convert a test image file to .mif for ROM initialization.
Supported input: .bin, .npy, .txt, .csv, .h, .c
"""

import argparse
import os
import re
import numpy as np


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return text


def extract_braced(text, start_idx):
    depth = 0
    for i in range(start_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx: i + 1], i + 1
    raise ValueError("Unbalanced braces in initializer")


def load_from_c_header(path, symbol):
    text = strip_comments(
        open(path, "r", encoding="utf-8", errors="ignore").read())
    pattern = re.compile(
        r"%s\s*(\[[^\]]*\])*\s*=\s*\{" % re.escape(symbol), re.M)
    m = pattern.search(text)
    if not m:
        raise ValueError("Symbol not found: %s" % symbol)
    braced, _ = extract_braced(text, m.end() - 1)
    tokens = re.findall(r"[-+]?0x[0-9a-fA-F]+|[-+]?\d+", braced)
    values = [int(x, 0) for x in tokens]
    return np.array(values, dtype=np.int32)


def load_from_text(path):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not tokens:
        raise ValueError("No numeric tokens found")
    is_float = any("." in t or "e" in t or "E" in t for t in tokens)
    if is_float:
        return np.array([float(t) for t in tokens], dtype=np.float32)
    return np.array([int(t) for t in tokens], dtype=np.int32)


def normalize_pixels(arr):
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(np.max(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 127.0
    return arr.astype(np.int8).flatten()


def to_hex(value, width):
    mask = (1 << width) - 1
    unsigned = int(value) & mask
    hex_width = (width + 3) // 4
    return f"{unsigned:0{hex_width}X}"


def write_mif(path, values, width):
    values = [int(v) for v in np.asarray(values, dtype=np.int64).flatten()]
    depth = len(values)
    addr_width = max(1, (depth - 1).bit_length())
    addr_hex_width = (addr_width + 3) // 4

    with open(path, "w", newline="\n") as f:
        f.write(f"WIDTH={width};\n")
        f.write(f"DEPTH={depth};\n")
        f.write("ADDRESS_RADIX=HEX;\n")
        f.write("DATA_RADIX=HEX;\n")
        f.write("CONTENT BEGIN\n")
        for addr, v in enumerate(values):
            f.write(
                f"    {addr:0{addr_hex_width}X} : {to_hex(v, width)};\n")
        f.write("END;\n")


def load_input(path, symbol):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        data = np.load(path)
    elif ext == ".bin":
        data = np.fromfile(path, dtype=np.int8)
    elif ext in (".txt", ".csv"):
        data = load_from_text(path)
    elif ext in (".h", ".c"):
        data = load_from_c_header(path, symbol)
    else:
        raise ValueError("Unsupported input type: %s" % ext)
    return normalize_pixels(data)


def main():
    parser = argparse.ArgumentParser(
        description="Convert test image to .mif")
    parser.add_argument("--input", required=True,
                        help="Input test image file")
    parser.add_argument("--output", required=True, help="Output .mif path")
    parser.add_argument("--symbol", default="test_image",
                        help="C array symbol name")
    parser.add_argument("--size", type=int, default=784,
                        help="Expected size (default 784)")
    parser.add_argument("--width", type=int, default=8,
                        help="Data width in bits (default 8)")
    args = parser.parse_args()

    values = load_input(args.input, args.symbol)

    if args.size > 0 and values.size != args.size:
        raise ValueError(f"Expected {args.size} values, got {values.size}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_mif(args.output, values, args.width)
    print(f"Wrote {args.output} ({values.size} values)")


if __name__ == "__main__":
    main()
