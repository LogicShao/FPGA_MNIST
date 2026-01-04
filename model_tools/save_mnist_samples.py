import argparse
from pathlib import Path

from torchvision import datasets


def main():
    parser = argparse.ArgumentParser(description="Save MNIST samples as images.")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "report" / "figures" / "real"),
        help="Output directory for saved images",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of images to save",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in the dataset",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Use training split (default: test split)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download MNIST if missing",
    )
    parser.add_argument(
        "--prefix",
        default="mnist",
        help="Filename prefix",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent / "data"
    dataset = datasets.MNIST(
        str(data_dir),
        train=args.train,
        download=args.download,
    )

    if args.start < 0 or args.count <= 0:
        raise ValueError("start must be >= 0 and count must be > 0")
    if args.start + args.count > len(dataset):
        raise ValueError("requested range exceeds dataset length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.start, args.start + args.count):
        img, label = dataset[idx]
        filename = f"{args.prefix}_idx{idx}_label{label}.png"
        out_path = out_dir / filename
        img.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
