import argparse
import csv
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _pick_column(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def _load_log(csv_path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header not found")
        fieldnames = reader.fieldnames
        epoch_col = _pick_column(fieldnames, ("epoch", "Epoch", "epochs"))
        loss_col = _pick_column(
            fieldnames,
            ("train_loss", "loss", "train_loss_avg", "trainLoss"),
        )
        acc_col = _pick_column(
            fieldnames,
            ("test_accuracy", "val_accuracy", "test_acc", "val_acc", "accuracy"),
        )
        lr_col = _pick_column(
            fieldnames,
            ("learning_rate", "lr", "learningRate"),
        )

        epochs = []
        losses = []
        accs = []
        lrs = []
        for i, row in enumerate(reader, start=1):
            if epoch_col:
                epochs.append(int(float(row[epoch_col])))
            else:
                epochs.append(i)
            if loss_col:
                losses.append(float(row[loss_col]))
            if acc_col:
                accs.append(float(row[acc_col]))
            if lr_col:
                lrs.append(float(row[lr_col]))

    return epochs, losses, accs, lrs


def _plot(epochs, losses, accs, lrs, title, out_path):
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    if losses:
        ax_loss.plot(epochs, losses, color="#1f77b4", linewidth=2, label="train loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, linestyle="--", alpha=0.4)
        ax_loss.legend(loc="best")
    else:
        ax_loss.text(0.5, 0.5, "No loss column", ha="center", va="center")
        ax_loss.set_axis_off()

    acc_is_fraction = accs and max(accs) <= 1.0
    acc_plot = [v * 100.0 for v in accs] if acc_is_fraction else accs
    if acc_plot:
        ax_acc.plot(epochs, acc_plot, color="#2ca02c", linewidth=2, label="test acc")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid(True, linestyle="--", alpha=0.4)
    else:
        ax_acc.text(0.5, 0.5, "No accuracy column", ha="center", va="center")
        ax_acc.set_axis_off()

    if lrs:
        ax_lr = ax_acc.twinx()
        ax_lr.plot(
            epochs,
            lrs,
            color="#7f7f7f",
            linestyle="--",
            linewidth=1.5,
            label="learning rate",
        )
        ax_lr.set_ylabel("LR")
        lines = ax_acc.get_lines() + ax_lr.get_lines()
        labels = [ln.get_label() for ln in lines]
        ax_acc.legend(lines, labels, loc="best")
    elif acc_plot:
        ax_acc.legend(loc="best")

    ax_acc.set_xlabel("Epoch")
    if title:
        fig.suptitle(title)

    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.9)
    fig.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(description="Plot training log CSV.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).parent / "logs" / "TinyLeNet_20251223_094124.csv"),
        help="Path to CSV log file",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output image path (png)",
    )
    parser.add_argument(
        "--title",
        default="TinyLeNet Training Curve",
        help="Figure title",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = Path(__file__).parent / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{csv_path.stem}_plot.png"

    epochs, losses, accs, lrs = _load_log(csv_path)
    _plot(epochs, losses, accs, lrs, args.title, out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
