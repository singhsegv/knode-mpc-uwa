"""
Parse train.log eval blocks and plot per-dimension metrics vs epoch.

Usage:
  python utils/plot_train_log.py
  python utils/plot_train_log.py --log runs/.../train.log --metric RMSE
  python utils/plot_train_log.py --log runs/.../train.log --metric MAE --save
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Script knobs (overridden by CLI when provided)
# ---------------------------------------------------------------------------
DEFAULT_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runs",
    "20260715_213721_evaluation1_basic1",
    "train.log",
)
METRIC = "RMSE"  # one of: MAE, RMSE, MSE, NRMSE
DIMS = ("q1 [rad]", "q2 [rad]", "qdot1 [rad/s]", "qdot2 [rad/s]")
EVAL_TYPES = ("Open-Loop", "Single-Steps")

EPOCH_RE = re.compile(r"^Epoch\s+(\d+),\s+Loss:")
EVAL_RE = re.compile(r"^\[TEST DATASET\]\s+Whole Trajectory\s+(Open-Loop|Single-Steps)\s*$")
COMBINED_RE = re.compile(r"^Combined MSE \(q\+qdot\):\s+([0-9.eE+-]+)")
# dim name may contain spaces/brackets; four floats follow
ROW_RE = re.compile(
    r"^(q1 \[rad\]|q2 \[rad\]|qdot1 \[rad/s\]|qdot2 \[rad/s\])\s+"
    r"([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*$"
)

METRIC_COL = {"MAE": 0, "RMSE": 1, "MSE": 2, "NRMSE": 3}


def parse_train_log(path):
    """
    Returns dict:
      {
        "Open-Loop": {
          "epochs": ndarray (K,),
          "combined_mse": ndarray (K,),
          "q1 [rad]": {"MAE": ..., "RMSE": ..., "MSE": ..., "NRMSE": ...},  # each (K,)
          ...
        },
        "Single-Steps": { ... },
      }
    """
    with open(path) as f:
        lines = f.read().splitlines()

    buckets = {
        name: {
            "epochs": [],
            "combined_mse": [],
            **{dim: {m: [] for m in METRIC_COL} for dim in DIMS},
        }
        for name in EVAL_TYPES
    }

    current_epoch = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        m_epoch = EPOCH_RE.match(line)
        if m_epoch:
            current_epoch = int(m_epoch.group(1))
            i += 1
            continue

        m_eval = EVAL_RE.match(line)
        if m_eval:
            if current_epoch is None:
                raise ValueError(f"Eval block at line {i + 1} before any Epoch line in {path}")

            eval_type = m_eval.group(1)
            i += 1

            # Combined MSE
            while i < len(lines) and not COMBINED_RE.match(lines[i].strip()):
                if EVAL_RE.match(lines[i].strip()) or EPOCH_RE.match(lines[i].strip()):
                    break
                i += 1
            if i >= len(lines) or not COMBINED_RE.match(lines[i].strip()):
                raise ValueError(f"Missing Combined MSE after {eval_type} @ epoch {current_epoch}")
            combined = float(COMBINED_RE.match(lines[i].strip()).group(1))
            i += 1

            # Skip header
            if i < len(lines) and lines[i].strip().startswith("dim"):
                i += 1

            rows = {}
            while i < len(lines):
                row_m = ROW_RE.match(lines[i].strip())
                if not row_m:
                    break
                dim = row_m.group(1)
                vals = [float(row_m.group(j)) for j in range(2, 6)]
                rows[dim] = vals
                i += 1

            if set(rows) != set(DIMS):
                raise ValueError(
                    f"Incomplete dim rows for {eval_type} @ epoch {current_epoch}: "
                    f"got {sorted(rows)}"
                )

            b = buckets[eval_type]
            b["epochs"].append(current_epoch)
            b["combined_mse"].append(combined)
            for dim, vals in rows.items():
                for metric, col in METRIC_COL.items():
                    b[dim][metric].append(vals[col])
            continue

        i += 1

    # Convert lists -> arrays
    out = {}
    for eval_type, b in buckets.items():
        out[eval_type] = {
            "epochs": np.asarray(b["epochs"], dtype=float),
            "combined_mse": np.asarray(b["combined_mse"], dtype=float),
        }
        for dim in DIMS:
            out[eval_type][dim] = {
                m: np.asarray(b[dim][m], dtype=float) for m in METRIC_COL
            }
    return out


def plot_metric(data, metric, log_path, save=False, show=True):
    metric = metric.upper()
    if metric not in METRIC_COL:
        raise ValueError(f"metric must be one of {list(METRIC_COL)}, got {metric}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle(f"{metric} vs epoch  ({os.path.basename(os.path.dirname(log_path))})", fontsize=12)

    for ax, eval_type in zip(axes, EVAL_TYPES):
        block = data[eval_type]
        epochs = block["epochs"]
        if len(epochs) == 0:
            ax.set_title(f"{eval_type} (no data)")
            continue

        for dim in DIMS:
            short = dim.split()[0]  # q1, q2, qdot1, qdot2
            ax.plot(epochs, block[dim][metric], marker="o", markersize=3, label=short)

        ax.set_title(eval_type)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # Open-loop numbers can span orders of magnitude
        if eval_type == "Open-Loop" or metric in ("MSE",):
            ax.set_yscale("log")

    fig.tight_layout()

    if save:
        out_dir = os.path.dirname(os.path.abspath(log_path))
        out_path = os.path.join(out_dir, f"eval_{metric.lower()}_vs_epoch.png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot eval metrics from train.log vs epoch.")
    parser.add_argument("--log", type=str, default=DEFAULT_LOG, help="Path to train.log")
    parser.add_argument(
        "--metric",
        type=str,
        default=METRIC,
        choices=list(METRIC_COL),
        help="Which column to plot on y-axis",
    )
    parser.add_argument("--save", action="store_true", help="Save PNG next to the log file")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        raise FileNotFoundError(args.log)

    data = parse_train_log(args.log)
    for eval_type in EVAL_TYPES:
        n = len(data[eval_type]["epochs"])
        print(f"{eval_type}: {n} eval checkpoints -> epochs {data[eval_type]['epochs'].tolist()}")

    plot_metric(data, args.metric, args.log, save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
