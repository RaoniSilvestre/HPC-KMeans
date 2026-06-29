#!/usr/bin/env python3
"""Generate performance graphs from results.csv.

For each dataset N in {1, 10, 100, 1000, 10000}, and for each implementation
(omp, mpi, mpi_omp), produce three plots: Execution Time, Speedup, Efficiency.
GPU is omitted from the per-implementation plots (constant 1 thread) and is
only included in the final per-dataset comparison graph.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = "results.csv"
OUT_DIR = Path("graphs")
OUT_DIR.mkdir(exist_ok=True)

DATASETS = [1, 10, 100, 1000, 10000]
IMPLS = ["omp", "gpu", "mpi", "mpi_omp", "cuda"]
IMPL_COLORS = {
    "omp": "#1f77b4",
    "gpu": "#ff7f0e",
    "mpi": "#2ca02c",
    "mpi_omp": "#d62728",
    "cuda": "#ffdd33",
}
IMPL_LABELS = {
    "omp": "OpenMP",
    "gpu": "GPU",
    "mpi": "MPI",
    "mpi_omp": "MPI+OpenMP",
    "cuda": "Cuda",
}


def get_data(df, impl, n):
    return df[(df["impl"] == impl) & (df["N"] == n)].copy()


def plot_time_speedup_efficiency(df, impl, n, x_col, x_label, out_prefix):
    sub = get_data(df, impl, n).sort_values(x_col)
    if sub.empty:
        return

    sub["total_cores"] = sub["threads"] * sub["processes"]

    if impl == "mpi_omp":
        baseline_time = sub[sub["processes"] == 1]["duration"].iloc[0]
    else:
        baseline_time = sub[sub[x_col] == 1]["duration"].iloc[0]

    x_vals = sub[x_col].values
    times = sub["duration"].values
    total_cores = sub["total_cores"].values
    speedups = baseline_time / times
    efficiencies = speedups / total_cores

    title_impl = IMPL_LABELS[impl]

    # Time vs X
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, times, marker="o", color="#1f77b4")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{title_impl} — Execution Time (N={n})")
    ax.grid(True, alpha=0.3)
    if impl == "mpi_omp":
        for x, t, row in zip(x_vals, times, sub.itertuples()):
            ax.annotate(
                f"({row.threads},{row.processes})",
                xy=(x, t),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                alpha=0.7,
            )
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_N{n}_time.png", dpi=120)
    plt.close()

    # Speedup
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, speedups, marker="o", color="#2ca02c", label="Measured")
    if impl != "mpi_omp":
        ax.plot(
            x_vals,
            total_cores,
            "--",
            color="gray",
            alpha=0.6,
            label="Ideal (linear)",
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Speedup")
    ax.set_title(f"{title_impl} — Speedup (N={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_N{n}_speedup.png", dpi=120)
    plt.close()

    # Efficiency
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, efficiencies, marker="o", color="#d62728", label="Measured")
    if impl != "mpi_omp":
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="Ideal (100%)")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Efficiency")
    ax.set_title(f"{title_impl} — Efficiency (N={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{out_prefix}_N{n}_efficiency.png", dpi=120)
    plt.close()


def plot_speedup_efficiency_comparison(df, n):
    fig, (ax_s, ax_e) = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        ("omp", "threads", "OpenMP", "-o", "#1f77b4"),
        ("mpi", "processes", "MPI", "--s", "#2ca02c"),
        ("mpi_omp", "processes", "MPI+OpenMP", ":^", "#d62728"),
    ]

    for impl, x_col, label, style, color in configs:
        sub = get_data(df, impl, n).sort_values(x_col)
        if sub.empty:
            continue

        sub["total_cores"] = sub["threads"] * sub["processes"]
        if impl == "mpi_omp":
            baseline_time = sub[sub["processes"] == 1]["duration"].iloc[0]
        else:
            baseline_time = sub[sub[x_col] == 1]["duration"].iloc[0]

        x_vals = sub[x_col].values
        total_cores = sub["total_cores"].values
        speedups = baseline_time / sub["duration"].values
        efficiencies = speedups / total_cores

        ax_s.plot(x_vals, speedups, style, color=color, label=label)
        ax_e.plot(x_vals, efficiencies, style, color=color, label=label)

    for ax, ylabel, title in [
        (ax_s, "Speedup", f"Speedup Comparison (N={n})"),
        (ax_e, "Efficiency", f"Efficiency Comparison (N={n})"),
    ]:
        ax.set_xlabel("Parallel workers (threads for OpenMP, processes otherwise)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"compare_speedup_efficiency_N{n}.png", dpi=120)
    plt.close()


def plot_comparison(df, n):
    labels, best_times = [], []
    for impl in IMPLS:
        sub = get_data(df, impl, n)
        if sub.empty:
            continue
        labels.append(impl)
        best_times.append(sub["duration"].min())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [IMPL_COLORS[impl] for impl in labels]
    bars = ax.bar([IMPL_LABELS[i] for i in labels], best_times, color=colors)
    ax.set_ylabel("Best Time (s)")
    ax.set_title(f"Best Execution Time Comparison (N={n})")
    ax.set_yscale("log")
    for bar, t in zip(bars, best_times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, alpha=0.3, axis="y", which="both")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"comparison_N{n}.png", dpi=120)
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)

    for n in DATASETS:
        plot_time_speedup_efficiency(df, "omp", n, "threads", "Threads", "omp")
        plot_time_speedup_efficiency(df, "mpi", n, "processes", "Processes", "mpi")
        plot_time_speedup_efficiency(
            df, "mpi_omp", n, "processes", "Processes", "mpi_omp"
        )

    for n in DATASETS:
        plot_comparison(df, n)
        plot_speedup_efficiency_comparison(df, n)

    n_graphs = len(list(OUT_DIR.glob("*.png")))
    print(f"Generated {n_graphs} graphs in {OUT_DIR}/")
    for f in sorted(OUT_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
