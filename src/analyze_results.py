import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = (
        df.groupby(["method", "batch_size", "context_len", "generation_len"], as_index=False)
        .agg(
            latency_mean_s=("latency_mean_s", "mean"),
            latency_p95_s=("latency_p95_s", "mean"),
            tokens_per_second=("tokens_per_second", "mean"),
        )
        .sort_values(["context_len", "batch_size", "generation_len", "method"])
    )
    summary.to_csv(outdir / "summary.csv", index=False)

    # Plot 1: tokens/sec by context length
    for batch_size in sorted(df["batch_size"].unique()):
        plt.figure()
        subset = df[df["batch_size"] == batch_size]
        for method, group in subset.groupby("method"):
            grouped = group.groupby("context_len")["tokens_per_second"].mean()
            plt.plot(grouped.index, grouped.values, marker="o", label=method)
        plt.xlabel("Context length")
        plt.ylabel("Generated tokens / second")
        plt.title(f"Throughput vs context length, batch={batch_size}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"throughput_context_batch_{batch_size}.png", dpi=160)
        plt.close()

    # Plot 2: latency by generation length
    for context_len in sorted(df["context_len"].unique()):
        plt.figure()
        subset = df[df["context_len"] == context_len]
        for method, group in subset.groupby("method"):
            grouped = group.groupby("generation_len")["latency_mean_s"].mean()
            plt.plot(grouped.index, grouped.values, marker="o", label=method)
        plt.xlabel("Generation length")
        plt.ylabel("Mean latency (s)")
        plt.title(f"Latency vs generation length, context={context_len}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"latency_generation_context_{context_len}.png", dpi=160)
        plt.close()

    print(summary.to_string(index=False))
    print(f"\nWrote plots and summary to {outdir}")


if __name__ == "__main__":
    main()
