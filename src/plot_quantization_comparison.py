from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BF16_CSV = Path(
    "results/processed/vllm_bf16_concurrency_final.csv"
)

AWQ_CSV = Path(
    "results/processed/vllm_awq_concurrency_final.csv"
)

OUTPUT_DIR = Path(
    "results/figures/quantization_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CONCURRENCIES = [1, 8, 16]


def load_results():
    bf16 = pd.read_csv(BF16_CSV)
    awq = pd.read_csv(AWQ_CSV)

    bf16 = bf16[
        bf16["concurrency"].isin(CONCURRENCIES)
    ].sort_values("concurrency")

    awq = awq[
        awq["concurrency"].isin(CONCURRENCIES)
    ].sort_values("concurrency")

    return bf16, awq


def plot_metric(
    bf16,
    awq,
    column,
    ylabel,
    title,
    filename,
):
    plt.figure(figsize=(6, 4))

    plt.plot(
        bf16["concurrency"],
        bf16[column],
        marker="o",
        label="BF16",
    )

    plt.plot(
        awq["concurrency"],
        awq[column],
        marker="o",
        label="AWQ INT4",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(CONCURRENCIES)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_tradeoff(bf16, awq):
    plt.figure(figsize=(6, 4))

    plt.plot(
        bf16["p50_ttft_ms"],
        bf16["output_throughput"],
        marker="o",
        label="BF16",
    )

    plt.plot(
        awq["p50_ttft_ms"],
        awq["output_throughput"],
        marker="o",
        label="AWQ INT4",
    )

    for _, row in bf16.iterrows():
        plt.annotate(
            f"C={int(row['concurrency'])}",
            (
                row["p50_ttft_ms"],
                row["output_throughput"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    for _, row in awq.iterrows():
        plt.annotate(
            f"C={int(row['concurrency'])}",
            (
                row["p50_ttft_ms"],
                row["output_throughput"],
            ),
            xytext=(5, -12),
            textcoords="offset points",
        )

    plt.xlabel("P50 TTFT (ms)")
    plt.ylabel("Output Throughput (tokens/s)")
    plt.title("BF16 vs AWQ: Throughput–Latency Trade-off")

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "throughput_latency_tradeoff.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main():
    bf16, awq = load_results()

    plot_metric(
        bf16,
        awq,
        "output_throughput",
        "Output Throughput (tokens/s)",
        "BF16 vs AWQ: Output Throughput",
        "throughput.png",
    )

    plot_metric(
        bf16,
        awq,
        "p50_ttft_ms",
        "P50 TTFT (ms)",
        "BF16 vs AWQ: Time to First Token",
        "ttft.png",
    )

    plot_metric(
        bf16,
        awq,
        "mean_tpot_ms",
        "Mean TPOT (ms/token)",
        "BF16 vs AWQ: Time per Output Token",
        "tpot.png",
    )

    plot_metric(
        bf16,
        awq,
        "mean_e2el_ms",
        "Mean End-to-End Latency (ms)",
        "BF16 vs AWQ: End-to-End Latency",
        "e2e.png",
    )

    plot_metric(
        bf16,
        awq,
        "peak_vram_mib",
        "Peak GPU Memory (MiB)",
        "BF16 vs AWQ: Peak Process GPU Memory",
        "vram.png",
    )

    plot_tradeoff(bf16, awq)

    print(
        "Saved quantization comparison figures to:",
        OUTPUT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
