from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


VLLM_CSV = Path(
    "results/processed/vllm_bf16_concurrency_final.csv"
)

SGLANG_CSV = Path(
    "results/processed/sglang_bf16_concurrency_final.csv"
)

OUTPUT_DIR = Path("results/figures/backend_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    vllm = pd.read_csv(VLLM_CSV).sort_values("concurrency")
    sglang = pd.read_csv(SGLANG_CSV).sort_values("concurrency")

    return vllm, sglang


def plot_metric(
    vllm,
    sglang,
    column,
    ylabel,
    title,
    filename,
):
    plt.figure(figsize=(6, 4))

    plt.plot(
        vllm["concurrency"],
        vllm[column],
        marker="o",
        label="vLLM",
    )

    plt.plot(
        sglang["concurrency"],
        sglang[column],
        marker="o",
        label="SGLang",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.xticks(vllm["concurrency"])
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def plot_tradeoff(vllm, sglang):
    plt.figure(figsize=(6, 4))

    plt.plot(
        vllm["p50_ttft_ms"],
        vllm["output_throughput"],
        marker="o",
        label="vLLM",
    )

    plt.plot(
        sglang["p50_ttft_ms"],
        sglang["output_throughput"],
        marker="o",
        label="SGLang",
    )

    for _, row in vllm.iterrows():
        plt.annotate(
            f"C={int(row['concurrency'])}",
            (
                row["p50_ttft_ms"],
                row["output_throughput"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    for _, row in sglang.iterrows():
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
    plt.title("Backend Throughput–Latency Trade-off")

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
    vllm, sglang = load_results()

    plot_metric(
        vllm,
        sglang,
        "output_throughput",
        "Output Throughput (tokens/s)",
        "vLLM vs SGLang: Output Throughput",
        "throughput.png",
    )

    plot_metric(
        vllm,
        sglang,
        "p50_ttft_ms",
        "P50 TTFT (ms)",
        "vLLM vs SGLang: Time to First Token",
        "ttft_p50.png",
    )

    plot_metric(
        vllm,
        sglang,
        "p95_ttft_ms",
        "P95 TTFT (ms)",
        "vLLM vs SGLang: P95 Time to First Token",
        "ttft_p95.png",
    )

    plot_metric(
        vllm,
        sglang,
        "mean_tpot_ms",
        "Mean TPOT (ms/token)",
        "vLLM vs SGLang: Time per Output Token",
        "tpot.png",
    )

    plot_metric(
        vllm,
        sglang,
        "mean_e2el_ms",
        "Mean End-to-End Latency (ms)",
        "vLLM vs SGLang: End-to-End Latency",
        "e2e.png",
    )

    plot_metric(
        vllm,
        sglang,
        "peak_vram_mib",
        "Peak GPU Memory (MiB)",
        "vLLM vs SGLang: Peak GPU Memory",
        "vram.png",
    )

    plot_tradeoff(vllm, sglang)

    print(
        f"Saved backend comparison figures to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
