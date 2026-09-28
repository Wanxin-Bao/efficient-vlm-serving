from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path(
    "results/processed/vllm_bf16_concurrency_final.csv"
)

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("concurrency")
    return df


def plot_throughput(df):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df["concurrency"],
        df["output_throughput"],
        marker="o",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel("Output Throughput (tokens/s)")
    plt.title("vLLM Throughput vs Concurrency")
    plt.xticks(df["concurrency"])
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "throughput_vs_concurrency.png",
        dpi=200,
    )
    plt.close()


def plot_ttft(df):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df["concurrency"],
        df["p50_ttft_ms"],
        marker="o",
        label="P50",
    )

    plt.plot(
        df["concurrency"],
        df["p95_ttft_ms"],
        marker="o",
        label="P95",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel("TTFT (ms)")
    plt.title("Time to First Token vs Concurrency")
    plt.xticks(df["concurrency"])
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "ttft_vs_concurrency.png",
        dpi=200,
    )
    plt.close()


def plot_tpot(df):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df["concurrency"],
        df["mean_tpot_ms"],
        marker="o",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel("Mean TPOT (ms/token)")
    plt.title("Time per Output Token vs Concurrency")
    plt.xticks(df["concurrency"])
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "tpot_vs_concurrency.png",
        dpi=200,
    )
    plt.close()


def plot_e2e(df):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df["concurrency"],
        df["mean_e2el_ms"],
        marker="o",
    )

    plt.xlabel("Maximum Request Concurrency")
    plt.ylabel("Mean End-to-End Latency (ms)")
    plt.title("End-to-End Latency vs Concurrency")
    plt.xticks(df["concurrency"])
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "e2e_vs_concurrency.png",
        dpi=200,
    )
    plt.close()


def plot_tradeoff(df):
    plt.figure(figsize=(6, 4))

    plt.plot(
        df["p50_ttft_ms"],
        df["output_throughput"],
        marker="o",
    )

    for _, row in df.iterrows():
        plt.annotate(
            f"C={int(row['concurrency'])}",
            (
                row["p50_ttft_ms"],
                row["output_throughput"],
            ),
            xytext=(6, 5),
            textcoords="offset points",
        )

    plt.xlabel("P50 TTFT (ms)")
    plt.ylabel("Output Throughput (tokens/s)")
    plt.title("Throughput–Latency Trade-off")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "throughput_latency_tradeoff.png",
        dpi=200,
    )
    plt.close()


def main():
    df = load_data()

    print("\n=== Loaded benchmark results ===")
    print(
        df[
            [
                "concurrency",
                "output_throughput",
                "p50_ttft_ms",
                "p95_ttft_ms",
                "mean_tpot_ms",
                "mean_e2el_ms",
            ]
        ].to_string(index=False)
    )

    plot_throughput(df)
    plot_ttft(df)
    plot_tpot(df)
    plot_e2e(df)
    plot_tradeoff(df)

    print("\nSaved figures to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
