import csv
import json
import re
from pathlib import Path

import numpy as np


RESULT_DIR = Path("results/raw/vllm_bf16_final")
GPU_DIR = Path("results/raw/gpu_final")
OUTPUT_DIR = Path("results/processed")

CONCURRENCIES = [1, 4, 8, 16]


def percentile_from_summary(data, metric, p):
    """
    Try several possible vLLM summary formats.
    Falls back to detailed per-request data.
    """

    # Possible direct key styles
    direct_keys = [
        f"p{p}_{metric}_ms",
        f"{metric}_p{p}_ms",
    ]

    for key in direct_keys:
        if key in data:
            return float(data[key])

    # Current vLLM-style percentile list/dict
    percentile_key = f"percentiles_{metric}_ms"

    if percentile_key in data:
        value = data[percentile_key]

        if isinstance(value, dict):
            for key in [str(p), f"p{p}", f"P{p}"]:
                if key in value:
                    return float(value[key])

        if isinstance(value, list):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    if float(item[0]) == float(p):
                        return float(item[1])

    # Fallback to detailed arrays.
    # vLLM detailed timing values are in seconds.
    detailed_map = {
        "ttft": ["ttfts"],
        "tpot": ["tpots"],
        "e2el": ["e2els", "latencies"],
    }

    for key in detailed_map.get(metric, []):
        if key in data and data[key]:
            return float(np.percentile(data[key], p) * 1000)

    return float("nan")


def get_mean_ms(data, metric):
    key = f"mean_{metric}_ms"

    if key in data:
        return float(data[key])

    detailed_map = {
        "ttft": ["ttfts"],
        "tpot": ["tpots"],
        "e2el": ["e2els", "latencies"],
    }

    for detailed_key in detailed_map.get(metric, []):
        if detailed_key in data and data[detailed_key]:
            return float(np.mean(data[detailed_key]) * 1000)

    return float("nan")


def parse_number(text):
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if match:
        return float(match.group())
    return float("nan")


def read_gpu_csv(path):
    memories = []
    utils = []
    powers = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            memory = parse_number(row[" memory.used [MiB]"])
            util = parse_number(row[" utilization.gpu [%]"])
            power = parse_number(row[" power.draw [W]"])

            memories.append(memory)
            utils.append(util)
            powers.append(power)

    memories = np.array(memories)
    utils = np.array(utils)
    powers = np.array(powers)

    # Use utilization > 0 as an approximate active window.
    active = utils > 0

    if active.any():
        active_utils = utils[active]
        active_powers = powers[active]
    else:
        active_utils = utils
        active_powers = powers

    return {
        "peak_vram_mib": float(np.nanmax(memories)),
        "mean_gpu_util_active": float(np.nanmean(active_utils)),
        "peak_gpu_util": float(np.nanmax(utils)),
        "mean_power_active_w": float(np.nanmean(active_powers)),
        "peak_power_w": float(np.nanmax(powers)),
    }


def read_benchmark(concurrency):
    path = RESULT_DIR / f"concurrency_{concurrency}.json"

    with open(path) as f:
        data = json.load(f)

    row = {
        "concurrency": concurrency,
        "completed": data.get("completed"),
        "failed": data.get("failed", 0),

        "request_throughput": data.get("request_throughput"),
        "output_throughput": data.get("output_throughput"),
        "total_token_throughput": data.get("total_token_throughput"),

        "mean_ttft_ms": get_mean_ms(data, "ttft"),
        "p50_ttft_ms": percentile_from_summary(data, "ttft", 50),
        "p95_ttft_ms": percentile_from_summary(data, "ttft", 95),
        "p99_ttft_ms": percentile_from_summary(data, "ttft", 99),

        "mean_tpot_ms": get_mean_ms(data, "tpot"),
        "p50_tpot_ms": percentile_from_summary(data, "tpot", 50),
        "p95_tpot_ms": percentile_from_summary(data, "tpot", 95),
        "p99_tpot_ms": percentile_from_summary(data, "tpot", 99),

        "mean_e2el_ms": get_mean_ms(data, "e2el"),
        "p50_e2el_ms": percentile_from_summary(data, "e2el", 50),
        "p95_e2el_ms": percentile_from_summary(data, "e2el", 95),
        "p99_e2el_ms": percentile_from_summary(data, "e2el", 99),
    }

    gpu_path = GPU_DIR / f"concurrency_{concurrency}.csv"
    row.update(read_gpu_csv(gpu_path))

    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [read_benchmark(c) for c in CONCURRENCIES]

    output_csv = OUTPUT_DIR / "vllm_bf16_concurrency_final.csv"

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n=== vLLM BF16 Concurrency Benchmark ===\n")

    for row in rows:
        print(
            f"C={row['concurrency']:>2} | "
            f"Req/s={row['request_throughput']:.2f} | "
            f"Tok/s={row['output_throughput']:.2f} | "
            f"TTFT p50={row['p50_ttft_ms']:.1f} ms | "
            f"TTFT p95={row['p95_ttft_ms']:.1f} ms | "
            f"TPOT={row['mean_tpot_ms']:.2f} ms | "
            f"E2E={row['mean_e2el_ms']:.1f} ms | "
            f"GPU={row['mean_gpu_util_active']:.1f}% | "
            f"VRAM={row['peak_vram_mib']:.0f} MiB"
        )

    print(f"\nSaved to: {output_csv}")


if __name__ == "__main__":
    main()
