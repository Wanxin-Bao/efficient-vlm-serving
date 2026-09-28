#!/usr/bin/env bash

set -e

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="assets/day2_benchmark.jsonl"

RESULT_DIR="results/raw/vllm_bf16_final"
GPU_DIR="results/raw/gpu_final"

mkdir -p "$RESULT_DIR"
mkdir -p "$GPU_DIR"

cleanup() {
    if [ -n "${GPU_MONITOR_PID:-}" ]; then
        kill "$GPU_MONITOR_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT

for C in 1 4 8 16
do

    echo
    echo "=========================================="
    echo "Concurrency = $C"
    echo "=========================================="

    #
    # Warm-up
    #
    echo "Running warm-up..."

    vllm bench serve \
        --backend openai-chat \
        --host 127.0.0.1 \
        --port 8000 \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --dataset-name custom_image \
        --dataset-path "$DATASET" \
        --custom-ensure-client-side-data \
        --custom-output-len 128 \
        --num-prompts 8 \
        --num-warmups 0 \
        --max-concurrency "$C" \
        --request-rate inf \
        --temperature 0 \
        --ignore-eos \
        > /dev/null

    echo "Warm-up completed."

    sleep 2

    #
    # GPU monitoring
    #
    GPU_LOG="$GPU_DIR/concurrency_${C}.csv"

    ./scripts/monitor_gpu.sh "$GPU_LOG" &

    GPU_MONITOR_PID=$!

    #
    # Formal benchmark
    #
    echo "Starting formal benchmark..."

    vllm bench serve \
        --backend openai-chat \
        --host 127.0.0.1 \
        --port 8000 \
        --endpoint /v1/chat/completions \
        --model "$MODEL" \
        --dataset-name custom_image \
        --dataset-path "$DATASET" \
        --custom-ensure-client-side-data \
        --custom-output-len 128 \
        --num-prompts 128 \
        --num-warmups 0 \
        --max-concurrency "$C" \
        --request-rate inf \
        --temperature 0 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --save-detailed \
        --result-dir "$RESULT_DIR" \
        --result-filename "concurrency_${C}.json"

    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true

    unset GPU_MONITOR_PID

    echo
    echo "Finished concurrency=$C"

    sleep 5

done

echo
echo "=========================================="
echo "All final benchmarks completed."
echo "=========================================="
