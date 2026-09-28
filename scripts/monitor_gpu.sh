#!/usr/bin/env bash

OUTPUT_FILE=$1

if [ -z "$OUTPUT_FILE" ]; then
    echo "Usage: $0 output.csv"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

exec nvidia-smi \
    --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
    --format=csv \
    -lms 200 \
    > "$OUTPUT_FILE"
