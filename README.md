# Efficient VLM Serving

A deployment-oriented benchmark project for vision-language models
using vLLM, SGLang, quantization, and multimodal evaluation.

## Baseline Model

- Qwen/Qwen2.5-VL-3B-Instruct

## Current Hardware

- NVIDIA GeForce RTX 3090
- 24 GB VRAM

## Progress

### Day 1 — vLLM Baseline

- [x] Created isolated Python environment
- [x] Installed CUDA 12.9 build of PyTorch and vLLM
- [x] Launched Qwen2.5-VL-3B-Instruct with vLLM
- [x] Verified OpenAI-compatible API endpoint
- [x] Sent a local image through the API
- [x] Received a correct multimodal response

## Planned Work

- vLLM performance benchmarking
- SGLang performance benchmarking
- BF16 vs quantized inference
- Multimodal quality evaluation
- Quality / latency / throughput / memory trade-off analysis


## Day 2 — vLLM Concurrency Benchmark

### Objective

The goal of Day 2 is to characterize the throughput-latency trade-off of
serving a vision-language model with vLLM under different request concurrency
levels.

The experiment focuses on a single controlled variable:

**maximum request concurrency**.

All other model, input, decoding, and server configurations are kept fixed.

---

### Experimental Setup

#### Model

- Qwen/Qwen2.5-VL-3B-Instruct
- Precision: BF16
- Serving backend: vLLM
- OpenAI-compatible chat completion API

#### Hardware

- NVIDIA GeForce RTX 3090
- 24 GB VRAM
- CUDA Toolkit 12.9
- PyTorch 2.13.0+cu129
- vLLM 0.30.0+cu129

#### Server Configuration

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --no-enable-prefix-caching
```

Prefix caching is disabled to avoid artificially improving repeated benchmark
requests through prefix KV-cache reuse.

#### Benchmark Workload

- Number of measured requests: 128
- Warm-up requests executed before each formal benchmark
- One image per request
- Image resolution: 640 × 640
- Fixed text prompt
- Output length: 128 tokens
- Temperature: 0
- EOS ignored to keep output length consistent
- Request rate: unlimited (`inf`)
- Concurrency levels: 1, 4, 8, 16

The benchmark dataset contains 128 image-prompt pairs. Images are kept at the
same resolution while being made byte-wise distinct to reduce unintended
multimodal cache reuse.

#### Metrics

The following serving metrics are measured:

- Request throughput
- Output token throughput
- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Inter-token Latency (ITL)
- End-to-End Latency (E2E)
- Peak GPU memory usage
- GPU utilization
- Request failure rate

GPU statistics are sampled every 200 ms using `nvidia-smi`.

---

### Results

| Concurrency | Request Throughput (req/s) | Output Throughput (tok/s) | TTFT P50 (ms) | TTFT P95 (ms) | Mean TPOT (ms/token) | Mean E2E (ms) | Peak VRAM (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.77 | 98.92  | 165.7  | 171.2  | 8.94  | 1293.3 | 18111 |
| 4  | 2.18 | 279.61 | 507.7  | 517.9  | 11.05 | 1828.7 | 18111 |
| 8  | 3.53 | 452.33 | 644.0  | 951.6  | 12.16 | 2262.0 | 18111 |
| 16 | 5.00 | 640.20 | 1184.0 | 1782.0 | 15.81 | 3194.7 | 18167 |

All 128 requests completed successfully at every tested concurrency level.

---

### Benchmark Figures

#### Output Throughput

![Throughput vs Concurrency](results/figures/throughput_vs_concurrency.png)

#### Time to First Token

![TTFT vs Concurrency](results/figures/ttft_vs_concurrency.png)

#### Time per Output Token

![TPOT vs Concurrency](results/figures/tpot_vs_concurrency.png)

#### End-to-End Latency

![End-to-End Latency](results/figures/e2e_vs_concurrency.png)

#### Throughput–Latency Trade-off

![Throughput-Latency Trade-off](results/figures/throughput_latency_tradeoff.png)

---

### Analysis

#### Throughput Scaling

Increasing request concurrency substantially improves aggregate serving
throughput.

Relative to concurrency 1:

| Concurrency | Output Throughput (tok/s) | Relative Throughput |
|---:|---:|---:|
| 1  | 98.92  | 1.00× |
| 4  | 279.61 | 2.83× |
| 8  | 452.33 | 4.57× |
| 16 | 640.20 | 6.47× |

The throughput gain is clearly sub-linear. Increasing concurrency enables vLLM
to batch more requests together and process more tokens in parallel, but
doubling concurrency does not result in a proportional doubling of throughput.

For example, increasing concurrency from 8 to 16 improves output throughput
from 452.33 to 640.20 tokens/s, corresponding to an increase of approximately
41.5%.

This indicates diminishing throughput gains as the system is pushed toward
higher concurrency.

#### Throughput–Latency Trade-off

The increase in throughput comes at the cost of higher per-request latency.

Median TTFT increases substantially with concurrency:

| Concurrency | TTFT P50 (ms) | TTFT P95 (ms) |
|---:|---:|---:|
| 1  | 165.7  | 171.2  |
| 4  | 507.7  | 517.9  |
| 8  | 644.0  | 951.6  |
| 16 | 1184.0 | 1782.0 |

The transition from concurrency 8 to 16 is particularly notable:

- Output throughput increases by approximately **41.5%**
- Median TTFT increases by approximately **83.9%**
- Mean E2E latency increases by approximately **41.2%**
- Mean TPOT increases from **12.16 ms/token to 15.81 ms/token**

Therefore, higher concurrency continues to improve aggregate throughput, but
the marginal throughput gain becomes increasingly expensive in terms of
user-facing latency.

For this specific model, hardware configuration, and workload, concurrency
around 8 appears to be an observed knee of the throughput-latency curve.

This should not be interpreted as a universally optimal concurrency level.
The appropriate operating point depends on the latency requirements and
throughput objectives of the target application.

#### Understanding TPOT

TPOT measures the average time required to generate each output token after
the first token has been produced.

TPOT increases from:

- 8.94 ms/token at concurrency 1
- 11.05 ms/token at concurrency 4
- 12.16 ms/token at concurrency 8
- 15.81 ms/token at concurrency 16

This means that individual requests generate tokens more slowly as concurrency
increases.

However, the server is simultaneously generating tokens for more requests.
As a result, aggregate output throughput increases even though per-request
token generation latency becomes worse.

This distinction illustrates an important serving-system trade-off:

**lower per-request latency and higher aggregate throughput are different
optimization objectives.**

#### GPU Behavior

Peak GPU memory usage remains nearly constant across concurrency levels:

| Concurrency | Peak VRAM (MiB) |
|---:|---:|
| 1  | 18111 |
| 4  | 18111 |
| 8  | 18111 |
| 16 | 18167 |

This behavior is expected because vLLM reserves a substantial GPU memory pool
for model weights, KV cache, CUDA graphs, and runtime workspaces according to
the configured GPU memory utilization limit.

Therefore, increasing request concurrency does not directly produce a
proportional increase in total allocated GPU memory.

During active inference periods, `nvidia-smi` reports approximately 97% GPU
utilization across the tested concurrency levels.

However, this metric should be interpreted carefully.

`nvidia-smi` GPU utilization primarily indicates whether GPU kernels are active
during a sampling interval. It does not directly measure:

- SM occupancy
- Tensor Core utilization
- achieved FLOPS
- memory bandwidth utilization
- overall computational efficiency

Therefore, high GPU utilization at concurrency 1 does not imply that the GPU's
full compute capacity is already efficiently utilized.

At low concurrency, the GPU may execute kernels almost continuously while the
individual operations remain relatively small.

Higher concurrency allows vLLM to form larger batches and exploit GPU
parallelism more effectively.

This is reflected directly in the throughput results:

- Concurrency 1: **98.92 output tokens/s**
- Concurrency 8: **452.33 output tokens/s**
- Concurrency 16: **640.20 output tokens/s**

Thus, similar `nvidia-smi` utilization values can correspond to very different
levels of useful serving throughput.

#### Tail Latency

The benchmark also shows that latency variation becomes more pronounced at
higher concurrency.

For example, at concurrency 16:

- TTFT P50: **1184.0 ms**
- TTFT P95: **1782.0 ms**

The difference between median and tail latency indicates that some requests
experience substantially longer waiting time under higher server load.

For an online serving system, this is important because average latency alone
does not fully describe user experience. Tail metrics such as P95 and P99 are
therefore important when evaluating deployment configurations.

---

### Key Findings

1. **Continuous batching substantially improves aggregate throughput.**

   Increasing concurrency from 1 to 16 improves output throughput from
   98.92 to 640.20 tokens/s, corresponding to a **6.47× increase**.

2. **Throughput scaling is sub-linear.**

   Higher concurrency continues to improve throughput, but the marginal gain
   decreases as concurrency increases.

3. **Higher throughput comes with higher latency.**

   TTFT, TPOT, and E2E latency all increase as more requests are processed
   concurrently.

4. **Concurrency 8 represents an observed throughput-latency knee for this
   workload.**

   Increasing concurrency from 8 to 16 provides approximately 41.5% more
   output throughput, while median TTFT increases by approximately 83.9%.

5. **High `nvidia-smi` utilization does not necessarily imply full compute
   efficiency.**

   The GPU can remain continuously active even when relatively small kernels
   do not fully exploit its available parallelism. Larger serving batches can
   therefore improve throughput even when reported GPU utilization is already
   close to 100%.

6. **Peak VRAM usage is largely insensitive to request concurrency.**

   vLLM pre-allocates a substantial GPU memory pool, so changes in concurrency
   mainly affect scheduling and compute behavior rather than total allocated
   memory.

7. **The server remained stable under the tested workload.**

   All 128 requests completed successfully at every tested concurrency level,
   with no failed requests.

---

### Reproducing the Experiment

#### 1. Activate the Environment

```bash
source .venv-vllm/bin/activate
source scripts/use_cuda_12_9.sh
```

#### 2. Prepare the Benchmark Dataset

```bash
python scripts/prepare_day2_dataset.py
```

The generated benchmark dataset is stored at:

```text
assets/day2_benchmark.jsonl
```

#### 3. Start the vLLM Server

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --no-enable-prefix-caching
```

#### 4. Run the Concurrency Benchmark

In another terminal:

```bash
source .venv-vllm/bin/activate
source scripts/use_cuda_12_9.sh

./scripts/run_vllm_concurrency_benchmark.sh
```

The benchmark evaluates:

```text
concurrency = 1, 4, 8, 16
```

with 128 measured requests for each configuration.

#### 5. Aggregate the Results

```bash
python src/aggregate_results.py
```

The processed results are written to:

```text
results/processed/vllm_bf16_concurrency_final.csv
```

#### 6. Generate Benchmark Figures

```bash
python src/plot_results.py
```

Generated figures are stored under:

```text
results/figures/
```

---

### Day 2 Summary

Day 2 establishes the first controlled serving benchmark for the project.

The experiment demonstrates that increasing request concurrency allows vLLM
to achieve substantially higher aggregate VLM throughput through batching, but
the improvement comes with increasingly higher TTFT, TPOT, and end-to-end
latency.

The results establish a BF16 vLLM baseline that will be used for subsequent
backend and optimization comparisons.
