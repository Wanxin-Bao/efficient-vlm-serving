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
