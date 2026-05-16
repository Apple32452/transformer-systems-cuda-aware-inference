# Transformer Systems & CUDA-Aware Inference Optimization

A compact research-engineering project for studying transformer decoding latency, KV-cache reuse, batching, sequence-length scaling, and GPU/CPU/MPS bottlenecks.

This project is designed for a Meta Research Engineer / AI Systems resume. It gives you real scripts to benchmark transformer inference behavior instead of only describing it.

## What this project demonstrates

- Transformer causal self-attention implementation in PyTorch
- Naive full-context decoding vs incremental decoding with KV cache
- Token-level latency and throughput measurement
- Batch-size and sequence-length scaling experiments
- Optional vLLM offline inference benchmark
- CSV/JSON outputs that can be plotted and cited in your resume/GitHub

## Repository structure

```text
transformer-systems-cuda-aware-inference/
├── configs/
│   └── benchmark_config.yaml
├── src/
│   ├── tiny_decoder.py
│   ├── benchmark_torch_decode.py
│   ├── benchmark_vllm.py
│   └── analyze_results.py
├── results/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Apple Silicon, PyTorch can use `mps` when available. For NVIDIA GPUs, use a CUDA-enabled PyTorch install.

## Run the PyTorch decoding benchmark

```bash
python src/benchmark_torch_decode.py --config configs/benchmark_config.yaml --out results/torch_decode_results.csv
python src/analyze_results.py --csv results/torch_decode_results.csv --outdir results
```

This compares:

1. **Naive decoding**: recomputes attention over the full prefix at every generation step.
2. **KV-cache decoding**: reuses previous keys and values, computing only the newest token representation.

## Optional: run vLLM offline inference benchmark

Install vLLM only if your environment supports it:

```bash
pip install vllm
python src/benchmark_vllm.py --model facebook/opt-125m --prompts 32 --max-tokens 32 --out results/vllm_results.json
```

## Interpreting results

Expected pattern:

- Naive decoding becomes slower as sequence length grows because it repeatedly recomputes attention over the prefix.
- KV-cache decoding should reduce per-token decoding cost, especially for longer generated sequences.
- Larger batches can improve throughput but may increase latency depending on hardware memory bandwidth.
- vLLM should be treated as the production-style reference path, while the PyTorch implementation helps explain the underlying systems idea.

## Resume bullets after running experiments

Use only after you run the benchmark and replace placeholders with your measured numbers:

- Implemented a PyTorch transformer decoding benchmark comparing full-prefix attention against KV-cache incremental decoding across batch size, context length, and generation length.
- Profiled token-level latency and throughput, identifying sequence-length and memory-bandwidth bottlenecks in autoregressive decoding.
- Evaluated batching, sequence bucketing, mixed precision, and optional vLLM offline inference to study throughput/latency tradeoffs for LLM serving workloads.

## Notes

This is intentionally small enough to run locally but structured like a serious research-engineering benchmark.
