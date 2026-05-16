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

# Transformer Systems & CUDA-Aware Inference Optimization

## Summary

This project implements a compact transformer inference benchmark for studying token-level latency, throughput, batching behavior, context-length scaling, and KV-cache reuse during autoregressive decoding.

The benchmark compares two decoding strategies:

- `naive_full_prefix`: recomputes the full prefix during every decoding step.
- `kv_cache`: reuses cached key/value tensors to avoid redundant attention computation over previous tokens.

The experiment was run on Apple Silicon using the PyTorch MPS backend with float32 precision. The benchmark generated 36 measurements across different batch sizes, context lengths, generation lengths, and decoding methods.

The main result is that KV-cache decoding becomes significantly faster as batch size and context length grow. In the strongest setting tested, KV-cache decoding achieved approximately **4.87x higher throughput** than naive full-prefix decoding.

## Project Goal

Autoregressive transformer inference is often bottlenecked by repeated attention computation during token-by-token generation. Without caching, each new token requires the model to repeatedly process the full prefix. KV caching avoids this redundant computation by storing previous attention keys and values.

This project demonstrates how inference performance changes as a function of:

- Batch size
- Context length
- Generation length
- Decoding method
- Latency distribution
- Tokens-per-second throughput

The goal is to provide a reproducible systems benchmark that connects transformer architecture behavior to practical model-serving performance.

## How to Run

From the project directory:

```bash
cd ~/Downloads/meta_research_engineer_projects/transformer-systems-cuda-aware-inference
```

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the benchmark:

```bash
python src/benchmark_torch_decode.py \
  --config configs/benchmark_config.yaml \
  --out results/torch_decode_results.csv
```

Analyze the results and generate plots:

```bash
python src/analyze_results.py \
  --csv results/torch_decode_results.csv \
  --outdir results
```

Open the results folder on macOS:

```bash
open results
```

## Output Files

After running the benchmark and analysis scripts, the following files are generated:

```text
results/
├── torch_decode_results.csv
├── summary.csv
├── latency_generation_context_32.png
├── latency_generation_context_128.png
├── latency_generation_context_256.png
├── throughput_context_batch_1.png
├── throughput_context_batch_4.png
└── throughput_context_batch_8.png
```

The CSV files contain the raw and summarized benchmark results. The PNG files visualize latency and throughput trends across context lengths, batch sizes, and generation lengths.

## Results

The benchmark successfully produced 36 inference measurements across `naive_full_prefix` and `kv_cache` decoding.

The measured fields include:

- Decoding method
- Device
- Data type
- Batch size
- Context length
- Generation length
- Number of generated tokens
- Mean latency
- p50 latency
- p95 latency
- Latency standard deviation
- Tokens per second

The benchmark was run on:

| Setting | Value |
|---|---|
| Device | Apple Silicon MPS |
| Precision | float32 |
| Number of Benchmark Rows | 36 |
| Decoding Methods | `naive_full_prefix`, `kv_cache` |
| Batch Sizes | 1, 4, 8 |
| Context Lengths | 32, 128, 256 |
| Generation Lengths | 16, 32 |

## Key Result

The strongest throughput improvement appeared at:

| Parameter | Value |
|---|---|
| Batch Size | 8 |
| Context Length | 256 |
| Generation Length | 32 |
| Tokens Generated | 256 |

Performance comparison:

| Method | Mean Latency | Tokens/sec |
|---|---:|---:|
| `naive_full_prefix` | 0.8832 s | 289.86 |
| `kv_cache` | 0.1815 s | 1410.73 |

Throughput improvement:

```text
1410.73 / 289.86 ≈ 4.87x
```

This shows that KV-cache decoding achieved approximately **4.87x higher throughput** than naive full-prefix decoding in the longest-context, largest-batch setting tested.

## Selected Benchmark Results

### Batch Size 1, Context Length 256, Generation Length 32

| Method | Mean Latency | Tokens/sec |
|---|---:|---:|
| `naive_full_prefix` | 0.1794 s | 178.35 |
| `kv_cache` | 0.1781 s | 179.70 |

At batch size 1, the difference is small because the workload is relatively light.

### Batch Size 4, Context Length 256, Generation Length 32

| Method | Mean Latency | Tokens/sec |
|---|---:|---:|
| `naive_full_prefix` | 0.4386 s | 291.85 |
| `kv_cache` | 0.1719 s | 744.76 |

At batch size 4, KV caching provides a large speedup because naive decoding repeatedly recomputes attention over a larger batch and longer context.

### Batch Size 8, Context Length 256, Generation Length 32

| Method | Mean Latency | Tokens/sec |
|---|---:|---:|
| `naive_full_prefix` | 0.8832 s | 289.86 |
| `kv_cache` | 0.1815 s | 1410.73 |

At batch size 8 and context length 256, the benefit of KV caching becomes very clear.

## Interpretation

The results confirm a core systems insight in autoregressive transformer inference:

> KV caching becomes increasingly important as context length and batch size grow.

For short contexts and small batches, naive full-prefix decoding and KV-cache decoding can have similar latency because the amount of repeated computation is small. However, for longer contexts and larger batches, naive decoding becomes much slower because it repeatedly recomputes attention over the full prefix.

KV-cache decoding avoids much of this redundant computation by storing previous key/value tensors. This leads to substantially higher throughput in longer-context batched inference settings.

The benchmark also shows that throughput does not scale linearly with batch size. Hardware utilization, memory bandwidth, attention computation, and framework overhead all affect the final tokens-per-second measurement.

## Conclusion

This project demonstrates a reproducible transformer inference benchmark for analyzing token-level latency and throughput tradeoffs. It compares naive full-prefix decoding against KV-cache decoding across batch sizes, context lengths, and generation lengths.

The experiment confirms that KV caching can greatly improve decoding efficiency for batched, long-context autoregressive generation. On the tested Apple Silicon MPS setup, KV-cache decoding achieved up to approximately **4.87x higher throughput** than naive full-prefix decoding.

Overall, this project provides a compact research-engineering pipeline for studying transformer inference bottlenecks, benchmarking decoding strategies, and analyzing how batching, context length, and KV-cache reuse affect model-serving performance.

## Resume Bullet Supported by This Project

```latex
\resumeItem{Benchmarked naive full-prefix decoding against KV-cache decoding across batch size, context length, and generation length, observing up to 4.87$\times$ higher throughput from KV caching on Apple Silicon MPS.}
```

## Future Work

Possible extensions include:

- Add CUDA GPU benchmarking on NVIDIA hardware.
- Add mixed precision benchmarking with float16 or bfloat16.
- Compare PyTorch implementation against vLLM-style paged attention.
- Benchmark real Hugging Face transformer models.
- Add memory usage profiling.
- Add throughput-vs-latency tradeoff curves.
- Add sequence-length bucketization experiments.
- Add batch scheduling and continuous batching simulations.

