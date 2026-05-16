import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

from tiny_decoder import TinyDecoderLM


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed_run(fn, warmup_steps: int, measure_steps: int, device: torch.device) -> Dict[str, float]:
    for _ in range(warmup_steps):
        fn()
    synchronize(device)

    latencies = []
    for _ in range(measure_steps):
        start = time.perf_counter()
        fn()
        synchronize(device)
        latencies.append(time.perf_counter() - start)

    arr = np.array(latencies, dtype=np.float64)
    return {
        "latency_mean_s": float(arr.mean()),
        "latency_p50_s": float(np.percentile(arr, 50)),
        "latency_p95_s": float(np.percentile(arr, 95)),
        "latency_std_s": float(arr.std(ddof=0)),
    }


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/torch_decode_results.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = select_device(cfg.get("device", "auto"))
    dtype = torch.float16 if cfg.get("dtype") == "float16" and device.type == "cuda" else torch.float32

    torch.manual_seed(int(cfg.get("seed", 7)))

    model = TinyDecoderLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_position_embeddings=cfg["max_position_embeddings"],
    ).to(device=device, dtype=dtype)
    model.eval()

    rows = []
    total_cases = (
        len(cfg["batch_sizes"])
        * len(cfg["context_lengths"])
        * len(cfg["generation_lengths"])
        * 2
    )

    with tqdm(total=total_cases, desc="benchmark") as pbar:
        for batch_size in cfg["batch_sizes"]:
            for context_len in cfg["context_lengths"]:
                for generation_len in cfg["generation_lengths"]:
                    input_ids = torch.randint(
                        low=0,
                        high=cfg["vocab_size"],
                        size=(batch_size, context_len),
                        device=device,
                    )

                    for method in ["naive_full_prefix", "kv_cache"]:
                        if method == "naive_full_prefix":
                            fn = lambda: model.generate_naive(input_ids, generation_len)
                        else:
                            fn = lambda: model.generate_with_cache(input_ids, generation_len)

                        metrics = timed_run(
                            fn,
                            warmup_steps=int(cfg.get("warmup_steps", 3)),
                            measure_steps=int(cfg.get("measure_steps", 8)),
                            device=device,
                        )
                        tokens = batch_size * generation_len
                        metrics["tokens_per_second"] = tokens / metrics["latency_mean_s"]
                        metrics.update(
                            {
                                "method": method,
                                "device": str(device),
                                "dtype": str(dtype).replace("torch.", ""),
                                "batch_size": batch_size,
                                "context_len": context_len,
                                "generation_len": generation_len,
                                "tokens": tokens,
                            }
                        )
                        rows.append(metrics)
                        pbar.update(1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "device",
        "dtype",
        "batch_size",
        "context_len",
        "generation_len",
        "tokens",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_std_s",
        "tokens_per_second",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
