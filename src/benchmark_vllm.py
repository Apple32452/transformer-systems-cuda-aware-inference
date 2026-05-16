import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out", default="results/vllm_results.json")
    args = parser.parse_args()

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed. Install it with `pip install vllm` in a supported GPU/Linux environment."
        ) from exc

    prompts = [
        f"Explain why KV caching improves transformer decoding. Example {i}:"
        for i in range(args.prompts)
    ]

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    llm = LLM(model=args.model)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start

    total_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    result = {
        "model": args.model,
        "num_prompts": args.prompts,
        "max_tokens": args.max_tokens,
        "elapsed_s": elapsed,
        "total_generated_tokens": total_generated,
        "generated_tokens_per_second": total_generated / elapsed if elapsed > 0 else None,
        "sample_output": outputs[0].outputs[0].text if outputs else "",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
