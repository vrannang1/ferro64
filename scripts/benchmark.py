#!/usr/bin/env python3
"""Repeatable raw-text latency benchmark; Python standard library only.

Run each engine separately with identical weights and quantization. llama.cpp
uses an already running server, so its results are explicitly warm-only.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, endpoint, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base.rstrip("/") + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(data, dict):
        raise RuntimeError("Backend returned a non-object JSON response")
    if data.get("error"):
        raise RuntimeError(str(data["error"]))
    return data


def unload_ollama(args):
    request(args.url, "/api/generate", {"model": args.model, "keep_alive": 0})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        loaded = request(args.url, "/api/ps", timeout=10).get("models", [])
        names = {model.get("name") for model in loaded}
        if args.model not in names:
            return
        time.sleep(0.1)
    raise RuntimeError("Ollama did not unload the selected model within 15 seconds")


def complete(args, prompt, output_limit):
    if args.backend == "ollama":
        endpoint = "/api/generate"
        payload = {
            "model": args.model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "keep_alive": 0 if args.mode == "cold" else "5m",
            "options": {
                "temperature": 0,
                "seed": 42,
                "repeat_penalty": 1,
                "num_ctx": args.context,
                "num_predict": output_limit,
            },
        }
    else:
        endpoint = "/completion"
        payload = {
            "prompt": prompt,
            "n_predict": output_limit,
            "temperature": 0,
            "seed": 42,
            "repeat_penalty": 1,
            "cache_prompt": False,
            "stream": False,
        }
    started = time.perf_counter()
    data = request(args.url, endpoint, payload)
    wall = time.perf_counter() - started
    if args.backend == "ollama":
        duration = data.get("eval_duration", 0)
        return {
            "wall_seconds": wall,
            "load_seconds": data.get("load_duration", 0) / 1e9,
            "prompt_tokens": data.get("prompt_eval_count"),
            "cached_prompt_tokens": data.get("prompt_eval_cached_count"),
            "output_tokens": data.get("eval_count"),
            "decode_tokens_per_second": data.get("eval_count", 0) * 1e9 / duration if duration else None,
            "output_text": data.get("response", ""),
        }
    timings = data.get("timings", {})
    if data.get("truncated"):
        raise RuntimeError("llama.cpp truncated the prompt; increase context or shorten it")
    return {
        "wall_seconds": wall,
        "load_seconds": None,
        "prompt_tokens": timings.get("prompt_n", data.get("tokens_evaluated")),
        "cached_prompt_tokens": data.get("tokens_cached"),
        "output_tokens": timings.get("predicted_n", data.get("tokens_predicted")),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "output_text": data.get("content", ""),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("ollama", "llamacpp"), required=True)
    parser.add_argument("--model", required=True, help="Ollama tag, or matching model label for the llama.cpp report")
    parser.add_argument("--url", help="Defaults to Ollama :11434 or llama.cpp :8081")
    parser.add_argument("--mode", choices=("warm", "cold"), default="warm")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path; includes generated text")
    parser.add_argument("--dry-run", action="store_true", help="Validate arguments and print configuration without network requests")
    args = parser.parse_args()
    if min(args.context, args.max_tokens, args.runs) < 1:
        parser.error("context, max-tokens, and runs must be positive")
    if args.max_tokens >= args.context:
        parser.error("max-tokens must be smaller than context")
    if args.backend == "llamacpp" and args.mode == "cold":
        parser.error("llamacpp cold timing requires restarting its process; this utility only measures a running server in warm mode")
    args.url = args.url or ("http://127.0.0.1:11434" if args.backend == "ollama" else "http://127.0.0.1:8081")
    prompt = args.prompt_file.read_text() if args.prompt_file else "Explain Rust ownership with one short code example."
    if not prompt.strip():
        parser.error("prompt cannot be empty")
    report = {
        "backend": args.backend,
        "url": args.url,
        "model": args.model,
        "mode": args.mode,
        "context": args.context,
        "max_tokens": args.max_tokens,
        "runs_requested": args.runs,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_strategy": "Each run prepends a deterministic, changing sample number to reduce prefix-cache reuse; use the same prompt file for both engines.",
        "limitations": "Match weights, quantization and GPU settings manually. llama.cpp warm results exclude process/model startup. Raw prompts measure engine behavior rather than chat templates. This utility is a developer tool, separate from the Rust application.",
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if args.backend == "llamacpp":
        props = request(args.url, "/props", timeout=10)
        actual_context = props.get("default_generation_settings", {}).get("n_ctx")
        if actual_context != args.context:
            raise RuntimeError(f"llama-server reports context {actual_context}; restart it with -c {args.context} -np 1")
        report["runtime"] = {"model_path": props.get("model_path"), "context": actual_context}
    else:
        report["runtime"] = request(args.url, "/api/version", timeout=10)
    samples = []
    try:
        if args.mode == "warm":
            print("Running one excluded warm-up request...", file=sys.stderr)
            complete(args, "Benchmark sample 0.\n" + prompt, 8)
        for number in range(1, args.runs + 1):
            if args.backend == "ollama" and args.mode == "cold":
                unload_ollama(args)
            sample = complete(args, f"Benchmark sample {number}.\n" + prompt, args.max_tokens)
            if not sample["output_text"]:
                raise RuntimeError("Backend returned no text; this raw benchmark requires a text-generation model")
            samples.append(sample)
            print(f"Run {number}/{args.runs}: {sample['wall_seconds']:.3f}s", file=sys.stderr)
    finally:
        if args.backend == "ollama":
            unload_ollama(args)
    report["samples"] = samples
    report["median_wall_seconds"] = statistics.median(sample["wall_seconds"] for sample in samples)
    rates = [sample["decode_tokens_per_second"] for sample in samples if sample["decode_tokens_per_second"] is not None]
    report["median_decode_tokens_per_second"] = statistics.median(rates) if rates else None
    result = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
