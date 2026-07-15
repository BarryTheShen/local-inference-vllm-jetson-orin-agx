#!/usr/bin/env python3
"""Measure streaming TTFT and decode speed against a running vLLM API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first = None
    chunks = 0
    text_chars = 0
    usage = None
    with urllib.request.urlopen(request, timeout=300) as response:
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if first is None and event.get("choices"):
                first = time.perf_counter()
            for choice in event.get("choices", []):
                piece = choice.get("text") or ""
                if piece:
                    chunks += 1
                    text_chars += len(piece)
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    output_tokens = (usage or {}).get("completion_tokens") or chunks
    ttft_ms = ((first or finished) - started) * 1000
    total_ms = (finished - started) * 1000
    decode_seconds = max(total_ms / 1000 - ttft_ms / 1000, 1e-9)
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "output_tokens": output_tokens,
        "chunks": chunks,
        "decode_tok_s": output_tokens / decode_seconds,
        "text_chars": text_chars,
        "usage": usage,
    }


def run_measurements(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    requests: int,
    concurrency: int,
) -> dict[str, Any]:
    if concurrency <= 1:
        return {
            "measurements": [
                stream_once(url, model, prompt, max_tokens)
                for _ in range(requests)
            ]
        }
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(stream_once, url, model, prompt, max_tokens)
            for _ in range(requests)
        ]
        measurements = [future.result() for future in futures]
    return {
        "concurrency": concurrency,
        "wall_s": time.perf_counter() - started,
        "measurements": measurements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b-int8")
    parser.add_argument(
        "--prompt",
        default=(
            "Explain in one paragraph why deterministic benchmarking matters "
            "for local inference performance."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    warmup = None
    for _ in range(args.warmups):
        warmup = stream_once(args.url, args.model, args.prompt, args.max_tokens)
    result = {
        "warmup": warmup,
        **run_measurements(
            args.url,
            args.model,
            args.prompt,
            args.max_tokens,
            args.requests,
            args.concurrency,
        ),
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(f"Saved {args.output}")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
