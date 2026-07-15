#!/usr/bin/env python3
"""Summarize detailed output from ``vllm bench serve``.

The vLLM 0.19 benchmark stores TTFT and SSE inter-token intervals.  When a
server reports usage output lengths, this script derives single-request decode
speed from the measured token count rather than aggregate throughput.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any


def as_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value if item is not None]


def as_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if item is not None]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} RESULT.json", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    ttfts = as_float_list(data.get("ttfts"))
    output_lens = as_int_list(data.get("output_lens"))
    itls = data.get("itls") if isinstance(data.get("itls"), list) else []
    e2els = as_float_list(data.get("e2els") or data.get("latencies"))

    # vLLM 0.19 detailed results retain per-request SSE ITLs.  If a latency
    # array is present, prefer it; otherwise reconstruct from TTFT + ITLs.
    latencies: list[float] = []
    for index, ttft in enumerate(ttfts):
        if index < len(e2els):
            latencies.append(e2els[index])
            continue
        intervals = itls[index] if index < len(itls) and isinstance(itls[index], list) else []
        latencies.append(ttft + sum(float(item) for item in intervals))

    decode_rates: list[float] = []
    tpot_ms: list[float] = []
    for index, latency in enumerate(latencies):
        if index >= len(output_lens) or index >= len(ttfts):
            continue
        tokens = output_lens[index]
        decode_seconds = latency - ttfts[index]
        if tokens > 1 and decode_seconds > 0:
            decode_rates.append((tokens - 1) / decode_seconds)
            tpot_ms.append(decode_seconds * 1000 / (tokens - 1))

    def show(name: str, values: list[float], unit: str = "") -> None:
        if not values:
            print(f"{name}: unavailable")
            return
        print(
            f"{name}: median={statistics.median(values):.3f}{unit} "
            f"p90={percentile(values, 0.90):.3f}{unit} "
            f"p99={percentile(values, 0.99):.3f}{unit} n={len(values)}"
        )

    print(f"result: {path}")
    print(f"model: {data.get('model', 'unknown')}")
    print(f"completed: {data.get('completed', 'unknown')} failed: {data.get('failed', 'unknown')}")
    show("TTFT", ttfts, " ms")
    show("decode tok/s", decode_rates, "")
    show("TPOT", tpot_ms, " ms")
    if data.get("output_throughput") is not None:
        print(f"aggregate output throughput: {float(data['output_throughput']):.3f} tok/s")
    if data.get("request_throughput") is not None:
        print(f"request throughput: {float(data['request_throughput']):.3f} req/s")
    if data.get("errors"):
        print(f"errors: {data['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
