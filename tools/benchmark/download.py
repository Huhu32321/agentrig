"""Download the benchmark models into <data>/raw (default ../bench_data next to the repo).

    python tools/benchmark/download.py [--data D:/bench_data]
"""
import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import CASES  # noqa: E402

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "..", "bench_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    a = ap.parse_args()
    raw = os.path.join(os.path.abspath(a.data), "raw")
    os.makedirs(raw, exist_ok=True)
    for c in CASES:
        dst = os.path.join(raw, c["id"] + ".glb")
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        r = requests.get(c["url"], timeout=120)
        r.raise_for_status()
        if r.content[:4] != b"glTF":
            raise RuntimeError(f"{c['id']}: not a binary glTF")
        with open(dst, "wb") as f:
            f.write(r.content)
        print(f"{c['id']:28s} {len(r.content) / 1e6:6.2f} MB  {c['license']}")
    print("models in", raw)


if __name__ == "__main__":
    main()
