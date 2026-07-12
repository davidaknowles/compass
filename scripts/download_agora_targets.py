#!/usr/bin/env python
"""Download the public Agora nominated AD target export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


AGORA_NOMINATED_TARGETS_URL = "https://agora.adknowledgeportal.org/api/v1/genes/nominated"
DEFAULT_OUT = Path.home() / "knowles_lab" / "data" / "compass" / "raw" / "agora" / "nominated_targets.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Agora nominated AD targets.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--url", default=AGORA_NOMINATED_TARGETS_URL)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    response = requests.get(args.url, timeout=args.timeout)
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Agora response did not contain nominated target items")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"downloaded {len(items)} Agora nominated targets to {out}")


if __name__ == "__main__":
    main()
