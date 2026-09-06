"""Offline H-G.1 FakeProvider identifiability smoke."""
from __future__ import annotations
import json
from .provenance_cascade_hg1 import run_hg1_fake_smoke, write_amendment_receipt

def main() -> int:
    summary = run_hg1_fake_smoke()
    write_amendment_receipt(summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
