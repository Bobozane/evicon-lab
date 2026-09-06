"""Offline 48-run FakeProvider smoke for the H-G identifiability amendment."""
from __future__ import annotations
import json
from .provenance_cascade_identifiability import run_hg_fake_smoke

def main() -> None:
    print(json.dumps(run_hg_fake_smoke(), ensure_ascii=True, sort_keys=True))

if __name__ == "__main__":
    main()
