"""Read-only H-F analysis smoke; it never writes reports."""
from __future__ import annotations

import json

from .provenance_cascade_pilot_analysis import run_hf_analysis


def main() -> None:
    result = run_hf_analysis(
        "configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_analysis.v1.toml",
        write_outputs=False,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
