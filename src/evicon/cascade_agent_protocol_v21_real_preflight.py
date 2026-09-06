"""Offline final preflight for the isolated H-D.2.1 pilot."""
from __future__ import annotations

import argparse
import json

from .cascade_agent_protocol_v21_pilot import final_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="H-D.2.1 real pilot preflight (offline by default)")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21.v1.toml")
    parser.add_argument("--approval", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd21_approval_template.toml")
    args = parser.parse_args()
    print(json.dumps(final_preflight(args.config, args.approval), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
