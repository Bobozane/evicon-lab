"""Offline final preflight for the isolated H-E H-D.2 pilot."""
from __future__ import annotations

import argparse
import json

from .cascade_agent_protocol_v2_pilot import final_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="H-E H-D.2 real pilot preflight (offline by default)")
    parser.add_argument("--config", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2.v1.toml")
    parser.add_argument("--approval", default="configs/provenance_cascade/pilot/provenance_cascade_pilot_hd2_approval_template.toml")
    args = parser.parse_args()
    print(json.dumps(final_preflight(args.config, args.approval), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
