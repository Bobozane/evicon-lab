"""Offline H-E 48-run FakeProvider smoke."""
from __future__ import annotations

import argparse
import json

from .cascade_agent_protocol_v2_pilot import HD2_CONFIG_RELATIVE, run_fake_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="H-E strict protocol FakeProvider smoke")
    parser.add_argument("--config", default=HD2_CONFIG_RELATIVE)
    args = parser.parse_args()
    print(json.dumps(run_fake_smoke(args.config), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
