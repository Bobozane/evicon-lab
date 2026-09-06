"""Read-only validator CLI for H-G.1.2 evaluator analysis inputs."""
from __future__ import annotations

from .provenance_cascade_hg12_analysis import main as _analysis_main


def main(argv=None) -> int:
    args = list(argv or [])
    if "--validate-only" not in args:
        args.append("--validate-only")
    return _analysis_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
