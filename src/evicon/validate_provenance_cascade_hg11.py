from __future__ import annotations
import argparse, json
from .provenance_cascade_hg11 import DEFAULT_CONFIG, hg11_preflight, run_hg11_fake_smoke, write_amendment_receipt

def main(argv=None):
    parser=argparse.ArgumentParser(description="Offline H-G.1.1 truncation amendment validator")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--fake-smoke", action="store_true")
    parser.add_argument("--write-amendment-receipt", action="store_true")
    args=parser.parse_args(argv)
    if args.fake_smoke:
        payload=run_hg11_fake_smoke(args.config)
        if args.write_amendment_receipt:
            payload={**payload,"amendment_receipt_path":str(write_amendment_receipt(payload,config_path=args.config))}
    else:
        payload=hg11_preflight(args.config)
    print(json.dumps(payload,sort_keys=True))
    return 0 if payload.get("status") in {"fake_smoke_passed","ready_for_real_pilot"} else 1

if __name__ == "__main__":
    raise SystemExit(main())
