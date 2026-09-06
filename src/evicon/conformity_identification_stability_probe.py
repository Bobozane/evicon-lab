"""CLI wrapper for the bounded protocol-stability qualification gate."""
from .conformity_identification_network_gates import stability_main

if __name__ == "__main__":
    raise SystemExit(stability_main())
