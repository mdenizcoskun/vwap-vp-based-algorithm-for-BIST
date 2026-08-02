#!/usr/bin/env python3
"""Backward-compatible entry point for the modular package."""

from vwap_volume_profile import *  # noqa: F403
from vwap_volume_profile import __all__
from vwap_volume_profile.cli import main


if __name__ == "__main__":
    main()
