#!/usr/bin/env python3
"""Inspect and qualify a declared quantization variant; separate from build regression."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.variant import main

if __name__ == '__main__':
    raise SystemExit(main())
