#!/usr/bin/env python3
"""Run the per-model inference qualification workflow.

The implementation lives in tools/gufo/qualify/; this shim keeps the model
tooling entry point next to the other tools/models scripts.

Typical use:

    python tools/models/qualify.py plan --base origin/windows-port --head HEAD
    python tools/models/qualify.py record --model qwen3.8-flash-next --side baseline ...
    python tools/models/qualify.py compare --model qwen3.8-flash-next ...
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from gufo.qualify.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
