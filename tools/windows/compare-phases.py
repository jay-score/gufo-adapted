#!/usr/bin/env python3
"""Compare first-prompt control arms per request and phase, without averaging.

Reads a control.json written by first-prompt-control.py. Base and head arms of
the same mode and repeat are paired; each request phase gets the repository's
paired verdict (a change counts only when every repeat moves by more than 5%
and 3 ms in the same direction). Phases are compared only when both arms did
the same work; correctness failures (reused tokens, wrong answer, unclean
shutdown) fail the comparison outright. When both arms use byte-identical
binaries and DLLs, the run is an unchanged-binary control and any flagged
phase is measurement noise, reported as inconclusive.

Exit status: 0 passed, 1 failed, 2 inconclusive.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "functional"))

from metrics import NOISE_MS, SLOWDOWN  # noqa: E402
from phases import verdict  # noqa: E402

PHASES = ("wall_ms", "queue_ms", "ttft_ms", "prefill_ms", "decode_ms",
          "cache_restore_ms", "cache_snapshot_ms", "max_inter_token_ms")
# Decode-side timings follow the generated tokens and accepted drafts.
WORK = ("prompt_tokens", "cached_tokens", "completion_tokens", "draft_tokens",
        "draft_tokens_accepted", "text_sha256")
PREFILL_WORK = ("prompt_tokens", "cached_tokens")
DECODE_PHASES = {"decode_ms", "max_inter_token_ms", "wall_ms", "cache_snapshot_ms"}


def identity(arm):
    binary = arm["binary"]
    return binary["sha256"], binary["dll_set_sha256"]


def compare(control):
    arms = control["arms"]
    labels = sorted({arm["label"] for arm in arms})
    if labels != ["base", "head"]:
        raise ValueError(f"expected base and head arms, found {labels}")
    unchanged = len({identity(arm) for arm in arms}) == 1
    correctness = [f"{arm['label']} {arm['mode']} #{arm['repeat']}: {problem}"
                   for arm in arms for problem in arm.get("problems", [])]
    rows = []
    for mode in sorted({arm["mode"] for arm in arms}):
        pairs = {}
        for arm in arms:
            if arm["mode"] == mode:
                pairs.setdefault(arm["repeat"], {})[arm["label"]] = arm
        pairs = [pair for _, pair in sorted(pairs.items()) if len(pair) == 2]
        if not pairs:
            raise ValueError(f"{mode}: no complete base/head pair")
        names = [request["name"] for request in pairs[0]["base"]["requests"]]
        for index, name in enumerate(names):
            observed = [(pair["base"]["requests"][index], pair["head"]["requests"][index])
                        for pair in pairs]
            for phase in PHASES:
                work = WORK if phase in DECODE_PHASES else PREFILL_WORK
                differs = sorted({field for base, head in observed for field in work
                                  if base.get(field) != head.get(field)})
                values = [(base["phases"].get(phase), head["phases"].get(phase))
                          for base, head in observed]
                complete = [(b, h) for b, h in values if b is not None and h is not None]
                result = verdict(complete) if not differs else "work differs"
                rows.append({"mode": mode, "request": name, "phase": phase,
                             "base": [b for b, _ in values], "head": [h for _, h in values],
                             "verdict": result, "work_differs": differs})
    slower = [row for row in rows if row["verdict"] == "slower"]
    unsettled = [row for row in rows if row["verdict"] in ("noisy", "work differs", "missing")]
    if correctness:
        status = "failed"
    elif unchanged:
        moved = [row for row in rows if row["verdict"] in ("slower", "faster", "noisy")]
        status = "inconclusive" if moved else "passed"
    elif slower:
        status = "failed"
    elif unsettled:
        status = "inconclusive"
    else:
        status = "passed"
    return {"schema": 1, "status": status, "unchanged_binary_control": unchanged,
            "slowdown": SLOWDOWN, "noise_ms": NOISE_MS, "correctness": correctness,
            "measurements": rows}


def table(result):
    lines = ["| Mode | Request | Phase | Base ms | Head ms | Verdict |",
             "| --- | --- | --- | ---: | ---: | --- |"]
    for row in result["measurements"]:
        cells = [", ".join("-" if value is None else f"{value:.2f}" for value in row[side])
                 for side in ("base", "head")]
        lines.append(f"| {row['mode']} | {row['request']} | {row['phase']} | "
                     f"{cells[0]} | {cells[1]} | {row['verdict']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("control", type=Path, help="control.json from first-prompt-control.py")
    parser.add_argument("--output", type=Path, help="New comparison JSON file")
    args = parser.parse_args()
    try:
        result = compare(json.loads(args.control.read_text(encoding="utf-8")))
    except (ValueError, KeyError, IndexError, OSError) as error:
        result = {"schema": 1, "status": "failed", "error": str(error), "measurements": []}
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(json.dumps(result, indent=2) + "\n")
    print(table(result))
    for problem in result.get("correctness", []):
        print(f"correctness: {problem}")
    if "error" in result:
        print(f"error: {result['error']}")
    control = " (unchanged-binary control)" if result.get("unchanged_binary_control") else ""
    print(f"{result['status']}{control}")
    return {"passed": 0, "failed": 1, "inconclusive": 2}[result["status"]]


if __name__ == "__main__":
    sys.exit(main())
