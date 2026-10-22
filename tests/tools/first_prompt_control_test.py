"""First-prompt control: request workload, verdicts and Windows server shutdown."""

import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest


TOOLS = Path(__file__).resolve().parents[2] / "tools" / "windows"


def load(name, module):
    spec = importlib.util.spec_from_file_location(module, TOOLS / name)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


control = load("first-prompt-control.py", "first_prompt_control")
compare = load("compare-phases.py", "compare_phases")


def request(name, prefill=6000.0, decode=50.0, completion=2, text="ACK"):
    return {"name": name, "prompt_tokens": 11036, "cached_tokens": 0,
            "completion_tokens": completion, "draft_tokens": 0, "draft_tokens_accepted": 0,
            "text_sha256": text,
            "phases": {"wall_ms": prefill + decode + 20, "queue_ms": 0.1, "ttft_ms": prefill + 25,
                       "prefill_ms": prefill, "decode_ms": decode, "cache_restore_ms": 0.0,
                       "cache_snapshot_ms": 40.0, "max_inter_token_ms": decode}}


def arm(label, repeat, binary="same", **changes):
    return {"label": label, "mode": "ar", "repeat": repeat, "problems": [],
            "binary": {"sha256": binary, "dll_set_sha256": "dlls"},
            "requests": [request("reuse", **changes), request("no-reuse", **changes)]}


def run(*arms):
    return compare.compare({"arms": list(arms)})


def verdicts(result, phase, name="reuse"):
    return [row["verdict"] for row in result["measurements"]
            if row["phase"] == phase and row["request"] == name]


class WorkloadTest(unittest.TestCase):
    def test_body_is_the_retained_workload(self):
        body = control.request_body(11000, True)
        user = body["messages"][1]["content"]
        self.assertEqual(len(user.split("\n")[0].split()), 11000)
        # Pinned so earlier evidence stays comparable (11,036 prompt tokens).
        self.assertEqual(hashlib.sha256(user.encode()).hexdigest(),
                         "f77b4e5d818c4d19674446b9720f46303ccbef3d37281dcaed9fdcfcc94712b2")
        self.assertTrue(body["cache_prompt"])
        self.assertFalse(control.request_body(11000, False)["cache_prompt"])
        self.assertEqual(body["messages"][1]["content"],
                         control.request_body(11000, False)["messages"][1]["content"])

    def test_observe_flags_reuse_and_wrong_answers(self):
        response = {"choices": [{"message": {"content": "NO"}}],
                    "usage": {"prompt_tokens": 11036, "cached_tokens": 512,
                              "completion_tokens": 2, "gufo": {"prefill_ms": 1.0}}}
        observed, problems = control.observe("reuse", True, 200, None, response, 9.0, "ACK")
        self.assertEqual(observed["phases"]["wall_ms"], 9.0)
        self.assertEqual(observed["phases"]["prefill_ms"], 1.0)
        self.assertEqual(problems, ["reuse: reused 512 tokens",
                                    "reuse: answered 'NO', expected 'ACK'"])
        _, problems = control.observe("reuse", True, 500, None, {"error": "x"}, 1.0, None)
        self.assertIn("reuse: HTTP 500", problems)


class CompareTest(unittest.TestCase):
    def test_identical_control_passes(self):
        result = run(arm("base", 0), arm("head", 0))
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["unchanged_binary_control"])

    def test_moved_phase_in_a_control_is_inconclusive(self):
        result = run(arm("base", 0), arm("head", 0, prefill=6600.0))
        self.assertEqual(verdicts(result, "prefill_ms"), ["slower"])
        self.assertEqual(result["status"], "inconclusive")

    def test_consistent_slowdown_fails_a_changed_binary(self):
        result = run(arm("base", 0), arm("head", 0, "new", prefill=6600.0),
                     arm("head", 1, "new", prefill=6700.0), arm("base", 1))
        self.assertFalse(result["unchanged_binary_control"])
        self.assertEqual(verdicts(result, "prefill_ms"), ["slower"])
        self.assertEqual(result["status"], "failed")

    def test_small_or_mixed_moves_do_not_fail(self):
        # 2 ms is under the 3 ms floor; mixed directions are noise.
        result = run(arm("base", 0), arm("head", 0, "new", decode=52.0),
                     arm("head", 1, "new", prefill=6600.0), arm("base", 1))
        self.assertEqual(verdicts(result, "decode_ms"), ["same"])
        self.assertEqual(verdicts(result, "prefill_ms"), ["noisy"])
        self.assertEqual(result["status"], "inconclusive")

    def test_different_work_is_not_compared(self):
        result = run(arm("base", 0), arm("head", 0, "new", decode=90.0, completion=3))
        self.assertEqual(verdicts(result, "decode_ms"), ["work differs"])
        # Prefill did the same work and is still compared.
        self.assertEqual(verdicts(result, "prefill_ms"), ["same"])
        self.assertEqual(result["status"], "inconclusive")

    def test_correctness_problems_fail(self):
        head = arm("head", 0, "new")
        head["problems"] = ["reuse: reused 512 tokens"]
        result = run(arm("base", 0), head)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["correctness"], ["head ar #0: reuse: reused 512 tokens"])

    def test_incomplete_pairs_are_rejected(self):
        with self.assertRaises(ValueError):
            run(arm("base", 0))
        base = arm("base", 0)
        lone = copy.deepcopy(base)
        lone["mode"] = "mtp"
        with self.assertRaises(ValueError):
            run(base, arm("head", 0), lone)


CHILD = """
import signal, sys, time
signal.signal(signal.SIGBREAK, lambda *_: sys.exit(0))
print("ready", flush=True)
for _ in range(600):
    time.sleep(0.1)
sys.exit(3)
"""


@unittest.skipUnless(os.name == "nt", "console control events are Windows-only")
class StopTest(unittest.TestCase):
    def test_ctrl_break_reaches_only_the_private_console(self):
        process = control.launch([sys.executable, "-c", CHILD], subprocess.PIPE)
        self.assertEqual(process.stdout.readline().strip(), b"ready")
        code, graceful = control.stop(process, timeout=20)
        process.stdout.close()
        self.assertEqual((code, graceful), (0, True))


if __name__ == "__main__":
    unittest.main()
