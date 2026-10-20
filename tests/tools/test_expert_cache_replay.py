import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT = (Path(__file__).resolve().parents[2] / "tools" / "bench" /
          "expert_cache_replay.py")
SPEC = importlib.util.spec_from_file_location("expert_cache_replay", SCRIPT)
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)


class ExpertCacheReplayTest(unittest.TestCase):
    def test_load_and_replay_trace_and_greedy_windows(self):
        content = (
            '{"format":"gufo-expert-route-v1","num_experts":4,'
            '"experts_used":2}\n'
            '{"layer":0,"phase":"prefill","rows":4,"used":2,'
            '"windows":[2,2],"ids":[0,1,0,1,2,3,2,3]}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "routes.jsonl"
            trace.write_text(content, encoding="utf-8")
            experts, used, slots, events = REPLAY.load_trace(trace)
        self.assertEqual((experts, used), (4, 2))
        self.assertIsNone(slots)
        actual = REPLAY.replay(events, 2, "lru", "trace")
        greedy = REPLAY.replay(events, 2, "lru", "greedy", compute_group_rows=1)
        self.assertEqual((actual.requests, actual.hits, actual.misses), (4, 2, 2))
        self.assertEqual((greedy.requests, greedy.hits, greedy.misses), (4, 2, 2))
        self.assertEqual(greedy.windows, 2)

    def test_decayed_policy_resets_prefill_hotness_at_decode(self):
        events = [
            {"layer": 0, "phase": "prefill", "rows": 6, "used": 2,
             "windows": [1, 1, 1, 1, 1, 1],
             "ids": [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2]},
            {"layer": 0, "phase": "decode", "rows": 1, "used": 2,
             "windows": [1], "ids": [3, 3]},
        ]
        decayed = REPLAY.replay(events, 3, "decayed", hotness_decay_tokens=1024)
        cumulative = REPLAY.replay(events, 3, "hotness", hotness_decay_tokens=1024)
        self.assertNotIn(0, decayed.resident_experts[0])
        self.assertIn(0, cumulative.resident_experts[0])

    def test_trace_rejects_bad_dimensions(self):
        content = (
            '{"format":"gufo-expert-route-v1","num_experts":4,'
            '"experts_used":2}\n'
            '{"layer":0,"phase":"decode","rows":1,"used":2,'
            '"windows":[1],"ids":[0]}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "bad.jsonl"
            trace.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "routed-ID count"):
                REPLAY.load_trace(trace)

    def test_trace_replays_exact_resolve_passes(self):
        content = (
            '{"format":"gufo-expert-route-v1","num_experts":4,'
            '"experts_used":2}\n'
            '{"layer":0,"phase":"prefill","rows":2,"used":2,'
            '"windows":[2],"ids":[0,1,2,3],"passes":['
            '{"first":0,"rows":2,"ids":[0,1]},'
            '{"first":0,"rows":2,"ids":[2,3]}]}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "passes.jsonl"
            trace.write_text(content, encoding="utf-8")
            _, _, _, events = REPLAY.load_trace(trace)
        result = REPLAY.replay(events, 3, "lru", "trace")
        self.assertEqual((result.requests, result.hits, result.misses), (4, 3, 1))

    def test_resident_pass_keeps_desk_and_allows_room_eviction(self):
        content = (
            '{"format":"gufo-expert-route-v1","num_experts":4,'
            '"experts_used":2}\n'
            '{"layer":0,"phase":"prefill","rows":2,"used":2,'
            '"windows":[2],"ids":[0,0,1,2],"desk":[0],'
            '"resident":[0,1],"passes":['
            '{"first":0,"rows":2,"ids":[0,0,1]},'
            '{"first":0,"rows":2,"ids":[2]}]}\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "resident-passes.jsonl"
            trace.write_text(content, encoding="utf-8")
            _, _, _, events = REPLAY.load_trace(trace)
        result = REPLAY.replay(events, 2, "lru", "trace")
        self.assertEqual((result.requests, result.hits, result.misses), (3, 2, 1))
        self.assertEqual(result.resident_experts[0], {0, 2})

    def test_desk_room_reports_coverage_and_room_pressure(self):
        events = [{"layer": 0, "phase": "prefill", "rows": 4, "used": 2,
                   "windows": [4], "ids": [0, 1, 0, 1, 0, 2, 0, 3]}]
        result = REPLAY.desk_room_summary(events, 3, 2, 1)
        self.assertEqual(result["desk_slots"], 1)
        self.assertEqual(result["room_slots"], 2)
        self.assertGreater(result["desk_coverage_pct"], 0)
        self.assertLessEqual(result["peak_room_experts"], 2)


if __name__ == "__main__":
    unittest.main()
