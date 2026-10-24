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


def _event(ids, phase="prefill", layer=0):
    return {"layer": layer, "phase": phase, "rows": len(ids), "used": 1,
            "windows": [len(ids)], "ids": ids}


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

    def test_prefill_reset_hotness_is_reset_only(self):
        events = [_event([0] * 10), _event([1]), _event([2], "decode")]
        cumulative = REPLAY.replay(events, 2, "hotness")
        reset = REPLAY.replay(events, 2, "prefill_reset_hotness")
        reset_small_decay = REPLAY.replay(
            events, 2, "prefill_reset_hotness", hotness_decay_tokens=1)
        self.assertEqual(cumulative.resident_experts[0], {0, 2})
        self.assertEqual(reset.resident_experts[0], {1, 2})
        self.assertEqual(reset.resident_experts, reset_small_decay.resident_experts)

    def test_flashmoe_frequency_reset_changes_victim(self):
        events = [_event([0] * 10), _event([1]), _event([2], "decode")]
        full = REPLAY.replay(events, 2, "flashmoe_norm",
                             flashmoe_alpha=0.2, reset_frequency=False)
        reset = REPLAY.replay(events, 2, "flashmoe_norm",
                              flashmoe_alpha=0.2, reset_frequency=True)
        self.assertEqual(full.resident_experts[0], {0, 2})
        self.assertEqual(reset.resident_experts[0], {1, 2})

    def test_flashmoe_retention_trades_recency_and_frequency(self):
        recent = REPLAY._flashmoe_retention(9, 1, 10, 10, 0.2)
        frequent = REPLAY._flashmoe_retention(0, 10, 10, 10, 0.2)
        recent_high = REPLAY._flashmoe_retention(9, 1, 10, 10, 0.8)
        frequent_high = REPLAY._flashmoe_retention(0, 10, 10, 10, 0.8)
        self.assertLess(recent, frequent)
        self.assertGreater(recent_high, frequent_high)
        self.assertEqual(REPLAY._flashmoe_retention(None, 10, 10, 10, 0.2), 0.8)

    def test_belady_uses_furthest_next_use(self):
        events = [_event([2]), _event([3]), _event([2]),
                  _event([4]), _event([3])]
        lru = REPLAY.replay(events, 2, "lru")
        belady = REPLAY.replay(events, 2, "belady")
        self.assertEqual((lru.hits, lru.misses), (1, 4))
        self.assertEqual((belady.hits, belady.misses), (2, 3))

    def test_five_step_evicted_refetch_overall_and_decode(self):
        events = [_event([2]), _event([3]),
                  _event([4], "decode"), _event([2], "decode")]
        result = REPLAY.replay(events, 2, "lru")
        self.assertEqual((result.prefill_requests, result.decode_requests),
                         (2, 2))
        self.assertEqual((result.prefill_misses, result.decode_misses), (2, 2))
        self.assertEqual(result.eviction_count, 4)
        self.assertEqual(result.decode_eviction_count, 2)
        self.assertEqual(result.refetch_5, 1)
        self.assertEqual(result.refetch_5_pct, 25.0)
        self.assertEqual(result.decode_refetch_5, 1)
        self.assertEqual(result.decode_refetch_5_pct, 50.0)

    def test_five_step_refetch_boundary(self):
        within = REPLAY.ReplayStats(
            evictions=[REPLAY.EvictionRecord(0, 5, 0, "decode")])
        REPLAY._score_evicted_refetches(within, {
            0: [frozenset({0}), frozenset(), frozenset(), frozenset(),
                frozenset(), frozenset({5})],
        })
        outside = REPLAY.ReplayStats(
            evictions=[REPLAY.EvictionRecord(0, 5, 0, "decode")])
        REPLAY._score_evicted_refetches(outside, {
            0: [frozenset({0}), frozenset(), frozenset(), frozenset(),
                frozenset(), frozenset(), frozenset({5})],
        })
        self.assertEqual((within.refetch_5, within.decode_refetch_5), (1, 1))
        self.assertEqual((outside.refetch_5, outside.decode_refetch_5), (0, 0))

    def test_replay_rejects_unknown_policy_and_alpha(self):
        events = [_event([0])]
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            REPLAY.replay(events, 2, "bogus")
        with self.assertRaisesRegex(ValueError, "flashmoe_alpha"):
            REPLAY.replay(events, 2, "flashmoe_norm", flashmoe_alpha=1.1)


if __name__ == "__main__":
    unittest.main()
