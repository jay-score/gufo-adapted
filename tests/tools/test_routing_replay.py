#!/usr/bin/env python3
"""Synthetic round-trip test for the GURT reader and slot solver."""

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "routing_replay", ROOT / "tools/prof/routing_replay.py")
solver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = solver
SPEC.loader.exec_module(solver)


def write_trace(path: Path, layers: int = 48, experts: int = 512,
                forwards: int = 32) -> None:
    records = []
    for layer in range(layers):
        for fwd in range(forwards):
            phase = 0 if fwd >= 4 else 1
            ids = tuple((layer * 3 + fwd + i) % experts for i in range(8))
            payload = struct.pack(f"<8i", *ids)
            header = struct.pack(
                "<IIHHIIIIIIIIIII", len(records), fwd, 0, layer, phase, 8, 8,
                290, 0, 8, 0, 0, 0, len(payload), 0)
            records.append(header + payload)
    blob = struct.pack("<4sHHIIIIQ", b"GURT", 1, 32, experts, 8, layers,
                       3, len(records)) + b"".join(records)
    path.write_bytes(blob)


class RoutingReplayTest(unittest.TestCase):
    def test_gurt_round_trip_and_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.gurt"
            write_trace(path)
            header, records = solver.read_gurt(path)
            self.assertEqual(header["num_layers"], 48)
            self.assertEqual(header["record_count"], 48 * 32)
            self.assertEqual(len(records), 48 * 32)
            self.assertEqual(records[0].experts, (0, 1, 2, 3, 4, 5, 6, 7))
            events = solver.layer_events(records, 0)
            self.assertEqual(len(events), 32)
            misses = solver.simulate(events, 128)
            self.assertEqual(len(misses), 28)
            self.assertGreaterEqual(sum(misses.values()), 8)
            self.assertLessEqual(sum(misses.values()), 8 * 28)

    def test_allocate_spends_exact_total(self):
        cost = [{s: (1000 - 4 * s, 900 - 3 * s, 1900 - 7 * s)
                 for s in range(128, 513, 4)} for _ in range(48)]
        slots = solver.allocate(cost, None, 13920, 128, 512, 4, 0)
        self.assertEqual(sum(slots), 13920)
        self.assertTrue(all(128 <= s <= 512 for s in slots))


if __name__ == "__main__":
    unittest.main()
