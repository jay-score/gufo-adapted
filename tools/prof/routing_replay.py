#!/usr/bin/env python3
"""Solve per-layer expert-cache slot counts from a binary GURT trace.

The trace is produced by `gufo prompt --dump-routing PATH` while the cache is
enabled.  Records are replayed through a per-layer LRU cache; prefill records
warm and evict the cache but only decode-phase misses are scored.  The solver
uses a discrete water-filling allocation over per-layer slot counts and reports
a train/validation split plus the reverse split, so a profile cannot simply
memorize one half of the trace.

Usage:
  python3 tools/prof/routing_replay.py trace.gurt \
      --total 13920 --floor 128 --cap 512 --step 4 \
      --json /tmp/slots.json --cpp

The JSON profile is accepted by `--cache-slot-profile <path>`.  The generated
C++ initializer can be copied into `kSkewedSlotProfile` in engine.cpp.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

HEADER = struct.Struct("<4sHHIIIIQ")
RECORD = struct.Struct("<IIHHIIIIIIIIIII")
HEADER_BYTES = 32
RECORD_BYTES = 56


@dataclass
class Pass:
    first_row: int
    rows: int
    experts: tuple[int, ...]


@dataclass
class Record:
    seq: int
    forward: int
    kind: int
    layer: int
    phase: int
    rows: int
    used: int
    slots: int
    experts: tuple[int, ...]
    passes: tuple[Pass, ...]
    desk: tuple[int, ...]
    resident: tuple[int, ...]


def read_gurt(path: Path) -> tuple[dict, list[Record]]:
    data = path.read_bytes()
    if len(data) < HEADER_BYTES:
        raise ValueError("trace is shorter than the GURT header")
    magic, version, header_bytes, num_experts, experts_used, num_layers, flags, \
        record_count = HEADER.unpack_from(data, 0)
    if magic != b"GURT":
        raise ValueError(f"bad GURT magic: {magic!r}")
    if version != 1 or header_bytes < HEADER_BYTES:
        raise ValueError(f"unsupported GURT version/header: {version}/{header_bytes}")
    offset = header_bytes
    records: list[Record] = []
    # Older writers left record_count at zero; accept EOF-delimited traces.
    while offset + RECORD_BYTES <= len(data) and \
            (record_count == 0 or len(records) < record_count):
        (seq, forward, kind, layer, phase, rows, used, slots, nwin, nids,
         npass, ndesk, nres, payload, _reserved) = RECORD.unpack_from(data, offset)
        offset += RECORD_BYTES
        start = offset
        windows = struct.unpack_from(f"<{nwin}I", data, offset)
        offset += 4 * nwin
        ids = struct.unpack_from(f"<{nids}i", data, offset)
        offset += 4 * nids
        passes = []
        for _ in range(npass):
            first_row, pass_rows, count, _ = struct.unpack_from("<IIII", data, offset)
            offset += 16
            experts = struct.unpack_from(f"<{count}i", data, offset)
            offset += 4 * count
            passes.append(Pass(first_row, pass_rows, experts))
        desk = struct.unpack_from(f"<{ndesk}i", data, offset)
        offset += 4 * ndesk
        resident = struct.unpack_from(f"<{nres}i", data, offset)
        offset += 4 * nres
        if offset - start != payload:
            raise ValueError(
                f"record {seq}: payload size {offset - start} != declared {payload}")
        records.append(Record(seq, forward, kind, layer, phase, rows, used, slots,
                              tuple(ids), tuple(passes), tuple(desk),
                              tuple(resident)))
    if offset != len(data):
        raise ValueError(f"{len(data) - offset} trailing bytes after {len(records)} records")
    header = {
        "version": version,
        "num_experts": num_experts,
        "experts_used": experts_used,
        "num_layers": num_layers,
        "flags": flags,
        "record_count": len(records),
        "windows": windows if records else (),
    }
    return header, records


def read_jsonl(path: Path) -> tuple[dict, list[Record]]:
    """Read the legacy GUFO_EXPERT_ROUTE_TRACE JSONL stream.

    The legacy file has no forward sequence number.  Records are emitted in
    layer order within each forward, so a return to layer zero starts a new
    forward.
    """
    records: list[Record] = []
    header: dict = {"num_layers": 0, "num_experts": 0, "experts_used": 0}
    forward = 0
    last_layer = None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "format" in item:
                header.update(item)
                continue
            layer = int(item["layer"])
            if last_layer is not None and layer <= last_layer:
                forward += 1
            last_layer = layer
            passes = tuple(
                Pass(int(p["first"]), int(p["rows"]),
                     tuple(int(v) for v in p["ids"]))
                for p in item.get("passes", []))
            records.append(Record(
                len(records), forward, 0 if not passes else 1, layer,
                1 if item["phase"] == "prefill" else 0, int(item["rows"]),
                int(item["used"]), int(header.get("slots_per_layer", 0)),
                tuple(int(v) for v in item.get("ids", [])), passes,
                tuple(int(v) for v in item.get("desk", [])),
                tuple(int(v) for v in item.get("resident", []))))
    if records:
        header["num_layers"] = max(r.layer for r in records) + 1
        header["num_experts"] = max(
            max(r.experts, default=0) for r in records) + 1
    header["record_count"] = len(records)
    return header, records


def layer_events(records: list[Record], layer: int) -> list[tuple[int, int, tuple[int, ...]]]:
    events = []
    for rec in records:
        if rec.layer != layer:
            continue
        if rec.kind == 0:
            if rec.experts:
                events.append((rec.phase, rec.forward, rec.experts))
            continue
        if rec.passes:
            for item in rec.passes:
                if item.experts:
                    events.append((1, rec.forward, item.experts))
        elif rec.experts:
            events.append((1, rec.forward, rec.experts))
        extra = tuple(dict.fromkeys(rec.desk + rec.resident))
        if extra:
            events.append((1, rec.forward, extra))
    return events


def decode_forwards(records: list[Record], layer: int) -> list[int]:
    return sorted({rec.forward for rec in records
                   if rec.layer == layer and rec.phase == 0 and rec.kind == 0})


def simulate(events, slots: int) -> dict[int, int]:
    """Return decode cold misses keyed by forward sequence number."""
    cache: OrderedDict[int, None] = OrderedDict()
    misses: dict[int, int] = defaultdict(int)
    for phase, forward, experts in events:
        for expert in experts:
            if expert in cache:
                cache.move_to_end(expert)
                continue
            if phase == 0:
                misses[forward] += 1
            cache[expert] = None
            if len(cache) > slots:
                cache.popitem(last=False)
    return misses


def split_misses(misses: dict[int, int], train: set[int],
                 validation: set[int]) -> tuple[int, int, int]:
    return (sum(misses.get(f, 0) for f in train),
            sum(misses.get(f, 0) for f in validation),
            sum(misses.values()))


def allocate(cost: list[dict[int, tuple[int, int, int]]], slots: list[int],
             total: int, floor: int, cap: int, step: int, index: int) -> list[int]:
    """Exact multiple-choice knapsack. cost[l][s] = (train, validation, all)."""
    layers = len(cost)
    baseline = floor * layers
    if total < baseline or (total - baseline) % step:
        raise ValueError("total - floor*num_layers must be a non-negative multiple of step")
    units = (total - baseline) // step
    options = []
    for per_layer in cost:
        options.append([(s, (s - floor) // step) for s in sorted(per_layer)
                        if floor <= s <= cap and (s - floor) % step == 0])
    infinity = float("inf")
    dp = [infinity] * (units + 1)
    dp[0] = 0.0
    choices = []
    for layer in range(layers):
        nxt = [infinity] * (units + 1)
        choice = [0] * (units + 1)
        per_layer = cost[layer]
        for used, base in enumerate(dp):
            if base == infinity:
                continue
            for value, add in options[layer]:
                if used + add > units:
                    continue
                total_cost = base + per_layer[value][index]
                if total_cost < nxt[used + add]:
                    nxt[used + add] = total_cost
                    choice[used + add] = value
        dp = nxt
        choices.append(choice)
    if dp[units] == infinity:
        raise ValueError("no slot profile fits the total and cap")
    counts = [floor] * layers
    used = units
    for layer in range(layers - 1, -1, -1):
        value = choices[layer][used]
        counts[layer] = value
        used -= (value - floor) // step
    if used != 0:
        raise ValueError("slot profile reconstruction did not spend the total")
    return counts


def evaluate(cost, counts, index):
    return sum(cost[l][s][index] for l, s in enumerate(counts))


def solve(header, records, total, floor, cap, step, weight_hot):
    layers = header["num_layers"]
    if layers != 48:
        print(f"warning: expected 48 layers, trace has {layers}", file=sys.stderr)
    cost = []
    for layer in range(layers):
        events = layer_events(records, layer)
        forwards = decode_forwards(records, layer)
        cut = int(len(forwards) * 0.75)
        train, validation = set(forwards[:cut]), set(forwards[cut:])
        per_slot = {}
        for slots in range(floor, cap + 1, step):
            misses = simulate(events, slots)
            per_slot[slots] = split_misses(misses, train, validation)
        if cap not in per_slot:
            per_slot[cap] = split_misses(simulate(events, cap), train, validation)
        cost.append(per_slot)
        print(f"layer {layer:2d}: events={len(events):5d} decode={len(forwards):4d} "
              f"misses(128)={per_slot[floor][2]} misses(512)={per_slot[cap][2]}",
              file=sys.stderr)
    profiles = {}
    reports = {}
    # index 0 = train, 1 = validation, 2 = all decode forwards.  "hot" weights
    # cold misses on L0..L2, which cannot hide behind lookahead prefetch.
    for name, hot_weight in (("pure", 1.0), ("hot", weight_hot)):
        weighted_cost = []
        for layer, per_slot in enumerate(cost):
            w = hot_weight if layer < 3 else 1.0
            weighted_cost.append(
                {s: (m[0] * w, m[1] * w, m[2] * w)
                 for s, m in per_slot.items()})
        train_fit = allocate(weighted_cost, None, total, floor, cap, step, 0)
        val_fit = allocate(weighted_cost, None, total, floor, cap, step, 1)
        final = allocate(weighted_cost, None, total, floor, cap, step, 2)
        profiles[name] = final
        reports[name] = {
            "train_fit_val_pure": sum(cost[l][s][1]
                                      for l, s in enumerate(train_fit)),
            "val_fit_train_pure": sum(cost[l][s][0]
                                      for l, s in enumerate(val_fit)),
            "val_fit_val_pure": sum(cost[l][s][1]
                                    for l, s in enumerate(val_fit)),
            "train_pure": sum(cost[l][s][0] for l, s in enumerate(final)),
            "val_pure": sum(cost[l][s][1] for l, s in enumerate(final)),
            "all_pure": sum(cost[l][s][2] for l, s in enumerate(final)),
            "slots": final,
        }
    # Pick the weighting with lower pure validation misses when fit on train.
    # Ties prefer the L0..L2-weighted profile because those layers have no
    # lookahead runway.
    winner = min(profiles, key=lambda name: (
        reports[name]["train_fit_val_pure"], name != "hot"))
    return winner, profiles, reports, cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--jsonl", action="store_true",
                        help="read the legacy JSONL trace instead of GURT")
    parser.add_argument("--total", type=int, default=13920)
    parser.add_argument("--floor", type=int, default=128)
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--weight-hot", type=float, default=8.0,
                        help="miss weight for L0..L2 (default: 8)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--cpp", action="store_true")
    args = parser.parse_args()

    header, records = (read_jsonl(args.trace) if args.jsonl
                       else read_gurt(args.trace))
    print(f"trace: {args.trace} records={len(records)} experts={header['num_experts']} "
          f"used={header['experts_used']} layers={header['num_layers']}")
    if not records:
        raise SystemExit("trace has no records")
    winner, profiles, reports, _cost = solve(
        header, records, args.total, args.floor, args.cap, args.step,
        args.weight_hot)
    for name, report in reports.items():
        print(f"{name:4s}: train_fit_val={report['train_fit_val_pure']} "
              f"val_fit_train={report['val_fit_train_pure']} "
              f"all={report['all_pure']} slots={report['slots']}")
    print(f"winner: {winner}")
    profile = {
        "version": 1,
        "source": str(args.trace),
        "total": args.total,
        "floor": args.floor,
        "cap": args.cap,
        "step": args.step,
        "winner": winner,
        "slots": profiles[winner],
        "reports": {name: {k: v for k, v in report.items() if k != "slots"}
                    for name, report in reports.items()},
    }
    if args.json:
        args.json.write_text(json.dumps(profile, indent=2) + "\n")
        print(f"wrote {args.json}")
    if args.cpp:
        slots = profiles[winner]
        print("constexpr std::array<std::uint32_t, %d> kSkewedSlotProfile = {" % len(slots))
        for i in range(0, len(slots), 12):
            print("    " + ", ".join(str(v) for v in slots[i:i + 12]) + ",")
        print("};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
