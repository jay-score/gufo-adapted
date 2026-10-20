#!/usr/bin/env python3
"""Replay Gufo expert-route JSONL traces against cache policies.

The simulator models demand-resolved unique experts and future-use protection.
Prefetch completions, speculative hotness updates, token-prior hints, and SSD
timing are intentionally not inferred from route IDs; use this for
replacement/packing comparisons, not as a substitute for measured SsdStats or
end-to-end runs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TRACE_FORMAT = "gufo-expert-route-v1"


def load_trace(path: Path) -> tuple[int, int, int | None, list[dict]]:
    with path.open("r", encoding="utf-8") as stream:
        try:
            header = json.loads(next(stream))
        except (StopIteration, json.JSONDecodeError) as error:
            raise ValueError("missing or invalid route-trace header") from error
        if header.get("format") != TRACE_FORMAT:
            raise ValueError(f"unsupported route-trace format: {header.get('format')}")
        num_experts = _positive_int(header.get("num_experts"), "num_experts")
        experts_used = _positive_int(header.get("experts_used"), "experts_used")
        slots_per_layer = header.get("slots_per_layer")
        if slots_per_layer is not None:
            slots_per_layer = _positive_int(slots_per_layer, "slots_per_layer")
        if experts_used > num_experts:
            raise ValueError("experts_used exceeds num_experts")

        events = []
        for line_number, line in enumerate(stream, 2):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at line {line_number}") from error
            layer = _nonnegative_int(event.get("layer"), "layer")
            rows = _positive_int(event.get("rows"), "rows")
            used = _positive_int(event.get("used"), "used")
            if used != experts_used:
                raise ValueError(f"experts_used changed at line {line_number}")
            phase = event.get("phase")
            if phase not in ("prefill", "decode"):
                raise ValueError(f"invalid phase at line {line_number}")
            ids = event.get("ids")
            windows = event.get("windows")
            if not isinstance(ids, list) or len(ids) != rows * used:
                raise ValueError(f"invalid routed-ID count at line {line_number}")
            if any(not isinstance(value, int) or value < 0 or value >= num_experts
                   for value in ids):
                raise ValueError(f"out-of-range expert ID at line {line_number}")
            if (not isinstance(windows, list) or not windows
                    or any(not isinstance(value, int) or value <= 0 for value in windows)
                    or sum(windows) != rows):
                raise ValueError(f"invalid resolve windows at line {line_number}")
            passes = event.get("passes", [])
            if not isinstance(passes, list):
                raise ValueError(f"invalid pass list at line {line_number}")
            for route_pass in passes:
                if not isinstance(route_pass, dict):
                    raise ValueError(f"invalid pass record at line {line_number}")
                pass_first = _nonnegative_int(route_pass.get("first"), "pass.first")
                pass_rows = _positive_int(route_pass.get("rows"), "pass.rows")
                pass_ids = route_pass.get("ids")
                if (pass_first + pass_rows > rows or not isinstance(pass_ids, list)
                        or any(not isinstance(value, int) for value in pass_ids)):
                    raise ValueError(f"invalid pass dimensions at line {line_number}")
                if (not pass_ids or any(value < 0 or value >= num_experts
                                        for value in pass_ids)):
                    raise ValueError(f"invalid pass expert IDs at line {line_number}")
            desk = event.get("desk", [])
            if (not isinstance(desk, list)
                    or any(not isinstance(value, int) or value < 0
                           or value >= num_experts for value in desk)
                    or len(set(desk)) != len(desk)):
                raise ValueError(f"invalid desk expert IDs at line {line_number}")
            resident = event.get("resident", [])
            if (not isinstance(resident, list)
                    or any(not isinstance(value, int) or value < 0
                           or value >= num_experts for value in resident)
                    or len(set(resident)) != len(resident)):
                raise ValueError(f"invalid resident expert IDs at line {line_number}")
            events.append({
                "layer": layer,
                "phase": phase,
                "rows": rows,
                "used": used,
                "windows": windows,
                "ids": ids,
                "passes": passes,
                "desk": desk,
                "resident": resident,
            })
    return num_experts, experts_used, slots_per_layer, events


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass
class LayerCache:
    residents: dict[int, tuple[int, int]] = field(default_factory=dict)
    desk: set[int] = field(default_factory=set)
    hotness: Counter = field(default_factory=Counter)
    clock: int = 0
    tokens_since_decay: int = 0
    last_phase: str | None = None


@dataclass
class ReplayStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    windows: int = 0
    rows: int = 0
    resident_experts: dict[int, set[int]] = field(default_factory=dict)

    @property
    def hit_pct(self) -> float:
        return 100.0 * self.hits / self.requests if self.requests else 0.0


def _greedy_windows(ids: list[int], rows: int, used: int, capacity: int,
                    group_rows: int) -> list[int]:
    if capacity < used:
        raise ValueError("cache capacity must hold at least one token's routes")
    windows = []
    first = 0
    while first < rows:
        union: set[int] = set()
        count = 0
        while first + count < rows:
            add_rows = min(group_rows, rows - first - count)
            candidate = set(union)
            begin = (first + count) * used
            end = (first + count + add_rows) * used
            candidate.update(ids[begin:end])
            if len(candidate) > capacity:
                if count != 0:
                    break
                add_rows = 1
                candidate = set(ids[begin:begin + used])
                if len(candidate) > capacity:
                    raise ValueError("one row routes more experts than the cache holds")
            union = candidate
            count += add_rows
        windows.append(count)
        first += count
    return windows


def replay(events: Iterable[dict], slots: int, policy: str,
           packing: str = "trace", compute_group_rows: int = 8,
           hotness_decay_tokens: int = 1024) -> ReplayStats:
    if slots <= 0:
        raise ValueError("slots must be positive")
    if policy not in ("lru", "hotness", "decayed"):
        raise ValueError(f"unknown policy: {policy}")
    if packing not in ("trace", "greedy"):
        raise ValueError(f"unknown packing: {packing}")
    caches: dict[int, LayerCache] = {}
    stats = ReplayStats()
    for event in events:
        layer = event["layer"]
        cache = caches.get(layer)
        if cache is None:
            cache = LayerCache(
                residents={expert: (expert + 1, 0)
                           for expert in range(slots)},
                clock=slots,
            )
            caches[layer] = cache
        phase = event["phase"]
        ids = event["ids"]
        used = event["used"]
        rows = event["rows"]
        if packing == "trace" and event.get("passes"):
            resolve_windows = [(route_pass["rows"], route_pass["ids"], [])
                               for route_pass in event["passes"]]
        else:
            windows = (event["windows"] if packing == "trace" else
                       _greedy_windows(ids, rows, used, slots,
                                       compute_group_rows))
            resolve_windows = []
            first = 0
            for window_rows in windows:
                start = first * used
                end = (first + window_rows) * used
                resolve_windows.append((window_rows, ids[start:end], ids[end:]))
                first += window_rows
        def apply_resolve(routed: list[int], future_ids: list[int],
                          routed_rows: int, count_stats: bool,
                          update_hotness: bool) -> None:
            route_hits = Counter(routed)
            unique = list(dict.fromkeys(routed))
            future_hits = Counter(future_ids)
            current = set(unique)
            for expert, (last_used, _) in list(cache.residents.items()):
                cache.residents[expert] = (last_used, future_hits[expert])
            if count_stats:
                stats.requests += len(unique)
                stats.windows += 1
                stats.rows += routed_rows

            missing = []
            for expert in unique:
                if expert in cache.residents:
                    cache.clock += 1
                    cache.residents[expert] = (cache.clock, future_hits[expert])
                    if update_hotness and policy != "lru":
                        cache.hotness[expert] = min(
                            0xFFFFFFFF, cache.hotness[expert] + route_hits[expert])
                    if count_stats:
                        stats.hits += 1
                else:
                    missing.append(expert)

            if count_stats:
                stats.misses += len(missing)
            if len(unique) > slots:
                raise ValueError("trace resolve window exceeds requested slot capacity")
            for expert in missing:
                if len(cache.residents) >= slots:
                    candidates = [item for item in cache.residents
                                  if item not in current and item not in cache.desk]
                    if not candidates:
                        raise ValueError("no unpinned victim in replay window")

                    def victim_key(candidate: int) -> tuple[int, int, int]:
                        _, future = cache.residents[candidate]
                        heat = cache.hotness[candidate] if policy != "lru" else 0
                        last_used = cache.residents[candidate][0]
                        return (future, heat, last_used)

                    victim = min(candidates, key=victim_key)
                    del cache.residents[victim]
                cache.clock += 1
                cache.residents[expert] = (cache.clock, future_hits[expert])
                if update_hotness and policy != "lru":
                    cache.hotness[expert] = min(
                        0xFFFFFFFF, cache.hotness[expert] + route_hits[expert])

            if update_hotness and policy == "decayed":
                cache.tokens_since_decay += routed_rows
                while cache.tokens_since_decay >= hotness_decay_tokens:
                    for expert in cache.hotness:
                        cache.hotness[expert] >>= 1
                    cache.tokens_since_decay -= hotness_decay_tokens

        desk_ids = event.get("desk", []) if packing == "trace" else []
        resident_ids = event.get("resident", []) if packing == "trace" else []
        preload_ids = resident_ids or desk_ids
        if preload_ids:
            # Pass 0 first resolves resident routed experts without counting
            # that reservation as a token-demand window.
            apply_resolve(preload_ids, [], 0, False, False)
            cache.desk = set(preload_ids)
        if policy == "decayed" and cache.last_phase == "prefill" and phase == "decode":
            cache.hotness.clear()
            cache.tokens_since_decay = 0
        for pass_index, (window_rows, routed, future_ids) in enumerate(resolve_windows):
            apply_resolve(routed, future_ids, window_rows, True, True)
            if pass_index == 0 and resident_ids:
                cache.desk = set(desk_ids)
        cache.last_phase = phase
        cache.desk.clear()
    stats.resident_experts = {
        layer: set(cache.residents) for layer, cache in caches.items()
    }
    return stats


def desk_room_summary(events: list[dict], slots: int, room_slots: int,
                      compute_group_rows: int) -> dict[str, int | float]:
    if room_slots <= 0 or room_slots >= slots:
        raise ValueError("room-slots must be between 1 and slots-1")
    counts: dict[int, Counter] = {}
    for event in events:
        counts.setdefault(event["layer"], Counter()).update(event["ids"])
    desk_capacity = slots - room_slots
    desk = {
        layer: {expert for expert, _ in frequency.most_common(desk_capacity)}
        for layer, frequency in counts.items()
    }
    total_rows = 0
    total_windows = 0
    peak_room_union = 0
    desk_hits = 0
    routed = 0
    for event in events:
        layer_desk = desk[event["layer"]]
        rows, used, ids = event["rows"], event["used"], event["ids"]
        routed += len(ids)
        desk_hits += sum(expert in layer_desk for expert in ids)
        first = 0
        while first < rows:
            union: set[int] = set()
            count = 0
            while first + count < rows:
                add = min(compute_group_rows, rows - first - count)
                begin = (first + count) * used
                end = (first + count + add) * used
                candidate = union | {expert for expert in ids[begin:end]
                                     if expert not in layer_desk}
                if len(candidate) > room_slots and count:
                    break
                if len(candidate) > room_slots:
                    add = 1
                    candidate = {expert for expert in ids[begin:begin + used]
                                 if expert not in layer_desk}
                    if len(candidate) > room_slots:
                        raise ValueError("a token exceeds the configured room slots")
                union = candidate
                count += add
            first += count
            total_rows += count
            total_windows += 1
            peak_room_union = max(peak_room_union, len(union))
    return {
        "desk_slots": desk_capacity,
        "room_slots": room_slots,
        "routed_assignments": routed,
        "desk_assignments": desk_hits,
        "desk_coverage_pct": 100.0 * desk_hits / routed if routed else 0.0,
        "greedy_room_windows": total_windows,
        "mean_rows_per_room_window": total_rows / total_windows if total_windows else 0.0,
        "peak_room_experts": peak_room_union,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slots-per-layer", type=int,
                        help="override the trace's cache capacity")
    parser.add_argument("--compute-group-rows", type=int, default=8)
    parser.add_argument("--hotness-decay-tokens", type=int, default=1024)
    parser.add_argument("--room-slots", type=int,
                        help="also estimate an oracle fixed Desk/Room split")
    args = parser.parse_args()
    try:
        num_experts, used, trace_slots, events = load_trace(args.trace)
        slots = args.slots_per_layer or trace_slots
        if slots is None:
            parser.error("--slots-per-layer is required for traces without capacity metadata")
        print(f"trace: {args.trace} events={len(events)} experts={num_experts} used={used}")
        for packing in ("trace", "greedy"):
            for policy in ("lru", "hotness", "decayed"):
                result = replay(events, slots, policy, packing,
                                args.compute_group_rows, args.hotness_decay_tokens)
                print(f"{packing:6} {policy:7} windows={result.windows:7d} "
                      f"unique_requests={result.requests:9d} hits={result.hits:9d} "
                      f"misses={result.misses:9d} hit_pct={result.hit_pct:7.3f}")
        if args.room_slots is not None:
            result = desk_room_summary(events, slots,
                                       args.room_slots, args.compute_group_rows)
            print("desk_room_oracle " + " ".join(
                f"{key}={value:.3f}" if isinstance(value, float)
                else f"{key}={value}" for key, value in result.items()))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
