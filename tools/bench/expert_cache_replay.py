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
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


TRACE_FORMAT = "gufo-expert-route-v1"
POLICIES = ("lru", "hotness", "decayed", "prefill_reset_hotness",
            "flashmoe_norm", "belady")
PACKINGS = ("trace", "greedy")


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
    step: int = -1
    last_access_step: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvictionRecord:
    layer: int
    expert: int
    step: int
    phase: str


@dataclass
class ReplayStats:
    requests: int = 0
    hits: int = 0
    misses: int = 0
    windows: int = 0
    rows: int = 0
    prefill_requests: int = 0
    prefill_hits: int = 0
    prefill_misses: int = 0
    prefill_windows: int = 0
    prefill_rows: int = 0
    decode_requests: int = 0
    decode_hits: int = 0
    decode_misses: int = 0
    decode_windows: int = 0
    decode_rows: int = 0
    evictions: list[EvictionRecord] = field(default_factory=list)
    refetch_5: int = 0
    decode_refetch_5: int = 0
    resident_experts: dict[int, set[int]] = field(default_factory=dict)

    @property
    def hit_pct(self) -> float:
        return 100.0 * self.hits / self.requests if self.requests else 0.0

    @property
    def prefill_hit_pct(self) -> float:
        return (100.0 * self.prefill_hits / self.prefill_requests
                if self.prefill_requests else 0.0)

    @property
    def decode_hit_pct(self) -> float:
        return (100.0 * self.decode_hits / self.decode_requests
                if self.decode_requests else 0.0)

    @property
    def eviction_count(self) -> int:
        return len(self.evictions)

    @property
    def decode_eviction_count(self) -> int:
        return sum(record.phase == "decode" for record in self.evictions)

    @property
    def refetch_5_pct(self) -> float:
        return (100.0 * self.refetch_5 / self.eviction_count
                if self.eviction_count else 0.0)

    @property
    def decode_refetch_5_pct(self) -> float:
        return (100.0 * self.decode_refetch_5 / self.decode_eviction_count
                if self.decode_eviction_count else 0.0)


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


def _resolve_windows(event: dict, slots: int, packing: str,
                     compute_group_rows: int) -> list[tuple[int, list[int], list[int]]]:
    ids = event["ids"]
    used = event["used"]
    rows = event["rows"]
    if packing == "trace" and event.get("passes"):
        return [(route_pass["rows"], route_pass["ids"], [])
                for route_pass in event["passes"]]
    windows = (event["windows"] if packing == "trace" else
               _greedy_windows(ids, rows, used, slots, compute_group_rows))
    resolve_windows = []
    first = 0
    for window_rows in windows:
        start = first * used
        end = (first + window_rows) * used
        resolve_windows.append((window_rows, ids[start:end], ids[end:]))
        first += window_rows
    return resolve_windows


def _request_steps(events: Iterable[dict], slots: int, packing: str,
                   compute_group_rows: int) -> dict[int, list[frozenset[int]]]:
    steps: dict[int, list[frozenset[int]]] = defaultdict(list)
    for event in events:
        layer = event["layer"]
        for _, routed, _ in _resolve_windows(event, slots, packing,
                                            compute_group_rows):
            steps[layer].append(frozenset(dict.fromkeys(routed)))
    return dict(steps)


def _next_use_positions(
        request_steps: dict[int, list[frozenset[int]]]
) -> dict[int, dict[int, list[int]]]:
    positions: dict[int, dict[int, list[int]]] = {}
    for layer, steps in request_steps.items():
        layer_positions: dict[int, list[int]] = defaultdict(list)
        for step, experts in enumerate(steps):
            for expert in experts:
                layer_positions[expert].append(step)
        positions[layer] = dict(layer_positions)
    return positions


def _flashmoe_retention(last_access_step: int | None, hotness: int,
                        max_hotness: int, step: int, alpha: float) -> float:
    recency_norm = (0.0 if last_access_step is None else
                    1.0 / (step - last_access_step + 1))
    frequency_norm = hotness / max_hotness if max_hotness else 0.0
    return alpha * recency_norm + (1.0 - alpha) * frequency_norm


def _score_evicted_refetches(
        stats: ReplayStats,
        request_steps: dict[int, list[frozenset[int]]]) -> None:
    for record in stats.evictions:
        steps = request_steps.get(record.layer, [])
        last_step = min(record.step + 5, len(steps) - 1)
        for future_step in range(record.step + 1, last_step + 1):
            if record.expert in steps[future_step]:
                stats.refetch_5 += 1
                if record.phase == "decode":
                    stats.decode_refetch_5 += 1
                break


def replay(events: Iterable[dict], slots: int, policy: str,
           packing: str = "trace", compute_group_rows: int = 8,
           hotness_decay_tokens: int = 1024,
           flashmoe_alpha: float = 0.5,
           reset_frequency: bool = False) -> ReplayStats:
    if slots <= 0:
        raise ValueError("slots must be positive")
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    if packing not in PACKINGS:
        raise ValueError(f"unknown packing: {packing}")
    if not 0.0 <= flashmoe_alpha <= 1.0:
        raise ValueError("flashmoe_alpha must be between 0 and 1")
    events = list(events)
    request_steps = _request_steps(events, slots, packing, compute_group_rows)
    next_use = _next_use_positions(request_steps) if policy == "belady" else {}
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
        resolve_windows = _resolve_windows(event, slots, packing,
                                           compute_group_rows)

        def apply_resolve(routed: list[int], future_ids: list[int],
                          routed_rows: int, count_stats: bool,
                          update_hotness: bool) -> None:
            route_hits = Counter(routed)
            unique = list(dict.fromkeys(routed))
            future_hits = Counter(future_ids)
            current = set(unique)
            step = None
            if count_stats:
                cache.step += 1
                step = cache.step
                stats.requests += len(unique)
                stats.windows += 1
                stats.rows += routed_rows
                if phase == "prefill":
                    stats.prefill_requests += len(unique)
                    stats.prefill_windows += 1
                    stats.prefill_rows += routed_rows
                else:
                    stats.decode_requests += len(unique)
                    stats.decode_windows += 1
                    stats.decode_rows += routed_rows

            for expert, (last_used, _) in list(cache.residents.items()):
                cache.residents[expert] = (last_used, future_hits[expert])

            missing = []
            for expert in unique:
                if expert in cache.residents:
                    cache.clock += 1
                    cache.residents[expert] = (cache.clock, future_hits[expert])
                    if step is not None:
                        cache.last_access_step[expert] = step
                    if update_hotness and policy != "lru":
                        cache.hotness[expert] = min(
                            0xFFFFFFFF, cache.hotness[expert] + route_hits[expert])
                    if count_stats:
                        stats.hits += 1
                        if phase == "prefill":
                            stats.prefill_hits += 1
                        else:
                            stats.decode_hits += 1
                else:
                    missing.append(expert)

            if count_stats:
                stats.misses += len(missing)
                if phase == "prefill":
                    stats.prefill_misses += len(missing)
                else:
                    stats.decode_misses += len(missing)
            if len(unique) > slots:
                raise ValueError("trace resolve window exceeds requested slot capacity")
            for expert in missing:
                if len(cache.residents) >= slots:
                    candidates = [item for item in cache.residents
                                  if item not in current and item not in cache.desk]
                    if not candidates:
                        raise ValueError("no unpinned victim in replay window")

                    max_hotness = max(cache.hotness.values(), default=0)
                    policy_step = step if step is not None else max(cache.step, 0)

                    def victim_key(candidate: int):
                        future = cache.residents[candidate][1]
                        last_used = cache.residents[candidate][0]
                        if policy == "lru":
                            heat = 0
                        elif policy == "flashmoe_norm":
                            heat = _flashmoe_retention(
                                cache.last_access_step.get(candidate),
                                cache.hotness[candidate], max_hotness,
                                policy_step, flashmoe_alpha)
                        elif policy == "belady":
                            uses = next_use.get(layer, {}).get(candidate, [])
                            index = bisect_right(uses, policy_step)
                            next_step = uses[index] if index < len(uses) else None
                            heat = (-next_step if next_step is not None
                                    else -1_000_000_000)
                        else:
                            heat = cache.hotness[candidate]
                        return (future, heat, last_used)

                    victim = min(candidates, key=victim_key)
                    if count_stats:
                        stats.evictions.append(
                            EvictionRecord(layer, victim, step, phase))
                    del cache.residents[victim]
                    cache.last_access_step.pop(victim, None)
                cache.clock += 1
                cache.residents[expert] = (cache.clock, future_hits[expert])
                if step is not None:
                    cache.last_access_step[expert] = step
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
        if cache.last_phase == "prefill" and phase == "decode":
            if policy in ("decayed", "prefill_reset_hotness"):
                cache.hotness.clear()
                cache.tokens_since_decay = 0
            elif policy == "flashmoe_norm" and reset_frequency:
                cache.hotness.clear()
        for pass_index, (window_rows, routed, future_ids) in enumerate(resolve_windows):
            apply_resolve(routed, future_ids, window_rows, True, True)
            if pass_index == 0 and resident_ids:
                cache.desk = set(desk_ids)
        cache.last_phase = phase
        cache.desk.clear()
    stats.resident_experts = {
        layer: set(cache.residents) for layer, cache in caches.items()
    }
    _score_evicted_refetches(stats, request_steps)
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


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError("invalid --flashmoe-alphas list") from error
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("--flashmoe-alphas values must be between 0 and 1")
    return alphas


def _print_result(packing: str, label: str, result: ReplayStats) -> None:
    print(f"{packing:6} {label:25} windows={result.windows:7d} "
          f"unique_requests={result.requests:9d} hits={result.hits:9d} "
          f"misses={result.misses:9d} hit_pct={result.hit_pct:7.3f} "
          f"prefill_misses={result.prefill_misses:7d} "
          f"decode_misses={result.decode_misses:7d} "
          f"decode_hits={result.decode_hits:7d} "
          f"decode_hit_pct={result.decode_hit_pct:7.3f} "
          f"evictions={result.eviction_count:7d} "
          f"refetch5={result.refetch_5:6d} "
          f"refetch5_pct={result.refetch_5_pct:7.3f} "
          f"decode_refetch5={result.decode_refetch_5:6d} "
          f"decode_refetch5_pct={result.decode_refetch_5_pct:7.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--slots-per-layer", type=int,
                        help="override the trace's cache capacity")
    parser.add_argument("--compute-group-rows", type=int, default=8)
    parser.add_argument("--hotness-decay-tokens", type=int, default=1024)
    parser.add_argument("--policy", action="append", choices=POLICIES,
                        dest="policies",
                        help="replay policy; repeat to select several "
                             "(default: lru,hotness,decayed)")
    parser.add_argument("--packing", action="append", choices=PACKINGS,
                        dest="packings",
                        help="resolve packing; repeat to select several "
                             "(default: trace,greedy)")
    parser.add_argument("--flashmoe-alphas", default="0.2,0.5,0.7,0.8",
                        help="comma-separated alpha sweep for flashmoe_norm")
    parser.add_argument("--room-slots", type=int,
                        help="also estimate an oracle fixed Desk/Room split")
    args = parser.parse_args()
    try:
        policies = tuple(dict.fromkeys(args.policies or
                                       ("lru", "hotness", "decayed")))
        packings = tuple(dict.fromkeys(args.packings or PACKINGS))
        alphas = _parse_alphas(args.flashmoe_alphas)
        num_experts, used, trace_slots, events = load_trace(args.trace)
        slots = args.slots_per_layer or trace_slots
        if slots is None:
            parser.error("--slots-per-layer is required for traces without capacity metadata")
        print(f"trace: {args.trace} events={len(events)} experts={num_experts} used={used}")
        for packing in packings:
            for policy in policies:
                if policy != "flashmoe_norm":
                    result = replay(events, slots, policy, packing,
                                    args.compute_group_rows,
                                    args.hotness_decay_tokens)
                    _print_result(packing, policy, result)
                    continue
                for reset_frequency in (False, True):
                    results = [
                        (alpha, replay(events, slots, policy, packing,
                                       args.compute_group_rows,
                                       args.hotness_decay_tokens, alpha,
                                       reset_frequency))
                        for alpha in alphas
                    ]
                    reset_label = "reset" if reset_frequency else "full"
                    for alpha, result in results:
                        _print_result(
                            packing,
                            f"flashmoe_norm_{reset_label}[{alpha:.2f}]",
                            result)
                    best_alpha, best = min(
                        results,
                        key=lambda item: (-item[1].decode_hit_pct,
                                          item[1].misses, item[0]))
                    _print_result(
                        packing,
                        f"flashmoe_norm_{reset_label}_best[{best_alpha:.2f}]",
                        best)
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
