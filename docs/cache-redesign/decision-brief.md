# Decision brief

A one-page summary of the evidence for choosing the continuation cache's
direction. Details: [experiments](experiments.md), [cost model](cost-model.md),
[hybrid design](hybrid-design.md).

## The question

Keep today's design (a full snapshot per checkpoint, in RAM and on disk) and
fix its defects, or move to the hybrid design: shared KV chunks, small
checkpoints holding only fixed state, and one index over RAM and disk?

## What the evidence says

Measured on Flash-Next and 27B with production-like settings, on real agent
sessions to 149k tokens, subagents, multi-user chat and restarts:

| Finding | Evidence |
| --- | --- |
| Checkpoint size is fixed state + per-token KV, exactly | Flash-Next with MTP: 113.8 MiB + 27.46 KB/token. 27B: 152 MiB (+80 MiB with DFlash2) + 64 KiB/token |
| Automatic budgets are small at long context | Two sessions: RAM 9.2 GB (Flash-Next) or 23 GB (27B) holds 2–3 checkpoints of a 100k conversation |
| Long checkpoints never reach disk | Automatic staging (2.3 / 5.8 GB) is smaller than a full checkpoint past ~75k tokens: 71 and 85 skipped writes on the agent runs |
| A RAM hit hides a longer disk hit | Disk is consulted only on a RAM miss; one W2 turn re-prefilled 22.6k tokens (24 s Flash-Next, ~60 s 27B) |
| A full RAM cache refuses checkpoints | 19–42 refusals per run, because a new checkpoint may evict only lower-ranked entries |
| 27B captures copy everything | 14 ms at 1.8k tokens up to 235 ms at 149k (10 GB). Flash-Next already captures only fixed state: 2–5 ms at any depth |
| In-session reuse is near ideal otherwise | Actual vs ideal reuse within 0.3–1.2 points except W2 and the restarts |
| The simulator is trustworthy | It reproduces actual reuse within 0.5% and prefill time within 0–8% on every run |

## Options compared (E7 simulation and cost model)

| | Today | Phase 0 (fixes, same format) | Hybrid (Phases 1–2) |
| --- | --- | --- | --- |
| W2 miss after forks | 22.6k tokens re-prefilled | Fixed | Fixed |
| Restart at 105k / 149k, Flash-Next | +24 s / +48 s prefill | Fixed | Fixed |
| Restart at 105k / 149k, 27B | +2.0 / +4.3 min prefill | Fixed | Fixed |
| Disk written, agent runs | 7–17 GB (deep writes skipped) | 24–57 GB | 6–13 GB |
| Disk time per checkpoint at 100k | 6.5 s Flash-Next, 15 s 27B (if staged) | Same, streamed | 0.4 s / 0.9 s |
| 27B capture per checkpoint at 100k / 149k | 181 / 236 ms | Same | ~14 ms |
| Checkpoints of one 100k conversation in RAM | 3 | 3 (27B), more for Flash-Next | 43–52 |
| Dense checkpoints (message boundaries, shared system prompt) | Unaffordable | Unaffordable | Affordable; W4 −5 s Flash-Next, −18 s 27B |
| Effort | — | Small: lookup rule, streamed writes, accounting | Large: model interface split for each model, chunk pool, new disk format |
| Risk | Known defects | Low | Medium: new store, format, eviction rules |

At concurrency 2 the hybrid adds efficiency, not reuse that Phase 0 cannot
reach. E6 tests whether concurrency 4 changes that.

## Recommendation

1. **Do Phase 0 now, independently of the redesign.** It fixes every lost
   reuse case measured, with small changes:
   - consult disk whenever it holds a longer prefix than RAM;
   - stream disk writes instead of staging whole snapshots in RAM;
   - count Flash-Next RAM checkpoints by unique bytes;
   - let new checkpoints evict lower-value entries instead of being refused
     (not simulated).
2. **Decide on the hybrid by the features you need**:

   | If you need | Then |
   | --- | --- |
   | Restarts at 100k+ with low disk wear | Hybrid: Phase 0 writes 3–10 GB per deep checkpoint |
   | 27B at long context with lower time to first token | Hybrid: ~90–220 ms less per checkpoint |
   | Many conversations, subagents or edit points kept warm | Hybrid: 10–20× more checkpoints per RAM budget |
   | Mostly Flash-Next, a few conversations, rare restarts | Phase 0 is enough |

3. **If the hybrid goes ahead, phase it:**
   - Phase 1: model interface split and the RAM chunk pool. Flash-Next is
     mostly there already.
   - Phase 2: the disk tier on the same chunks, with background compaction.
   - Phase 3 (optional): paged KV.

## Features to agree on

- Restart survival at long context, and acceptable disk wear per day.
- Several server processes sharing one cache directory.
- Edit and rewind points inside a conversation, or only its latest state.
- Subagents and shared system prompts as a primary workload.
- Output equality after a restore: greedy only, or sampled too.
- Model priority: Flash-Next vs 27B.
- Typical concurrency (2, 4, 8) and context length.
