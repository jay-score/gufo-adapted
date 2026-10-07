# Raw findings log (unedited notes kept during the runs; see experiments.md)

## E2 fn/w2 (subagents), 2026-10-07 21:03
- RAM budget with production-like Flash-Next (MTP, --sessions 2, ctx 260000):
  capacity_bytes=9230553088 (9.23 GB). ~3 checkpoints at 100k tokens.
- 22 `event=snapshot action=skipped reason=byte_capacity` lines: the RAM cache
  refuses new checkpoints once full instead of evicting (ranks prevent it).
- Defect: disk restore runs only when RAM has no hit at all
  (`src/cli/serve/text_model_runner.cpp:1877`, `!lease.cache_hit()`). Parent
  turn 8 (r32): RAM hit 4,661 (system prompt), disk had 24,599 (stored 21:04:43)
  -> prefilled 22,558 tokens, 24.3 s instead of ~2k tokens.
- E3: ideal 89.0%, actual 85.7%; misses: 4 requests, 20.7k tokens.
- Tokenization: re-tokenized prompts are 9-21 tokens shorter than server counts
  (decode/encode is not a round trip); rescale positions per request.
- E5 fn/w2 (staging 2.31 GB as in the run): actual 553,620; sim full 553,658
  (34/36 within 64 tokens); full+disk fix 573,599; chunked 573,606;
  chunked+disk fix 573,606. Hit-rate gain here = the disk-lookup fix.
  Writes: RAM capture copies 48.6 GB full vs 12.8 GB chunked; disk 16.3 vs 5.2 GB.
- Auto staging in the production-like Flash-Next run: 2.31 GB, i.e. no
  checkpoint past ~80k tokens can be persisted.

## E2 fn/w3 (multi-user chat)
- RAM 9.11 GB, staging 2.28 GB. E3: ideal 87.9%, actual 87.6% (1 miss, 991 tokens).
- E5: sim full 743,755 vs actual 745,248 (53/60 within 64); every variant 743,755:
  no hit-rate gain from chunking or the disk fix at these depths (<= ~40k).
- Accounted RAM checkpoint bytes 80.0 GB full vs 24.5 GB chunked; disk 13.5 vs 6.3 GB.
  Caveat: for Flash-Next the RAM figure is accounting only (KV already borrowed,
  #445); it is physical copy volume only for 27B.

## E2 fn/w4 (restart)
- RAM 8.92 GB, staging 2.23 GB. E3: ideal 89.7%, actual 88.5%.
- After restart: agent restored 51,243 from disk (ideal 54,968); chat restored
  12,762 from disk. New subagent sharing the 6.6k system prompt reused nothing:
  no checkpoint at that boundary (learned boundaries need more conversations).
- Checkpoints past ~70k tokens hit `staging_capacity` (2.06-2.28 GB files vs
  2.23 GB staging): never persisted. The restart happened at ~55k, below that
  cliff, so W4 does not exercise it.
- E5 (chunked staging = new bytes only): all variants 872,701 vs actual 868,875
  (30/33 within 64). No hit-rate gain. Accounted RAM 85.2 GB vs 16.9 GB;
  disk writes 26.7 GB vs 6.7 GB.

## Flash-Next W2-W4 so far
Chunking adds no reuse beyond the one-line disk-lookup fix at these depths
(<= 80k). Its gain is write volume: disk writes 3-4x lower.

## E2 q27/w1 (real pi agent to 149k)
- RAM 22.97 GB, staging 5.74 GB. 36 requests; E3 ideal = actual = 95.4%
  (linear agent: live session reuse). E5 all variants equal (36/36 agree).
- Captures: 35 live checkpoint captures, 5.9 s total, max 399 ms; ~28 ms/GB.
  At 149k a 10.0 GB capture takes 235-265 ms per checkpoint (real device copy
  on 27B). Accounted RAM checkpoint bytes 651 GB full vs 42 GB chunked.
- Disk: only 6 writes (16.5 GB); 85 `staging_capacity` skips. Past ~40k tokens
  (two queued 27B files > 5.74 GB) nothing reaches disk: a restart would lose
  the conversation. 39 RAM byte_capacity refusals (budget holds 2 checkpoints
  at 149k).

## E4 so far (analyze_e4.py)
- fn w2/w3/w4: capture totals 0.5-0.9 s per workload, max 110-246 ms.
  staging skips 8/9/29; RAM refusals 22/41/27.
- Disk write rate fn/w2: 16.3 GB in ~2.5 min of workload.

## E2 q27/w2 (subagents)
- RAM 23.2 GB, staging 5.80 GB. E3 ideal 89.0%, actual 85.6%; 2 misses,
  20.4k tokens, ~60.5 s (the parent-after-forks miss, disk lookup defect).
- E5: full 553,206 (34/36), full+disk fix 572,917, chunked 572,924.
  RAM copies 112.5 GB vs 28.0 GB (real on 27B); disk 37.8 vs 11.7 GB.

## E2 q27/w3 (multi-user chat)
- RAM 23.3 GB, staging 5.84 GB. E3 ideal 89.3%, actual 88.8% (2 small misses).
- E5: all variants 853,943 vs actual 853,387 (55/60 within 64). No hit gain.
  RAM copies 194.7 GB vs 51.7 GB (real on 27B); disk 31.2 vs 13.5 GB.

## E2 q27/w4 (restart at ~55k)
- RAM 23.0 GB, staging 5.75 GB. E3 ideal 89.7%, actual 87.7%.
- After restart the agent restored an older disk entry: its deeper checkpoints
  were skipped by staging -> 13,121 tokens re-prefilled (~36.6 s).
- New subagent after restart: no checkpoint at the 6.6k shared boundary ->
  6,634 tokens (~21.6 s); neither design fixes this.
- E5: full 869,489 (28/33), full+disk fix 869,489, chunked 882,556
  (+13,067 tokens, ~36 s): first hit-rate gain from chunking, via persisting
  deep checkpoints (staging only holds new bytes).
  RAM copies 194.5 GB vs 36.7 GB; disk 70.9 vs 15.9 GB.

## E5 simulated restarts on q27/w1 (sim full matched 36/36 on the real trace)
| restart before request | depth | today restores | chunked restores |
| 12 | 60,622 | 60,396 | 60,396 |
| 20 | 104,505 | 75,026 | 104,386 |
| 28 | 148,523 | 75,026 | 133,842 |
Today's deepest persisted 27B checkpoint stops at ~75k (staging); deeper
restarts re-prefill 29k-59k tokens.
27B prefill at 90-150k depth is 196-250 tok/s (measured in w1), so a restart
at 104k costs ~2 min extra today, at 148k ~4.5-5 min (58.8k tokens).

## Note: GPU overlap during E6
chunkcopy ran 22:31:59-22:32:05 concurrently with E6 q27 w3-c4 (a waiter bug:
pgrep -x cannot see scripts started via `env bash`). Those chunkcopy numbers
are discarded and rerun after E6; E6 requests in flight at 22:32 shared the GPU
for ~6 s.

## E6 q27 w3-c4 (4 sessions, 4 workers, 10 users, 100 requests, ctx 131072)
- RAM 20.2 GB, staging 5.05 GB. E3 ideal 86.7%, actual 86.1%.
- 92 RAM refusals; disk 36 writes, 33.7 GB, write_ms total 903 s over ~29 min
  (writer busy ~half the time); 157 min_step skips, 1 staging skip.
- E7: all variants 956,214 tokens; prefill today 500.6 s sim vs 478.4 s measured.
  Hybrid: disk 33.6 -> 20.1 GB, capture 6.6 -> 1.7 s.

## Physical vs accounted RAM (W1 logs)
- fn: accounted retained 7.1-8.9 GB; host_available_mib 16.5 GB at start,
  10.7-12.4 GB during the session: physical drop 4-6 GB including the live
  session -> borrowed KV is physically shared (supports unique-bytes accounting).
- q27: accounted 16.5-22.3 GB; host_available 42.9 -> 22.6-28.5 GB (drop
  14-20 GB): full copies are physical.

## E6 q27 w2-c4 (subagents at concurrency 4)
- RAM 20.1 GB, staging 5.03 GB. E3 ideal 89.2%, actual 89.0%: the W2
  parent-after-forks miss does not happen with 4 sessions (parent stays live).
- E7: all variants 576,353 (sim) vs 576,243 actual; prefill 228.5 s sim vs
  229.9 s measured. Hybrid: disk 36.1 -> 11.4 GB; capture 3.0 -> 0.5 s.
- E4: 22 RAM refusals, 8 staging skips, disk write time 96.6 s for 36.1 GB.

## E6 fn w3-c4
- RAM 8.48 GB, staging 2.12 GB. E3 ideal 87.5%, actual 87.1%.
- 77 RAM refusals, 10 staging skips; disk 34 writes 16.2 GB, 31.4 s write time.
- E7: all variants 1,008,872; prefill 157.8 s sim vs 158.6 s measured.
  Hybrid disk 15.9 -> 8.7 GB.
