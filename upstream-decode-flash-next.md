# Upstream decode-speed review — Qwen3.8 Flash-Next (2026-10-12)

Fork: `windows-port` at `ab4d9818`. `official` = `gufo-org/gufo` (fetched, not merged).
Scope: PRs open / closed / merged that may improve decode (token generation) speed for Qwen3.8 Flash-Next. Prefill-only work is listed as near-misses so it is not re-investigated.

## 1. Direct decode win, merged upstream, missing locally — backport first

### #518 MERGED — perf(qwen-flash): faster decode kernels
- URL: https://github.com/gufo-org/gufo/pull/518 (commit `7e4621c3`, merged Oct 10)
- What: 15 small gfx1151 decode-kernel changes (Q8_1 row fusion, up-front reads, DPP reductions, MoE grouping/epilogue, decode attention batching, Q4_K branch removal, vector loads). No new flags; prefill unchanged; outputs bit-identical.
- Measured (UD-Q4_K_XL + Q8_0 MTP, 2x Strix Halo, median of 4 ABBA rounds vs main `fd747a51`): +6–7% single-session decode, +3.9–4.4% aggregate at 2–4 sessions; `gufo bench tg32` +6–7%; `pp4096` flat.
- Local status: `git merge-base --is-ancestor 7e4621c3 HEAD` fails on `windows-port` — **not imported**. Author notes rebase friction only with #426/#350, neither in this fork's log.

## 2. Open PRs claiming decode gains

### #570 OPEN — feat(qwen38-flash-next): UD-IQ4_XS support, faster prefill/decode, token-history drafting
- URL: https://github.com/gufo-org/gufo/pull/570 (+5278/−485)
- Decode pieces: large dense projections decode from a Q5_1 copy of Q8_0 (`GUFO_DECODE_Q5=0` disables); per-token compaction for decode attention; token-history drafting (suffix automaton + lookup index, on by default, `GUFO_SUFFIX=0` disables).
- Caveats: format support and speed share kernels (author could not split); no vs-main speed comparison (main cannot load the model); only greedy 24-token equality on UD-IQ4_XS.
- Local status: fork already loads IQ4_XS/IQ3_S (`src/models/qwen38_flash_next/weights.cpp`, re-encoded to Q8_0 at upload), so only the Q5_1-decode / compaction / drafting deltas are new. Needs matched-token logit + perplexity validation before marking complete.

### #561 OPEN — feat(qwen-flash): Q8 KV cache precision mode
- URL: https://github.com/gufo-org/gufo/pull/561 (stacks into #566)
- What: opt-in `--kv-quant {f16,q8}`; Q8_0 KV halves KV bytes; narrow decode keeps scalar dequant (memory-bound, fewer bytes win).
- Measured: decode ~43–55 tok/s greedy, at/above F16 baseline; main value is 4x262K sessions where F16 cannot fit. Snapshot magic bump orphans old snapshots — migration cost.

### #513 DRAFT — feat(qwen38_flash_next): memory-bounded routed-expert streaming (--expert-cache)
- URL: https://github.com/gufo-org/gufo/pull/513
- Status: explicit WIP, **no GPU run**, bit-exactness and throughput unmeasured. Memory-bounding, not a speedup — do not backport for speed.

## 3. Near-misses (perf-labeled but not decode)

- #551 MERGED (128 n-gram readers): fresh-prompt prefill only; decode explicitly untested.
- #571 OPEN (prefill residual once per layer): author states "Decode is unaffected"; +1.5% prefill.
- #566 OPEN (int8 WMMA for Q8 prefill): author states "decode unchanged"; requires #561.
- #568 OPEN (XDNA2 NPU offload): prefill only (+7% on author's IQ4_XS fork); GPU default unchanged.
- #485 MERGED / #470 MERGED / #445 MERGED / #459 MERGED: prefill + MTP catch-up / chunking / checkpoint / spills. Fork already imported #485's MTP-boundary correctness fix (`0e5fa0a8`) and #470's attention/scheduling parts — not the perf remainder. tg tables flat where reported.
- #247 MERGED: Qwen27B Q4/Q8 decode, not Flash-Next kernels — not applicable.
- #572/#573, #536/#540: continuation/cache wiring — prefix reuse, not per-token decode.

## Recommendation

1. Backport #518 first (small, measured, flag-free, bit-identical).
2. Then evaluate #570's decode-only pieces individually (Q5_1 decode, attention compaction, suffix drafting) on UD-Q4_K_XL and IQ4_XS with exact-logit + acceptance checks; skip its format half.
3. Take #561 only for long-context/multi-session capacity; budget for the snapshot-magic migration.
