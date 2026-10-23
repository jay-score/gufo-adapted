# Qwen3.8 Flash-Next on Windows

Evidence from the native Windows port. [BENCHMARKS.md](BENCHMARKS.md),
[QUALITY.md](QUALITY.md) and [EXPERIMENTS.md](EXPERIMENTS.md) stay identical
to official Gufo and describe Linux measurements. Windows results go here, so
syncs never conflict in the shared model documents. Add new entries at the top
of each section.

## Upstream sections that do not apply

PR #699 is [reverted on Windows](../../WINDOWS-UPSTREAM-SYNC.md#pr-699-windows-first-request-mtp-decode-regression)
(see `tools/windows/known-regressions.json`). These upstream entries describe
code the Windows build does not contain:

- [Exact sampled decode](EXPERIMENTS.md#exact-sampled-decode-october-21-2026) experiments.
- [Sampled decode follow-up](BENCHMARKS.md#sampled-decode-follow-up-october-21) benchmarks.
- [Exact sampled decode](QUALITY.md#exact-sampled-decode-october-21-2026) qualification.

## Benchmarks

### YaRN matched follow-up (October 23, 2026)

The retained base/candidate executables and all 143 runtime files still match
the October 22 hashes. Fresh unchanged-baseline controls verify effective
8 GiB RAM caches at startup and restart. AR restore observations become
inconclusive; three MTP total-duration cells initially remain flagged despite
matching decode time and token/draft work.

A same-path ABBA control stages the original binaries at one executable path,
retaining each arm's SHA-256. The repeated warm thinking/tool request has the
same output, zero prefill tokens, eight completion tokens and seven/five
proposed/accepted drafts:

| Pair/order | Baseline wall (ms) | Candidate wall (ms) |
| --- | ---: | ---: |
| Base, then candidate | 192.83 | 170.33 |
| Candidate, then base | 193.48 | 178.76 |

The unchanged baseline thus reproduces the extra duration previously seen on
the candidate. The standalone ABBA comparison still flags one legacy-tool
snapshot phase: its baseline samples are 12.88/13.60 ms and candidate samples
20.10/25.64 ms. The original, equally budgeted baseline records 12.30/30.30 ms
for that same phase. Every observation is retained.

`verified-builds.py` combines all five valid pairs through the upstream
comparison/qualification functions after validating build hashes, runtime
files, harness/settings and effective budgets. It reports 577 passed timing
cells, 283 inconclusive, no confirmed slowdown and no output/work changes.
Executable-path differences are recorded explicitly; raw reports are unchanged.
Clamped and instrumented runs do not enter this assessment. Original failed
results remain available alongside the controls. No speedup or full Windows
performance pass is claimed, and no inference workaround is retained.

Evidence: `build/sync-350-2026-10-23/`, especially `same-path-arms.json`,
`same-path-comparison.json`, `verified-builds.py` and
`verified-builds-comparison.json`. The next upstream layer may proceed while
these timing limits remain visible.

### YaRN adoption control (October 22, 2026)

Windows base `d10a852f`, candidate production code including `2354ba89`,
TheRock 10.0.0, the same Q4_K_XL/shared-Q8 artifacts, 32K context, two sessions,
8 GiB RAM cache and a three-second arm pause. Two alternating pairs per mode
send the varied 11,036-token prompt with reuse enabled and then disabled.
Every response is `ACK`, every request reports zero reuse, and all servers
stop cleanly. Complete responses, identities, DLL hashes and per-request
measurements are retained in `build/sync-350-2026-10-22/cold-control-1/`.

Prefill, adjusted wall/queue/TTFT and AR decode remain within the existing
paired gates. Snapshot cells and first MTP decode have noisy observations;
the overall result is **inconclusive**. No speedup or complete performance
qualification is claimed. An unchanged-base MTP control also reports moved
snapshot cells on identical binaries; every observation is retained.
The matched API timing investigation is recorded in the
[sync follow-up](../../WINDOWS-UPSTREAM-SYNC.md#october-23-2026-pr-350-qualification-follow-up).

One retained pp2048/tg128 spot per mode uses the same target, MTP sidecar and
toolchain. All four greedy output hashes match; both MTP runs propose 109
and accept 58 drafts. These single spots do not qualify a speedup.

| Mode | Base/head pp2048 (tok/s) | Base/head tg128 (tok/s) |
| --- | ---: | ---: |
| AR | 1806.60 / 1781.47 | 28.41 / 29.14 |
| MTP | 1744.80 / 1748.52 | 39.98 / 38.99 |

### Sync first-decode control (October 21, 2026)

Native TheRock 10.0.0; same four-shard Q4_K_XL/shared-Q8 MTP artifacts,
32K context, two sessions, thinking off, temperature zero and 8 GiB RAM cache.
Each fresh process handles a varied 11,036-token prompt, then repeats it with
prompt reuse disabled. Both requests report zero reuse and return `ACK`.
Every observation is retained separately; no averaging is used.

| Binary/control | First decode (ms) | Repeat decode (ms) |
| --- | ---: | ---: |
| Windows base `db4f6fbc`, initial | 57.51 | 52.79 |
| Same base, focused control | 53.86 | 51.40 |
| Same base, history control | 51.32 | 52.85 |
| Full upstream import, initial | 60.90 | 51.88 |
| Full import, focused repeat | 63.35 | 52.32 |
| Import without #699, isolation | 53.55 | 52.92 |
| Import without eager vision setup, rejected | 59.60 | 51.83 |
| Full predictor catch-up, scalar isolation | 52.75 | 52.72 |
| Full predictor catch-up, combined policy, rejected | 66.46 | 52.96 |
| Explicit Q8 startup preparation, rejected | 67.02 | 51.59 |
| Rebuilt explicit #699 revert | 53.25 | 51.52 |

PR #699 is reverted on Windows. The scalar isolation did not produce a
stable workaround; none of the experimental changes is retained. See the
[regression record](../../WINDOWS-UPSTREAM-SYNC.md#pr-699-windows-first-request-mtp-decode-regression)
for evidence and requirements before reintroduction. These controls do not
replace Linux sweeps or complete Windows performance qualification. Cumulative
asynchronous copy totals remain unqualified when copied byte counts differ.

#### Standard post-revert spot

One repetition of pp2048/tg128 at depth zero, same artifacts/toolchain:

| Mode | pp2048 (tok/s) | tg128 (tok/s) | Proposed/accepted drafts |
| --- | ---: | ---: | ---: |
| AR | 1807.04 | 28.88 | 0/0 |
| MTP | 1753.56 | 38.65 | 109/58 |

Both return the same greedy 128-token hash
`5177a75da6988d5ce9a14e1a744ed16829784eda18d40139d8a1e76469af8412`.
These retained standard spots do not qualify a sync speedup. See the sync
log for the inherited numerical-oracle failure and open timing limits.

### Cold-prompt control (October 20, 2026)

On the native Windows port, restoring cached overlapped n-gram reads fixes an
NTFS direct-I/O regression introduced during the upstream forward-port.
This local control uses the same four-shard `UD-Q4_K_XL` target and shared-Q8
MTP sidecar, one session, context 150000, thinking enabled, greedy tg128,
5 GiB RAM cache and no disk cache. Both requests explicitly disable prompt
reuse and process 16459 prompt tokens; the second repeats the same text in
the same process. A fresh process does not imply a cold filesystem cache.

| Request | Rebased direct I/O (tok/s) | Restored Windows reads (tok/s) |
| --- | ---: | ---: |
| First long prompt | 783.93 | 1693.09 |
| Same prompt, no prompt reuse | 1634.86 | 1705.27 |

The saved default-branch engine (`5d14289a`, unchanged engine sources through
`db324991`) measures 1387.87 and 1520.73 tok/s for the same two requests.
Rerunning the saved pre-fix binary after the fixed run reproduces 787.28 and
1637.85 tok/s, ruling out run order as the explanation for the large gain.
GUI live rates match the server's measured per-chunk processing rates.
The fixed standard warmed CLI MTP pp2048/tg128 control measures 1729.35 and
38.01 tok/s respectively; its synthetic workload differs from HTTP prose.
These focused Windows controls do not replace the Linux tables in
[BENCHMARKS.md](BENCHMARKS.md).

## Quality

### OS info adoption (October 23, 2026)

PR #709 changes the host-memory query to the shared native OS leaf, using the
same Windows API/result and preserving the host reserve. Native CPU OS-oracle
and budget checks pass. No model or inference arithmetic changes.

The HIP executable at `be07d64b63ce` passes the bounded MTP RAM startup/retry
fixture with the same Q4_K_XL/shared-Q8 artifacts, context 8,192, one session
and an effective 1 GiB RAM cache. Cold generation consumes 2,294 prompt tokens;
retry reuses all 2,294 with zero prefill and exactly matches the response.
Each request proposes 19 and accepts 16 drafts. RAM-only startup leaves the
durable model-identity cache empty, and the server stops cleanly.

Commands, binary identity, responses and logs are in
`build/sync-709-2026-10-23/ram-startup-job.json` and `ram-startup-mtp/`.
This is a platform-budget smoke check; it does not refresh the existing
numerical or performance qualification.

### YaRN adoption, October 22, 2026

- YaRN-off full-model matched-token decode: all 12 logit rows
  (11,919,360 bytes) and token IDs are byte-identical to the exact Windows base.
  Logit SHA-256: `62ac18232069b28ed69430997cbb7deb5b812ee985791f92e9f6a5d38a3d46a7`.
- Native sampling and cache-replay checks pass, including greedy/default-sampled
  RAM and serialized continuation, sampler state and actual MTP drafting.
- AR/MTP 4K, two-slot component round trips pass, including restored device
  components, next-step logits/tokens and actual draft work.
- CPU HF-golden/scalar-oracle and GPU attention, projection, selection and
  snapshot-copy checks pass. Upstream's cache-policy golden initially finds
  the MSVC hexfloat-padding difference; a Windows-only canonicalization fixes
  that failure without changing the golden or Linux output.
- The Windows-owned YaRN snapshot fixture at factor 1.5625 passes AR/MTP
  checkpoint and serialized replay. It rejects the pre-YaRN executor marker,
  checks clean reset and exact recovery with matched prefill boundaries, and
  rejects a snapshot at factor 2.5. MTP actually drafts 14 tokens in the bounded
  continuation. Its `--extended` run also passes AR/MTP at capacity 409,600,
  with a checkpoint at token 262,145 and 13 actual MTP draft proposals. This
  proves exact replay across the native boundary, not external long-context
  answer quality.
- Public MTP serving with `--enable-yarn --context 262145`, one session and
  1 GiB RAM cache passes cold generation, RAM retry and durable disk restart
  with exact response replay. Restart reuses 2,287 of 2,294 prompt tokens and
  proposes 19/accepts 16 drafts. This API fixture checks scaled capacity and
  disk identity; actual above-native inference is covered by the native test.
  Above-native serving without opt-in and a no-op opt-in at 32K are rejected.
- Selected long-context/cache API runs pass seeded outputs, interruption and
  restart assertions in AR/MTP. The initial matched pairs preserve expected
  cache work but fail timing gates. Later unequal-budget controls differ in
  warm image-prefix work and cannot qualify the original profile. All failed
  comparisons remain retained; the October 23 follow-up investigates them.

Evidence is retained under `build/sync-350-2026-10-22/`, including earlier
failed observations. Correctness scope is recorded above; the full timing
matrix remains unqualified. See the sync follow-up for current attribution
controls and their effective cache budgets.

### Sync, October 21, 2026

PR #699 introduced a Windows/TheRock first-request MTP decode regression and
is explicitly reverted, while its upstream ancestry is preserved. Do not
reapply it unchanged. The remaining sync retains #692, #540 and #690, together
with the Windows compatibility changes. The initial full import passes the
independent attention/projection/expert/snapshot and sampled-verification
checks, AR/MTP continuation probes and long-context/deep-cache correctness;
that does not qualify its cold-start timing. Final post-revert checks and
remaining limits are recorded in the [sync log](../../WINDOWS-UPSTREAM-SYNC.md).
The extra scalar MTP audit fails the image head-mixer tolerance on both the
candidate and the exact Windows base with identical errors (relative RMS
0.0000695 versus limit 0.00005). This inherited numerical qualification gap
remains open; the tolerance is unchanged and no oracle pass is claimed.

### n-gram I/O, October 20, 2026

Restore the Windows-only cached overlapped file open that the upstream
forward-port dropped. Linux retains direct I/O.
The existing `qwen38_flash_next.ngram` check passes independent hash fixtures,
BF16/IQ4_NL row decoding, distant offsets, descriptor identity and counters.
Both uncached 16459-token MTP HTTP requests retain byte-identical greedy
128-token completions, prompt counts and zero prompt reuse before/after the
change. The GUI processing-time counter agrees with response and server
timings; all 14 launcher process/metrics tests pass. This is a focused I/O
qualification, not a new full-model numerical or platform qualification.
Local commands, executable identities, full responses and metric scrapes are
retained in `build/prefill-investigation/`.

## Experiments

### Upstream sync (October 21, 2026)

| Experiment | Outcome |
| --- | --- |
| Import #699 unchanged | Reverted/deferred: Windows first MTP HTTP decode is 60.90/63.35 ms against baseline controls of 51.32-57.51 ms. Warm repeats and output/cache/draft counts agree. |
| Import with #699 removed | Retained for final validation: isolation control measures 53.55/52.92 ms first/repeat decode. |
| Remove eager empty vision preparation | Rejected: first decode remains 59.60 ms. |
| Restore full predictor catch-up | Rejected as a workaround: scalar isolation measures 52.75 ms, but the combined scalar/batch fallback measures 66.46 ms. |
| Prepare missing Q8 predictor functions at startup | Rejected: first decode remains 67.02 ms. No speculative platform policy or kernel preparation change is retained. |

See the [known regression and reintroduction requirements](../../WINDOWS-UPSTREAM-SYNC.md#pr-699-windows-first-request-mtp-decode-regression)
and [individual measurements](#sync-first-decode-control-october-21-2026).

### n-gram I/O (October 20, 2026)

| Experiment | Outcome |
| --- | --- |
| Restore cached overlapped n-gram reads on Windows | Retained: the upstream forward-port lost `O_CONCURRENT_RANDOM` and used direct reads of a model shard that is also mapped, triggering NTFS serialization. The first uncached 16459-token MTP prompt improves from 783.93 to 1693.09 tok/s; the identical uncached repeat measures 1634.86 to 1705.27. Both tg128 completions remain byte-identical, and independent n-gram row/hash tests pass. GUI live processing rates agree with server timings. Linux file-open behavior is unchanged. See [controls](#cold-prompt-control-october-20-2026). |
