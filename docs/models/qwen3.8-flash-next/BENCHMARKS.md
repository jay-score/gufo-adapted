# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user MTP, concurrency, loading and llama.cpp MTP at 64K/128K:
September 23, 2026; other results: September 22.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.

September 29 Windows screenshot-cache check (separate from the throughput tables):
on a 14.5K-token C1 tool conversation, appending two 512x512 screenshots changed
time to first token from **12.56 / 11.21 s** to **2.36 / 0.83 s**. Each append
processed **317** new prompt tokens instead of **14,885 / 15,213**. One matched
sequence per build, UD-Q4_K_XL, Q8_0 MTP, BF16 projector, context 200,000,
greedy, thinking off, disk cache off. The first screenshot includes lazy vision
startup in both runs. Response text matched; these are workload-specific
latencies, not a general throughput claim.
[Measurements and identities](artifacts/image-prefix-cache.json).

October 6 Windows incremental-checkpoint check (separate from the throughput
tables): on matched C1 IQ4_XS/Q8_0 MTP conversations growing from 16K/32K
backgrounds, snapshot elapsed falls **552.00 -> 243.36 ms / 896.99 -> 266.25 ms**.
Median warm-turn latency falls **5.168 -> 4.747 s / 5.667 -> 4.869 s**
(**8.1% / 14.1%**). Four corresponding warm turns per depth, 2079 new prompt
tokens, 126-128 actual output tokens, greedy, thinking off, context 49152, disk
off; replies and token/draft counts match. These are workload-specific HTTP
latencies, not a refresh of the kernel throughput tables.
[Method and evidence](../../plans/flash-next-host-snapshots.md#production-measurements).

[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1628.52 | 489.59 | +232.6% | 26.04 | 22.20 | +17.3% |
| 4,096 | 1523.39 | 455.24 | +234.6% | 25.98 | 21.21 | +22.5% |
| 8,192 | 1499.13 | 428.85 | +249.6% | 25.91 | 20.40 | +27.0% |
| 12,288 | 1477.98 | 400.21 | +269.3% | 25.46 | 19.64 | +29.6% |
| 16,384 | 1457.16 | 375.89 | +287.7% | 25.20 | 18.95 | +33.0% |
| 32,768 | 1421.93 | 301.31 | +371.9% | 24.33 | 16.54 | +47.1% |
| 65,536 | 1304.01 | 221.94 | +487.6% | 23.21 | 11.62 | +99.7% |
| 131,072 | 1292.02 | 144.78 | +792.4% | 21.93 | 7.98 | +174.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1602.82 | 468.76 | +241.9% | 32.18 | 31.73 | +1.4% | 59.41 | 48.12 | +23.5% |
| 4,096 | 1506.32 | 423.79 | +255.4% | 34.57 | 34.48 | +0.3% | 47.92 | 45.71 | +4.8% |
| 8,192 | 1492.95 | 394.98 | +278.0% | 34.97 | 32.20 | +8.6% | 50.02 | 44.31 | +12.9% |
| 12,288 | 1488.03 | 365.91 | +306.7% | 35.18 | 32.32 | +8.8% | 39.95 | 43.53 | -8.1% |
| 16,384 | 1471.81 | 341.98 | +330.4% | 33.84 | 32.07 | +5.5% | 50.55 | 43.66 | +15.8% |
| 32,768 | 1449.56 | 277.61 | +422.2% | 30.67 | 25.41 | +20.7% | 46.81 | 39.28 | +19.2% |
| 65,536 | 1316.70 | 208.04 | +532.9% | 31.78 | 19.65 | +61.7% | 45.12 | 27.75 | +62.6% |
| 131,072 | 1335.91 | 135.42 | +886.5% | 28.12 | 14.63 | +92.2% | 42.32 | 20.96 | +101.9% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.85 | 22.34 | +15.7% |
| 2 | 45.70 | 37.08 | +23.2% |
| 4 | 76.29 | 54.82 | +39.2% |
| 6 | 95.66 | 65.34 | +46.4% |
| 8 | 108.67 | 68.79 | +58.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.09 | 31.77 | +1.0% | 59.20 | 46.93 | +26.1% |
| 2 | 51.68 | 42.79 | +20.8% | 93.16 | 52.57 | +77.2% |
| 4 | 75.92 | 48.78 | +55.6% | 127.91 | 48.23 | +165.2% |
| 6 | 89.18 | 52.59 | +69.6% | 142.14 | 48.51 | +193.0% |
| 8 | 106.47 | 61.92 | +71.9% | 157.22 | 58.13 | +170.5% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.45 | 117.83 | +662.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)

## Windows clang 23 prefill regression check

October 7, 2026: native `gufo bench`, TheRock 10.0.0, gfx1151, release
baseline `dce753fc36aa` versus the PR #459 adaptation. UD-IQ4_XS three-shard
weights, with shared Q8_0 MTP (maximum seven drafts). This model is a control;
the paired Q4/Q5 cache change is qualified by operators and resource metadata.
C1, depth zero, capacity 4096, temperature zero, seed 459. Two rounds in
baseline/candidate then candidate/baseline order, three repetitions per process;
the table averages the two process means. Loading is excluded and prompt shapes
are warmed. This is a matched fork comparison, separate from the HTTP/reference
tables above. [Build/model identities and all per-process statistics](artifacts/prefill-spills-pr459.json)
and [qualification, commands and limits](../../plans/prefill-spills-pr459.md).

| Mode / workload | Baseline (tok/s) | Adapted (tok/s) | Gain |
| --- | ---: | ---: | ---: |
| AR pp8 | 73.90 | 73.09 | -1.1% |
| AR pp16 | 116.47 | 116.82 | +0.3% |
| AR pp64 | 292.83 | 291.03 | -0.6% |
| AR pp2048 | 984.60 | 978.99 | -0.6% |
| AR tg128 | 27.18 | 27.16 | -0.1% |
| MTP pp8 | 70.70 | 70.19 | -0.7% |
| MTP pp16 | 106.41 | 109.87 | +3.2% |
| MTP pp64 | 287.23 | 287.28 | +0.0% |
| MTP pp2048 | 979.44 | 954.96 | -2.5% |
| MTP tg128 | 43.18 | 42.40 | -1.8% |

The initial MTP deficit did not persist in a focused confirmation with five
repetitions per process and the same reversed order. Both sets of measurements
remain in the artifact; small control differences are not claimed as speedups.

| MTP confirmation | Baseline (tok/s) | Adapted (tok/s) | Gain |
| --- | ---: | ---: | ---: |
| pp2048 | 967.54 | 971.77 | +0.4% |
| tg128 | 43.10 | 43.00 | -0.2% |
