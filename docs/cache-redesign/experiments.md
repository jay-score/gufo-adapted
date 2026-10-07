# Experiments

Measurements that decide between [option A and option C](options.md). All of
them run on unmodified main binaries. The drivers and analysis scripts are in
[scripts/](scripts/), and small result files in [results/](results/). The
scripts expect the gufo binary at `bin/gufo` next to them and write `results/`
and `cache/` there.

## Scope (agreed 2026-10-07)

- **Models:** Flash-Next and Qwen 27B. DeepSeek V4 is out of scope: its
  compressed KV leaves little to share.
- **Workloads:** all four in [E2](#e2-workload-traces).
- **Restarts:** surviving a restart is essential, so the disk tier is measured
  as a first-class feature.
- **Disk space:** the test host has about 76 GB free (92% used). Disk cache
  budgets stay small, each run's cache directory is deleted afterwards, and
  `df -h /` is checked before every run. A `cache` suite run alone leaves
  about 8 GiB.

## E1. Snapshot size model

**Question.** How large are a checkpoint's fixed part and its per-token part,
per model and configuration? This bounds what sharing KV can save.

**Method.** Capture checkpoints at several depths (for example 2k, 8k, 32k,
64k and 128k tokens) and read the `live_checkpoint bytes=` and disk
`payload_bytes=` log fields. Fit fixed bytes + bytes per token. Vary the
settings that may change the fixed part: speculative mode (MTP or DFlash2
state), `--context`, and thinking.

**Models.** Flash-Next UD-Q4_K_XL and 27B UD-Q4_K_XL.

## E2. Workload traces

**Question.** What does real use look like for the cache?

**Method.** Run the server with `--trace PATH` (it records the rendered
prompts) and keep the logs (they record cache events). Workloads:

| Id | Workload | Why |
| --- | --- | --- |
| W1 | One coding-agent session past 100k tokens (`tests/functional/pi_agent.py` replay) | Long linear run, the #275 shape |
| W2 | An agent whose subagents share a long prefix | Sharing across conversations |
| W3 | Several users or conversations alternating on few session slots | Multi-user chat; the #275 comment scenario |
| W4 | W1 with a server restart in the middle | Disk tier value |

Start with Flash-Next, then 27B, at the default RAM budget and a disk budget
that fits the free space.

## E3. Missed reuse

**Question.** How much prefill does today's cache fail to avoid, and why? This
is the upper bound for any redesign.

**Method.** For each request in the traces, compare the reused tokens with the
longest prefix the prompt shares with anything earlier in the trace
(re-tokenized offline). Convert the gap to prefill seconds using the measured
prefill rate at that depth. Attribute each gap to one of:
- RAM eviction;
- disk eviction (`lru` or `superseded`);
- checkpoint spacing (grid or `min_step`);
- a checkpoint skipped for capacity;
- a prompt changed upstream, which is a template problem no cache design
  fixes.

## E4. Overhead of the current design

**Question.** What does the cache cost on the request path today?

**Method.** From the logs:
- capture and snapshot time per request, including captures that overlap a
  disk write;
- disk bytes written per hour and `write_ms`;
- restore time.

## E5. Simulation

**Question.** How much would option C save on the same traces?

**Method.** Replay the token traces under the same budgets through a small
simulator with two policies:
- (a) today's rules: full-size accounting and the current eviction ranks;
- (b) chunked: shared 2,048-token KV chunks plus fixed state per checkpoint.

Report prefill seconds saved, bytes written and checkpoints retained. Trust
(b) only after (a) reproduces the reuse measured in E3.

## Decision rule (proposal)

Redesign if the chunked policy saves at least about 20% of prefill time on the
agent traces, or materially cuts capture and write overhead. Otherwise keep
the current design and merge #409. The threshold is still to be agreed.

## Results

### E1. Snapshot size model (2026-10-07)

Disk `payload_bytes` against checkpoint tokens, main-equivalent build,
`--sessions 1`. Every configuration fits fixed + per-token exactly (largest
residual under 0.01%) **(measured)**:

| Configuration | Fixed | Per token | At 100k tokens |
| --- | ---: | ---: | ---: |
| Flash-Next, AR, `--context 260000` | 113.5 MiB | 25.35 KB | 2.65 GB |
| Flash-Next, AR, `--context 32768` | 113.5 MiB | 25.35 KB | 2.65 GB |
| Flash-Next, MTP, `--context 260000` | 113.8 MiB | 27.46 KB | 2.87 GB |
| 27B, AR, `--context 256000` | 152.4 MiB | 65.54 KB (64 KiB) | 6.71 GB |
| 27B, AR, `--context 32768` | 152.4 MiB | 65.54 KB | 6.71 GB |
| 27B, DFlash2, `--context 256000` | 232.4 MiB | 65.54 KB | 6.80 GB |

- **Context size doesn't matter.** Checkpoint size depends only on the token
  position, not on `--context`.
- **Speculative drafting adds state.** MTP adds 2.1 KB per token; DFlash2 adds
  80 MiB of fixed state.
- **27B is hybrid too.** It has a 152 MiB fixed part, but its per-token cost
  is 2.6× Flash-Next's, so sharing KV would save proportionally more on 27B.
- **Automatic RAM budgets are small.** With one session loaded they were
  12.9–17.5 GB for Flash-Next and 33–34 GB for 27B. At 100k tokens that is
  about 4–6 Flash-Next checkpoints or 5 for 27B.
- **Deep checkpoints don't reach disk.** The automatic staging limit (3.4 GiB
  with Flash-Next loaded) is below one Flash-Next checkpoint at 131k tokens,
  and two 27B checkpoints at 64k exceed even 6 GiB. Today's defaults skip
  persisting those deep checkpoints (`reason=staging_capacity`).
- **The RAM cache can refuse the newest checkpoint.** At 127k tokens it
  refused the prompt checkpoint (`reason=byte_capacity`, 3.34 GB against a
  14.6 GB budget), because the request's own earlier checkpoints already
  filled the budget.

Scripts: `e1_snapshot_size.py`, `run_e1.sh`, `fit_e1.py`.
