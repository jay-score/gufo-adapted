# Flash-Next end-of-turn accounting audit

The fork at `f693fab13bf6bd31e67de3c0ff77aea41133efd6` did not reproduce the
checkpoint failure reported in [upstream #400](https://github.com/gufo-org/gufo/pull/400).
This change adds focused regression coverage; it changes no engine, sampler,
prompt, cache format or production serving code.

## Accounting contract

The reviewed upstream implementation is pinned at
[`0564df5`](https://github.com/gufo-org/gufo/commit/0564df5ffcabb24b6b70df3a17f7fd8cce1718bb).
Its scalar and batched Flash-Next adapters disable engine EOS stopping and
account for the executed EOS and subsequent tokens. That approach was not
imported. Upstream discussion also reports additional end-of-turn latency;
those Linux measurements are not measurements of this Windows fork.

Here, `Session::FinishDecode` stops before accepting EOS, rolls target state
back to the kept prefix, and returns that same committed prefix. The MTP head
rewinds to the verification base. Both serving adapters leave EOS stopping
enabled and report the session's actual position. `TextRunnerPool::Request`
records every returned selection before publication; a stop string or
cancellation may publish fewer tokens without relabelling the executed state.
The pool reconciles that state when retaining the continuation checkpoint.

The audit checks these relationships independently: an autoregressive session
replays every committed token and must have byte-identical complete frontier
logits; the speculative session's physical position, token history and sampler
history must agree. Snapshot restoration must preserve continuation tokens,
full logits, RNG and any pending residual draw.

## Coverage

- Hosted `text_generation_scheduler_test`: EOS after 0-7 tokens with ordinary
  or multi-token decode, with and without first-token preview, paired requests,
  streaming text/usage, and a fully cached EOS continuation. These fixtures
  exercise the serving contract without a model.
- Explicit `qwen38_flash_next_eos_test`: real-model EOS offsets 0-7 and a
  complete natural turn, scalar and paired MTP sessions, GPU greedy, penalized
  greedy and sampled top-k=1 verification. A separate top-k=40 sampled turn
  reaches EOS after residual corrections. Snapshot continuation intentionally
  proceeds past EOS to exercise restored state rather than two empty stops.
- The same explicit test exercises serving with AR/MTP and one/two sessions:
  token budgets 1/2/3/8/128, warm EOS continuation, stops on the first output
  and inside a speculative block, callback cancellation, and concurrent EOS.
- `history_replay_test.py`: actual typed calls and identity-preserving tool
  replay in Chat Completions/Responses, buffered/streamed, retries and Chat cold
  controls. A separate larger-schema fixture requires an actual `add(2,3)`
  call and the exact BETA continuation, including concurrent requests.

The GPU test is explicit and excluded from ordinary builds. It requires a
compatible target and MTP sidecar and fails if its fixtures miss EOS, residual
correction or actual paired execution. Missing artifacts are not a pass.

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target qwen38_flash_next_eos_test -Jobs 4
.\build\gpu-test\qwen38_flash_next_eos_test.exe --model "$MODEL" --mtp-model "$MTP"
# Against an isolated Flash-Next server with --served-model-name gufo:
python tests/tools/history_replay_test.py --url http://127.0.0.1:18194/v1 --out build/mtp-accounting/replay.json
python tests/tools/history_replay_test.py --url http://127.0.0.1:18194/v1 --literal --out build/mtp-accounting/literal.json
```

## Recorded Windows validation

Native Windows 11/gfx1151, TheRock ROCm 10.0.0 (clang 23), MSVC 14.51 headers,
pinned x64-windows vcpkg dependencies, RelWithDebInfo production libraries.
The explicit GPU test uses checked exceptions rather than assertions. Fresh
CPU targets keep assertions enabled. The model is Unsloth UD-IQ4_XS with the
shared Q8_0 MTP sidecar, maximum seven drafts; its fresh SHA-256 hashes match
the [recorded Flash-Next identities](device-loss-pr390.md). The matching BF16
projector beside the model is auto-discovered; no images or vision weight
uploads are exercised. The GPU test uses
context 2048; HTTP uses context 4096, thinking off, seed 47 and isolated
memory caches. Platform tuning overrides and GPU queue overrides are unset.
No GUI settings or user cache were changed.

The unchanged serving baseline executable is `1f8e70791309-dirty`, SHA-256
`88f163be3a63241af05796e211ee06e8cb6240a117fb1e3855f401a6212772d5`.
It was built before the final history commit but contains that commit's final
source. Build and execution logs, raw HTTP responses and measurements are
retained under ignored `build/mtp-accounting/`.
The explicit test executable's SHA-256 is
`aee1a0b78d4dcd1a56d69ca9b22cdc0b4cb92cc89e3e873187652f6c281fe263`.

The final explicit GPU test passes 54 EOS cohorts (81 session trajectories)
with complete frontier equality, snapshot continuation and RNG checks.
Actual EOS stops occur at committed-prefix lengths 0/1/2/3/4/6 within the
ending decode step; offsets from the prompt to EOS are distinct from those
within-step positions. The separate sampled turn reaches EOS after ten
rejection cycles and retains byte-identical frontier logits. All four serving
configurations pass their budget, stop, cancellation, cache and concurrent
checks. Fresh `text_model_runner_test`, `text_generation_scheduler_test`,
`openai_chat_test` and `http_server_test` pass.

An additional 28 literal-history HTTP checks pass in MTP at one/two-session
capacity. They put `<|endoftext|>`, `<|im_end|>`, `<|image_pad|>` and a
non-vocabulary spelling in user content, historical arguments and tool output.
Both APIs and response modes preserve the exact BETA continuation, successful
termination and full-prompt retry reuse. The fixture does not require the
model to generate its own EOS spelling as literal text.

All 56 ordinary typed-tool HTTP checks pass across AR/MTP and one/two-session
capacity. Every configuration has identical normalized text, actual function
names/argument values, finish/status and completion counts. All 18 measured
larger-schema replies return exactly BETA with two completion tokens and a
fully cached 902-token prompt. The table retains medians from three warm rounds;
paired rounds contribute two requests each. This is an end-of-turn control,
not a throughput benchmark or a baseline/candidate speedup claim: production
code and the serving executable are unchanged.

| Mode / capacity | Requests | Median decode ms | Median HTTP wall ms |
| --- | ---: | ---: | ---: |
| AR / 1 | 3 | 72.70 | 113.38 |
| MTP / 1 | 3 | 67.50 | 99.88 |
| AR / 2 | 6 | 86.37 | 134.38 |
| MTP / 2 | 6 | 84.72 | 140.95 |

All paired measurements report physical width two. Four of the six paired
MTP replies actually draft (two proposed, one accepted); the last pair uses
the existing controller's AR fallback. Scalar MTP proposes three and accepts
one. The test does not treat configured speculation as proof of executed MTP.

A fresh capture matches all 174 complete logit rows from the
[earlier parser validation](tool-output-parser.md), across schedules `1`, `1`
and `2,4,8`, with 58 positions each. Perplexity remains 63.369153 on every
schedule. The identical 60-token corpus has SHA-256
`5100c8ab6e0ec059b2dfaf7d0d488f1c157f97b33a910b4a711b36ef6cb904d7`.
Whole dump SHA-256 is
`b56bd33c767f74a51ec421b11ef6f6821ab197458b14acdfcbe8d22d1128fb8b`
for each scalar schedule and
`8150e24160ce8b199bc81f14b82cae1a091f63e51258bd4d959bcb4a45d6063d`
for the mixed schedule, exactly matching the retained earlier dumps.

```powershell
& $EXE bench --model $MODEL --speculative mtp --mtp-model $MTP --draft-tokens 7 --logit-eval build/pr393-396-review/corpus.txt --logit-schedules '1:1:2,4,8' --logit-out build/mtp-accounting/current
python tools/bench/logit-eval.py build/pr393-396-review/candidate-s0.bin.sha256 build/mtp-accounting/current-s0.bin.sha256
python tools/bench/logit-eval.py build/pr393-396-review/candidate-s1.bin.sha256 build/mtp-accounting/current-s1.bin.sha256
python tools/bench/logit-eval.py build/pr393-396-review/candidate-s2.bin.sha256 build/mtp-accounting/current-s2.bin.sha256
```

The first GPU fixture passed the EOS matrix but failed its coverage assertion
because the exact-sentence task rejected no proposals. The final test uses a
separate sampled sentence to require real residual correction followed by EOS.
An initial HTTP setup returned `model_not_found`; supplying the harness's
explicit served-model name corrected the fixture before collecting results.

This is bounded consistency coverage on the available quantization, not
independent model quality qualification or proof over all possible EOS states.
Linux execution, other quantizations, long-context sweeps and vision/EOS
interleaving were not run. No upstream EOS implementation was integrated, so
this audit creates no new `UPSTREAM.md` adaptation entry.
