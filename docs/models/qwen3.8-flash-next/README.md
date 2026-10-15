# Qwen3.8 Flash-Next

Hybrid recurrent/QSA mixture-of-experts text/image model on gfx1151.
Supported target: `unsloth/Qwen3.8-Flash-Next-GGUF`, **UD-Q4_K_XL** (four shards).
Optional shared-Q8 MTP predictor; optional BF16 vision projector.
Original unquantized-model and GGUF-conversion parity remain unqualified.

**UD-IQ4_XS** also loads (unqualified): its IQ3_S/IQ4_XS gate/up and IQ4_NL
down experts run on the vendored MMQ/MMVQ kernels, and its Q6_K embedding and
output head are re-encoded to Q8_0 at load (Q8_0's step is about a quarter of
Q6_K's, so the added rounding is small, but it is not bit-identical).
`qwen38_flash_next_iq_experts_ops_test` checks those expert paths against a
double-precision reference.

**Swift-1.5 IQ4_XS** (`ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF`, three shards)
also loads (unqualified): its IQ4_XS/IQ4_NL dense projections (mixers,
SSM/attention, shared experts, PLE) and Q5_K/Q6_K attention are re-encoded to
Q8_0 at upload, its IQ4_XS embedding likewise, and its F16 PLE convolution to
F32. The reference oracle reads the source formats directly.

[Benchmarks](BENCHMARKS.md) · [Quality](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Load and run

```sh
nix develop -c hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 38bb39ee97821de2c9009abb7e93950eec396e66 \
  --include "UD-Q4_K_XL/*" "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" "mmproj-BF16.gguf" \
  --local-dir models/qwen3.8-flash-next
nix build
MODEL=/path/to/first-target-shard.gguf
MTP=/path/to/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
./result/bin/gufo chat --model "$MODEL"
./result/bin/gufo serve llm --model "$MODEL" --speculative mtp \
  --mtp-model "$MTP" --sessions 2 --context 32768
```

The loader discovers the remaining shards. Omit the speculative options for AR;
AR sessions allocate no predictor state even if a shared model has MTP loaded.
Adaptive MTP is default, with `--draft-tokens` capping 1–7 proposals. A request
that is not in a jointly planned batch drafts toward that cap and stops on the
draft head's own confidence (`mtp_policy.hpp`, from upstream PR #603):

- Greedy, one session: stop before the draft at which the product of the drafts'
  top-1 probabilities falls below 0.6; the first draft is always kept. After 15
  cycles, a request keeping fewer than 0.95 drafts per cycle drafts one token.
- Sampled, or greedy with `--sessions` above 1: stop when the next draft's
  acceptance, learned per confidence bucket from the request's committed
  accept/reject prefix, no longer pays for its share of the cycle-cost table.

Jointly planned all-greedy C>1 batches keep a shared width from measured cycle
costs. Every policy reads only committed history, so seeded sampling replays.
Each request keeps private caches, rollback and RNG. See
[MTP qualification](QUALITY.md).

Two opt-in MTP options change only what the draft side proposes; verification
is unchanged, so sampled outputs keep the target distribution and greedy text is
byte-identical. Prompt lookup applies to single-session decoding (measured on
Windows, sampled, thinking on, 3 seeds per prompt, against the length
controller that preceded the confidence stop); the prefix head was measured
against `full` at one session on the confidence stop:

| Option | What it does | Measured |
| --- | --- | --- |
| `--mtp-draft-vocab prefix` | Drafts score the output head's first 98,304 rows in place (no copy; prefix ids are token ids) plus a 33-row copy of the special tokens above them, 40% of the head per draft step. A cycle drafts over the full head when its anchor, or more than one of the last 32 committed tokens, is outside them, so other scripts draft as before. Single and batched heads; `latin` is an accepted alias | Story, code, explain, edit: greedy +10-15% (texts identical, same acceptance), sampled +10-19%; Chinese: same acceptance as `full` (falls back), +1% |
| `--prompt-lookup` | After each kept MTP draft, a match of 12+ tokens in the conversation proposes the tokens that followed it (point-mass proposals, exact under rejection sampling). Shorter matches propose nothing: an ungated 3-token rule lost 6-8% at temperature 1 | File-editing replies +10% sampled; log quoting +23% greedy; prose unchanged. Timings report `lookup_n` / `lookup_n_accepted` |

The official template defaults to thinking on, `xhigh` effort and preserving
prior reasoning. Use the [reasoning controls](../../SERVER.md#reasoning-controls)
for explicit effort/thinking overrides. Native context is 262144; YaRN extension
is unsupported. Memory grows with used context and selected rollback depth;
admission reserves the configured capacity before creating sessions.

## Images

Use this model's `mmproj-BF16.gguf`, discovered beside the target or selected
with `--mmproj`. PNG/JPEG/static WebP CLI and HTTP requests use the
[same image interface](../qwen3.8-27b/README.md#images). Image state participates
in prefill, decoding, verification, multi-turn reuse and disk cache identity.
The predictor embeds shifted text IDs; visual information comes from target
hidden states and mRoPE. Image snapshots require matching prompt attachment.

## Tools and artifacts

Model tests are in `tests/models/qwen38_flash_next`, focused microbenchmarks in
`tools/qwen-flash`. [Quality](QUALITY.md) summarizes qualification and focused checks.
Build a microbenchmark with
`nix develop -c tools/bench/build.sh tools/qwen-flash/projection_plans.hip`;
`dense_blaslt_sweep.hip` times hipBLASLt on the dense prefill shapes.
No historical logit dump is required. New retained result summaries belong in
`artifacts/`; generated traces stay in the ignored top-level `artifacts/` tree.
