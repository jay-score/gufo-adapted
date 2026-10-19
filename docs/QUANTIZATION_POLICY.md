# Quantization qualification policy

Policy version 2, adopted October 19, 2026 for future campaigns. Runner support
is fixture-tested; a real version 2 campaign has not yet been validated.

This policy governs workflow [#3](AGENT_WORKFLOWS.md#3-qualify-a-quantization-variant).
It separates evidence of a runtime bug, evidence of quality loss and evidence
that merits investigation. Distribution distance from BF16 is useful evidence,
but a large distance at one history does not by itself establish a worse answer
or a defective quantization.

## Decisions and gates

| Evidence | Treatment |
| --- | --- |
| Invalid artifacts, unsupported routes, loading errors, nonfinite outputs, failed operator/state/replay checks or capability requirements | Hard failure; fix or explicitly narrow the supported scope |
| Missing required artifacts, skipped checks, unmatched histories/settings or absent required comparator | No pass; retain the failure or unverified status and the gap |
| Held-out matched-token likelihood exceeds its predeclared budget, or a required functional task fails | Quality failure for the declared scope |
| Mean KL, mean TV or maximum row TV exceeds a predeclared investigation trigger | Visible distribution alert; does not automatically fail quantization quality |
| Same-weight engine comparison exceeds its separately declared execution budget | Execution gate failure for that coverage; investigate without attributing the bug to either engine prematurely |
| Changed-build comparison with identical weights | Workflow #2's regression contract, including exact logits when arithmetic should be unchanged |

An overall runner `PASS` requires all hard gates and required evidence. Alerts
remain in both JSON and Markdown, including on a pass. Report support, quality,
independent execution and coverage separately when explaining the outcome.
A pass on short prefixes cannot establish cross-engine parity on corpus tails.
Unresolved differences outside declared coverage remain explicit limitations;
they cannot be hidden by an aggregate score or promoted to a broader pass.

KL/TV describe distribution fidelity. NLL/perplexity score observed next tokens;
functional fixtures score the specified behavior. None alone establishes broad
chat, coding, reasoning or long-context quality. A task that actually requires
a worst-case fidelity bound needs a separate, justified contract; the generic
version 2 quantization gate does not impose one.

## Predeclare a new campaign

1. Pin the candidate, teacher, independent engine, corpus, build and runtime
   identities. State each reference's purpose and precision limits. Gufo BF16
   measures quantization plus native arithmetic; it does not prove conversion
   or original-checkpoint accuracy.
2. Choose the claimed use and required functional tasks before measuring the
   candidate. Pin task inputs, expected behavior, scoring and limits. Use fresh
   held-out corpus windows and task cases for a new quality verdict; previously
   examined outliers are diagnostic regression cases. Keep calibration and
   acceptance samples separate and record their selection method and hashes.
3. Justify a model/use-specific likelihood budget independently of the desired
   candidate outcome. A higher-quality local quant is a useful control, not an
   automatic acceptance oracle. Do not select a new limit to admit a known run.
4. Predeclare distribution investigation triggers and report their frequency,
   magnitude and task impact. Keep the maximum, nearest-rank 50/90/95/99th TV
   percentiles and counts above the declared maximum-TV trigger. Preserve
   complete-vocabulary rows and exact histories for selected diagnostics.
5. Declare same-weight execution limits separately, with batching, activation,
   KV and recurrent-state precision recorded. Reference engines can use
   different approximate arithmetic; matching one is not proof of correctness.

The existing native runner covers short/prefill fixtures, one 512-label corpus
window, operator/state tests and three fixed capability smoke cases. Those
known smoke cases remain required but cannot stand in for fresh held-out
functional evaluation. Retain any additional model-owned task results alongside
the run and limit claims to the coverage actually exercised. Extend the model's
capture/fixture contract before claiming broader automated coverage.

## Declaration and runner versions

`tools/models/qualify-variant.py run --declaration CAMPAIGN.json` accepts both
`gufo-quant-variant/1` and `gufo-quant-variant/2`. The version 1 pilot remains the
default for reproducibility; the runner never silently migrates its declaration.

Version 2 retains the artifact, capture, capability, checks and `llama_limits`
fields. Replace `limits` with these fields:

| Field | Contract |
| --- | --- |
| `quality_limits` | Required finite nonnegative `perplexity_ratio` maximum, applied independently to short, prefill and corpus captures |
| `investigation_triggers` | Required finite nonnegative maxima for `mean_kl_nats`, `mean_tv`, `maximum_row_tv`; exceedances append to `alerts`, not `errors` |
| `policy_rationale` | Nonempty explanation of the selected likelihood budget and diagnostic triggers |
| `heldout_evaluation_plan` | Nonempty description of fresh inputs, selection/hashes, functional scoring, calibration separation and scope limitations |

`llama_limits` remains a hard execution contract with all four version 1
metrics. Functional smoke checks, state checks and malformed/nonfinite capture
refusals remain hard gates. The runner validates the declaration's structure
and pinned corpus identity; an agent must also verify the stated held-out design
and any supplemental functional evidence. A declaration string alone is not
proof that an evaluation is held out. No ready-to-run version 2 Q5 declaration
is supplied: its fresh acceptance inputs and justified budgets still need to
be chosen before a new campaign.

## Historical records and current Q5 status

The original Qwen27B [version 1 declaration](models/qwen3.8-27b/artifacts/variant-qualify.json)
and [failed pilot record](models/qwen3.8-27b/artifacts/qualification-q5-k-m-20261019.json)
remain unchanged. Its SSM failure has a separately validated correction;
the original maximum-TV decision remains a failure under its original policy.
The [follow-up](models/qwen3.8-27b/QUALITY.md#q5-follow-up-corpus-tails-and-reference-execution)
supports revising future policy, not relabeling that run or declaring Q5 defective.

Q5 has no new version 2 qualification. A fresh held-out quality campaign and
any expanded engine-execution claim need their own evidence. The remaining
cross-engine work is [planned separately](plans/qwen27b-cross-engine-differences.md)
and is deferred. No new GPU work or performance qualification accompanies this
policy change.
