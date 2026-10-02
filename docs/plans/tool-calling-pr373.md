# Tool-calling improvements from upstream PR #373

Date: 2026-10-01. Status: parser phase implemented and contract-tested; real-model
qualification is partial. The optional generation-time enforcement phase remains
separate. See the [implementation record](#implementation-record).

## Objective and agreed approach

Improve tool-call parsing, history replay and disabled-tool streaming in the
Windows fork through small adaptations of upstream fixes. Start with parser
changes and deterministic regression tests, then measure real agent behavior.
Decide whether generation-time schema enforcement is necessary after that work.

The user selected this approach to avoid importing the complete upstream #283
and #324 ports just to obtain useful parts of #373. The original plan is retained
below, followed by the actual implementation and validation record.

The first phase should preserve native Qwen/DeepSeek prompts, existing Windows
sampling optimizations, cache behavior and the fork's thinking-boundary fix.
It should primarily change the API parser and tests. Do not introduce a separate
partial JSON Schema validator or heuristic suppression of repeated tool calls.

## Pinned context and evidence

| Item | Revision or source |
| --- | --- |
| Fork used for the initial review | `windows-port`, `7ab8dd9c33d0beeca388ef44263b7c652b1e649d` |
| Fork when this plan was written | `windows-port`, `20846606e884f97a2e985cfed5ebf8ee5f28dfe3` |
| Shared official merge base from the review | `d9a84f13f35d1f98da22886a12eb25dc7062e392` |
| Official merged #373 | [594a623913b4109e4499885e9f73ed4d4ad3698e](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e) |
| Final PR head, with an identical tree to the merge | [3b8dd1691907a4499a17c8ae70a3f256754877d4](https://github.com/gufo-org/gufo/commit/3b8dd1691907a4499a17c8ae70a3f256754877d4) |
| Revision criticized in the linked review | `3424539ae42d5c8683541db3e791d077da8c89b4` |
| Base used by that reviewer | `d707143223c52b91da3f0fb231ba85ac3eab247c` |

The change from the initial review's fork HEAD to the planning HEAD is a
documentation commit. Reconfirm the actual code and HEAD when resuming; these
identities are historical checkpoints, not moving branch guarantees.

Primary references:

- [PR #373: preserve native tool schemas and historical calls](https://github.com/gufo-org/gufo/pull/373).
- [Detailed regression review](https://github.com/gufo-org/gufo/pull/373#issuecomment-5941397880).
- [Initial native-tool and literal-marker change](https://github.com/gufo-org/gufo/commit/5b509f6239d78a3c8305c55eb8cd9a10df362b4e).
- [Multiline edit regression coverage](https://github.com/gufo-org/gufo/commit/78cbefe9c50c6f8fe33153642a5ec916b8ec5b81).
- [Untyped tools and historical-call change](https://github.com/gufo-org/gufo/commit/0b245fd7709504908ca97168c747db682f6c5f07).
- [Final grammar safety tests](https://github.com/gufo-org/gufo/blob/594a623913b4109e4499885e9f73ed4d4ad3698e/tests/core/json_constraint_test.cpp#L1398)
  and [HTTP schema-edge fixtures](https://github.com/gufo-org/gufo/blob/594a623913b4109e4499885e9f73ed4d4ad3698e/tests/functional/tool_agent.py#L405).

The review fetched official history, inspected patches and test source, checked
the live GitHub merge state, and confirmed final-head/merged-tree equality.
A read-only patch check failed because the server files diverge and the fork
lacks `json_constraint.cpp`/`.hpp`. Upstream hosted CPU CI succeeded on the
merge. Local C++ tests, GPU correctness, Pi workloads and timings were not run
for this review. Upstream runtime results are author/reviewer reports, not
Windows qualification of this fork.

## Why the smaller adaptation is appropriate

At the planning revision the fork has no `JsonConstraint`/`TokenConstraint`
integration. Tools are rendered in their native template, then generated calls
are parsed. It does not automatically switch every tool to an injected JSON
protocol when a schema is unsupported.

The current `ParseQwenCalls` uses root `properties` to interpret parameter types.
Unknown parameters default to text. This is a concrete place to improve typed
wildcard handling, independent of a grammar engine. Historical Chat Completions
calls receive the current declaration-name restrictions; Responses history
already accepts broader names. The streaming filter recognizes tool markers
even when no tools are active, which provides another independent adaptation.

| Finding from upstream review | Final upstream disposition | Implication here |
| --- | --- | --- |
| Typed Qwen wildcard values become strings | Ambiguous wildcard calls use JSON fallback | Improve declared-type lookup where unambiguous; native text alone cannot recover arbitrary untyped values |
| Conditional schemas exclude valid branch fields | Best-effort grammar keeps those objects open | That grammar restriction is absent here; do not add a conditional-schema compiler just to reproduce its fix |
| Impossible native grammar reaches generation and causes HTTP 500 | Productivity checking/pruning and non-strict JSON-object fallback | That grammar failure path is absent here |
| Optional `metadata:{}` erases nested requirements | Unconstrained leaf preserves surrounding grammar constraints | This fork has no generation-time requirement enforcement; parser changes do not reproduce that guarantee |
| URI/nested required calls gain extra output framing and latency | Required extended calls keep compact JSON | Preserve the fork's existing framing; do not promise upstream timing gains |

Upstream's final author report qualifies the specific Q4 required-call controls:
81 timing comparisons passed existing margins, with maximum observed total
request slowdown of 0.4%. Earlier automatic Flash URI calls were 7% slower,
and those earlier snapshot timings remained unqualified. These are workload-
specific observations, not a general speedup claim.

The upstream long-context fixture's shared-message-list mutation was fixed.
The Q4 literal-copy failure was shared with the baseline model, rather than
shown to be a PR regression. Use synthetic backend output for exact parser
assertions so a model's failure to copy `Example:` does not obscure the contract.

## Relevant local areas

- [API parser](../../src/cli/serve/openai_chat.cpp): `ParseMessage`,
  `ParseArguments`, `ParseTools`, `ResolveDeclaredTypes`, `ParseQwenCalls`,
  `ParseDsmlCalls`, `ParseGeneration`, `StreamingTextFilter`, and the
  Chat Completions/Responses callers.
- [API regression tests](../../tests/cli/openai_chat_test.cpp) and
  [HTTP contracts](../../tests/cli/http_server_test.cpp).
- [Backend/error contracts](../../src/cli/serve/text_generation_backend.hpp).
- [Existing prompt preparation](../../src/cli/serve/inference_backend.cpp),
  useful for checking that parser work leaves rendered prompts unchanged.
- [Qwen template tests](../../tests/models/qwen/tokenization/chat_template_test.cpp).
- [Server documentation](../SERVER.md), [testing guidance](../TESTING.md),
  [Windows constraints](../WINDOWS.md), and [upstream ledger](../../UPSTREAM.md).

Local commits `6b570d4` and `c8c5b30` implement/document the thinking-boundary
behavior. Preserve it: quoted markers inside tool reasoning must not close
reasoning before `</think>`. Do not replace the entire parser with upstream's
file; adapt the relevant branches to the existing `require_think_end_` logic.

## Implementation sequence

### 0. Re-establish the baseline

1. Read current [AGENTS.md](../../AGENTS.md), this plan and the linked contracts.
2. Check root, branch, HEAD, worktree status and remotes. Use CodeGraph before
   code lookup if `.codegraph/` exists; do not create an index.
3. Work in an isolated branch/worktree using the repository's
   [upstream workflow](../UPSTREAM_SYNC.md). Preserve unrelated edits and the
   running server; do not overwrite its executable or DLLs.
4. Locate existing parser fixtures, and build/run the two API test targets on
   the actual baseline. Retain failing baseline cases before changing code.
5. Recheck whether subsequent local work already implements any item below.

Expected source remotes at planning time: `origin` is `thomas9120/gufo`,
`official` is `gufo-org/gufo`, and `upstream` is `pixmaate/gufo`. Verify URLs.
Pin any newly fetched official tip immediately; use immutable revisions for
provenance instead of relying on `FETCH_HEAD` after another fetch.

### 1. Treat markers as ordinary data when tools are disabled

Add one consistent recognition decision based on tools being declared and
`tool_choice` allowing calls. Apply it to both buffered parsing and streaming
marker/prefix handling, through both API callers. Preserve the current
thinking delimiter and UTF-8 buffering rules.

Acceptance fixtures:

- No tools, and declared tools with `tool_choice:"none"`.
- Qwen and DeepSeek marker literals in visible text and reasoning.
- Marker split across callbacks, marker-prefix suffix, and split UTF-8 bytes.
- Immediate delivery of ordinary content once no active delimiter requires
  buffering; do not wait for end-of-generation merely because it resembles a
  disabled tool marker.
- Enabled-tool parsing still recognizes real calls and hides their markup.
- Existing required-choice, reasoning cutoff and stop behavior stays covered.

Keep this as one focused improvement with the upstream source SHA recorded.

### 2. Preserve historical names consistently

Adapt upstream's historical-call parsing helper rather than weakening
`RenderableToolName` for declarations. Reuse it for Chat Completions assistant
history and Responses `function_call` history.

Acceptance fixtures:

- Past names absent from today's tools, Unicode names and names rejected by
  current declaration rules are preserved as history.
- Empty/non-string names, embedded NUL and invalid/non-object JSON arguments
  are rejected before generation.
- `call_id`/tool-result identity, image ordering and assistant-item grouping in
  stateless Responses replay are retained.
- New calls remain limited to today's declared functions; declaration-name
  validation stays intact.
- Cover delimiter-bearing historical names against the actual renderers. The
  history relaxation must not be mistaken for authorization to invoke them.

Responses already permits broad historical names, so its main changes may be
consistent validation and helper reuse. Include template checks if rendering
or historical framing needs any adaptation.

### 3. Resolve declared Qwen argument types without guessing

Start from a fixture using this non-strict declaration:

```json
{
  "type": "object",
  "properties": {"value": {"type": "string"}},
  "required": ["value"],
  "patternProperties": {"^x_": {"type": "integer"}}
}
```

Given a native `x_count` parameter containing `1`, the current parser treats it
as the string `"1"`; the declared type should allow recovery of integer `1`.
Keep a named string parameter containing `1` as a string.

Implement bounded schema lookup for named properties and local references,
then typed `patternProperties` and schema-valued `additionalProperties` as
needed for retained fixtures. A named property can also match patterns; all
applicable type constraints must be considered. Additional properties apply
only when neither a named property nor a pattern matches. Multiple matching
patterns are not a first-match choice.

Keep lookup results explicit about unsupported, cyclic or ambiguous schemas.
Preserve existing non-strict guidance semantics when type information cannot
be resolved safely. Do not infer arbitrary types from the spelling of a value,
and do not claim support for all JSON Schema applicators.

If pattern matching is needed, inspect the existing ICU dependency before
adding an engine. Use JSON Schema-compatible matching and bounded execution;
do not substitute an incompatible regex dialect silently. CMake currently
finds ICU `uc` and `i18n`, but `gufo_core` links only `ICU::uc`. Add `ICU::i18n`
only if this implementation needs it and verify the pinned Windows libraries.

Acceptance fixtures should cover integer, number, boolean, null, array and
object types; declared strings that look like JSON; escaped/Unicode keys;
multiple patterns; local reference chains and cycles; and additional-property
precedence. Existing Python-literal compatibility, duplicate-parameter rules
and whitespace preservation must continue to pass.

String/non-string unions and genuinely untyped wildcard values remain a
documented limitation: identical native Qwen text may represent different
JSON values. Do not introduce automatic JSON prompt switching in this phase.

### 4. Retain multiline edits and diagnose malformed calls

Adapt the upstream multiline regression into local fixtures before changing
parsing. Valid escaped newlines inside an array/object must reach the client
unchanged, including indentation and literal protocol text in edit strings.
Raw unescaped controls in JSON remain invalid; do not repair them with a
blanket replacement or strip meaningful whitespace.

Separate complete malformed tool-shaped output from output interrupted by a
stop, token limit or cancellation. The latter must terminate normally without
emitting an incomplete call. Ordinary prose mentioning tool syntax is also
distinct from an attempted call.

Reproduce the current malformed-call behavior and select an explicit error
contract using the existing backend/API error machinery. Check buffered HTTP
errors and SSE errors after headers have been sent. A complete recognized
malformed attempt should not silently look like successful task completion.
If robust classification requires prompt/sampler changes or breaks ordinary
prose, retain the regression and defer that diagnostic change separately.

This step improves parsing and observability. It does not enforce nested
required fields during generation or guarantee that agent loops disappear.

## Validation and completion criteria

For each improvement, add the smallest deterministic fixtures that reproduce
the defect, build the affected targets freshly, and run the relevant checks.
Exercise both APIs and buffered/streaming delivery. Include `http_server_test`
when error/status contracts change, and Qwen template tests when history or
rendering behavior changes.

Examples from the current Windows workflow, run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target openai_chat_test -Jobs 4
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target http_server_test -Jobs 4
ctest --test-dir build/gpu-test -R '^(openai_chat_test|http_server_test)$' --output-on-failure --no-tests=error
python tools/ci/check-format.py
python tools/ci/check-docs.py
```

Use the repository-compatible formatter. Pass tool-location overrides when
installations differ. For bounded Windows CPU checks, follow
[CI](../../.github/workflows/ci.yml); Linux shared-code CI is also required.
The commands above illustrate the workflow; actual runs are recorded below.

After parser contracts pass, retain a bounded real-model agent workload:

- Qwen27B AR and DFlash2, and Flash-Next AR and MTP when artifacts are available.
- A nested/multiline edit, tool-result replay, disabled marker literals, and
  typed wildcard calls through both APIs.
- A read/edit/verify/finish sequence with independently checked file contents,
  capped turns and recorded repeated actions without progress.
- Fresh, cached and replayed requests; an interruption/resume case if affected.
- Baseline/candidate prompt tokens and unchanged-template controls, plus total
  request latency and completion counts. Preserve identical workloads and
  cache histories; a throughput benchmark alone is insufficient.

Record executable/source identities, toolchain, model hashes/revisions,
quantization, sidecars, context, sampling, speculation, requests and responses.
Use isolated client settings/cache. Keep large logs under ignored `build/`;
retain concise results and exact reproduction commands in the implementation
record. Missing artifacts and skipped checks remain coverage gaps.

For changes affecting inference correctness, follow
[matched-token validation](../TESTING.md#matched-token-and-layer-comparisons):
retain matched full-logit and perplexity checks, requiring exact full-logit
equality when arithmetic is unchanged. Parser fixtures remain necessary even
when logits match. If work reaches sampling/speculation, also cover rejection,
rollback, residual draws and snapshot interleaving.

Finish each improvement with a focused Conventional Commit, immutable
`Upstream-Commit:` provenance for manual adaptations, and an accurate
[UPSTREAM.md](../../UPSTREAM.md) entry documenting imports, omissions and checks
actually run. This planning document itself is not an integrated change and
does not warrant a ledger entry. Do not push unless requested.

## Optional later phase: generation-time enforcement

Revisit this only if measured agent failures require prevention during
generation, or the user requests strict structured-output behavior. Parser
validation can reject a completed bad call; it cannot prevent the model from
generating one or recover information absent from the native representation.

The efficient full integration would adapt final versions of the required
constraint components directly, rather than replay every overlapping PR:

- [#283 foundation](https://github.com/gufo-org/gufo/commit/982bffea86fd5568759a420c4808c5b2123161c8):
  schema grammar/lexeme/regex code, response-format contracts, vocabulary masks,
  sampler state and required model connections.
- [#324 native-tool support](https://github.com/gufo-org/gufo/commit/b26de0d30caa363cc6694bddd094af9bd88ec62b):
  native tool grammars and constraint-aware AR/speculative execution.
- [Final #373](https://github.com/gufo-org/gufo/commit/594a623913b4109e4499885e9f73ed4d4ad3698e):
  open nested schemas, faithful fallback routing, finite grammars, required-call
  compact routing, and relevant regression coverage.

This remains a substantive inference change. Preserve Windows fast-sampling
paths, device-memory handling, cache adaptations and per-request controls.
Keep unrelated sampler-default migrations, metrics/progress features and GUI
changes outside the port. Do not replace the upstream grammar with a homemade
subset validator or count parser-only work as full strict-schema support.

## Implementation record

### Revisions and scope

Implemented on `t3code/implement-tool-calling-pr373` in the isolated T3 worktree,
starting from clean `c4dcca536af5dc73998f24b28e548985e7f91efc`. Official history
was pinned at `594a623913b4109e4499885e9f73ed4d4ad3698e`; the sync report is
`build/upstream-sync/c4dcca536af5-594a623913b4.json`. No index was created, GUI
settings were not changed, and the existing release executable was preserved.

| Improvement | Local implementation commit |
| --- | --- |
| Disabled-tool markers and immediate prefix delivery | `f997a39f14da75719a407ee99931736cdf1b4a4b` |
| Shared historical-function validation | `5a95703c90afac4a448047e28044ed64a4d3bb2c` |
| Bounded declared-type recovery using existing ICU | `5935c53e99b451a1cd26a37c7cf47aa5391d036b` |
| Multiline/literal boundaries and malformed-call diagnostics | `55ff2aff89dfa6c01a45a1b15c13f55fa09b9b8c` |

Each adaptation carries the pinned `Upstream-Commit:` trailer and an
[upstream ledger entry](../../UPSTREAM.md). Prompt-renderer implementation,
sampling, cache and model arithmetic were not changed. Historical renderer
tests verify preserved names. Typed lookup supplies conservative parser hints,
not complete JSON Schema validation. Unsupported, cyclic, bounded or ambiguous
hints retain text. See the precise limits in [server contracts](../SERVER.md).

Complete framed malformed attempts at normal EOS now produce a non-retryable
`malformed_tool_call` error. Recognized but unclosed output remains ambiguous;
it is not repaired or classified as a complete malformed frame. A required call
at EOS still fails if no valid call survives. Stop, limit and cancellation
interruptions omit incomplete calls normally, including required choice.

### Builds and deterministic checks

Windows 11 x64 / Strix Halo `gfx1151`: TheRock ROCm 10.0.0, Clang 23.0.0,
MSVC headers/libraries 14.51.36231, Windows SDK 10.0.26100.0, Ninja 1.12.0,
CMake 3.29.2 for configuration, pinned vcpkg `x64-windows` (ICU 78.3#2).
Formatting used clang-format 21.1.8. Python was 3.14.6. The candidate release
build linked/staged `icuin78.dll` successfully.

Baseline API targets were freshly built at `c4dcca536af5` before edits and
passed (`openai_chat_test` 0.17 s, `http_server_test` 1.95 s). New disabled-prefix,
historical-name, wildcard-type, multiline and cross-protocol literal fixtures
failed before their fixes; retained logs live under `build/`.

Final CPU contracts were freshly built from the candidate source and passed:

| Check | Result |
| --- | --- |
| `openai_chat_test` | Pass, 0.13 s |
| `http_server_test` | Pass, 1.97 s |
| `qwen_chat_template_test` | Pass, 0.05 s |
| `ds4.template` | Pass, 0.05 s |
| Repository formatting | Pass, 483 C++ files |
| Documentation links | Pass |
| Dependency contracts | Pass, 36 dependencies |
| New HTTP harness | Python compilation/help pass; actual matrix failures retained below |

The bounded CPU setup follows [Windows CI](../../.github/workflows/ci.yml).
With the MSVC developer environment imported and the local `$toolParserRocm`,
`$toolParserVcpkg` and `$toolParserNinja` installations selected, the relevant
commands are:

```powershell
cmake --preset cpu-test "-DCMAKE_C_COMPILER=$toolParserRocm/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=$toolParserRocm/lib/llvm/bin/clang++.exe" "-DCMAKE_TOOLCHAIN_FILE=$toolParserVcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows "-DVCPKG_INSTALLED_DIR=$pwd/build/vcpkg_installed"
cmake --build --preset cpu-test --parallel 4 --target openai_chat_test http_server_test qwen_chat_template_test ds4_chat_template_test
ctest --test-dir build/cpu-test -R '^(openai_chat_test|http_server_test|qwen_chat_template_test|ds4\.template)$' --output-on-failure --no-tests=error
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Target gufo -Jobs 4 -Rocm $toolParserRocm -Vcpkg $toolParserVcpkg -Ninja $toolParserNinja
python tools/ci/check-format.py
python tools/ci/check-docs.py
python tools/ci/check-dependencies.py
```

CPU runtime DLLs were staged from the pinned manifest install. Actual commands,
build output and CTest results are retained in `build/baseline*.log`,
`build/final-cpu-check.log`, `build/history-ds4-check.log` and
`build/candidate-final-build.log`; `build/check.ps1` imports the compiler
environment for the incremental CPU checks.

### Real-model matrix

The reusable [HTTP harness](../../tests/tools/tool_parser_agent_check.py) sends
11 direct probes per API, then a capped six-turn agent sequence. Both APIs are
stateless, with exact request/response bodies retained. Each configuration was
run against a dedicated baseline server and then a dedicated candidate server,
with fresh RAM cache and no disk cache or client cache. All owned servers were
stopped after their run.

The baseline release was `f444916ce5c07c3028915f98d7e51f1c3875bdd8`;
`git diff --quiet f444916ce5c0 c4dcca536af5 -- src CMakeLists.txt vcpkg.json`
confirmed the pre-change source/dependency inputs were identical. Both release
build caches identify the same compiler and dependency toolchain. Executable
SHA-256 values:

- Baseline `gufo version f444916ce5c0`:
  `73a9768a1f496b58bddc3691ff24ec0b81070c161315125372c8c3ba396c8398`.
- Candidate `gufo version 55ff2aff89df`:
  `2636e7eadea7f50527c6b9e33915624eea569655ca23276b89b69ea58dbb6e2a`.

Matched settings: context 4096, max output 256, temperature 0, seed 31,
thinking off, one session, port 18080, served model `tool-parser-test`. AR uses
`--speculative off`; Qwen DFlash2 uses its Q8_0 sidecar, draft width 7 and
adaptive policy; Flash MTP uses its shared Q8_0 sidecar, draft width 7, length
policy and full draft vocabulary. The same adjacent BF16 projectors were
auto-loaded in each build. No image inference was exercised.

| Mode | Direct probes baseline → candidate | Exact-file agent goals baseline → candidate | Full workload latency baseline → candidate |
| --- | --- | --- | --- |
| Qwen27B UD-Q4_K_XL AR | 9/22 → 22/22 | 0/2 → 0/2 | 68,708 → 76,284 ms |
| Qwen27B UD-Q4_K_XL DFlash2 | 9/22 → 22/22 | 0/2 → 0/2 | 32,755 → 37,086 ms |
| Flash-Next IQ4_XS AR | 9/22 → 18/22 | 0/2 → 0/2 | 34,342 → 38,265 ms |
| Flash-Next IQ4_XS MTP | 9/22 → 18/22 | 0/2 → 0/2 | 21,435 → 25,014 ms |

Candidate direct probes total 80/88, versus baseline 36/88. Qwen passes all
direct cases. Flash's remaining four failures per mode are the exact multiline
probe, fresh/repeated through both APIs (`tool_choice_unsatisfied`, HTTP 502).
The baseline also lost integer wildcard types, rejected historical Chat names,
failed multiline calls and returned 502 for a one-token required interruption.

All eight candidate agent sequences execute `read_fixture`, `edit_fixture`,
`read_fixture`, `finish`, with zero repeated actions without progress. The
baseline executes only the first read: the subsequent multiline edit is not a
structured call. Nevertheless, **all independently checked exact-file goals
fail in both builds**. Candidate models replace only `before`, leaving old
indentation/blank lines around a replacement that already contains whitespace.
The resulting file is `"    after\nExample: <tool_call> </parameter> </tool_call>\n\n\n"`,
instead of the requested `"  after\nExample: <tool_call> </parameter> </tool_call>\n"`.
The parser preserves the model's edit exactly; the harness does not repair it
or suppress repeated actions. All eight harness invocations therefore exit 1.

Full workload latency is not a matched completion comparison: candidate agent
runs make four requests per API versus two on baseline, increasing total
requests from 26 to 30 per mode. The 15 direct probes returning HTTP 200 in
both builds provide matched controls. All 60 pairs have identical request
bodies, prompt/completion token counts and cached-token counts:

| Mode | Sum of 15 matched request latencies baseline → candidate | Median per-request delta |
| --- | --- | --- |
| Qwen AR | 32,887 → 32,999 ms | +0.5% |
| Qwen DFlash2 | 14,827 → 14,818 ms | +0.4% |
| Flash AR | 19,199 → 18,851 ms | -0.8% |
| Flash MTP | 11,506 → 11,453 ms | -0.2% |

These are single observations, not qualified speedups or regression margins.
No build ran concurrently with measurement. Complete token/cache/timing records
and exact machine-local server commands are in
`build/tool-parser-runtime/matrix.json`, with per-mode `requests.json`,
`summary.json`, `checks.log`, `server.log` and fixture files. The ignored runner
is `build/run-tool-models.py`. To reproduce each mode, set `$toolParserExe` and
`$toolParserModel` to the selected executable/model paths. Start the isolated
server with the common command below; use the mode's exact suffix for
`$toolParserModeArgs`:

| Mode | Server suffix |
| --- | --- |
| Both AR modes | `--speculative off` |
| Qwen DFlash2 | `--speculative dflash2 --dflash-model <sidecar> --draft-tokens 7 --draft-policy adaptive` |
| Flash MTP | `--speculative mtp --mtp-model <sidecar> --draft-tokens 7 --mtp-policy length --mtp-draft-vocab full` |

```powershell
$toolParserModeArgs = @('--speculative', 'off')
& $toolParserExe serve llm --port 18080 --model $toolParserModel --served-model-name tool-parser-test --context 4096 --max-tokens 256 --temperature 0 --seed 31 --think off --sessions 1 @toolParserModeArgs
```

In another shell, run the harness and retain its nonzero result when any probe
or independently verified file goal fails:

```powershell
python tests/tools/tool_parser_agent_check.py --url http://127.0.0.1:18080 --out build/tool-parser-runtime/reproduction
```

### Artifact identities

All GGUF shards, sidecars and projectors were fully hashed. Absolute local paths
and byte counts are retained in `build/tool-parser-runtime/artifacts.json`.

| Artifact | SHA-256 |
| --- | --- |
| Qwen3.8-27B-UD-Q4_K_XL.gguf | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Qwen27B mmproj-BF16.gguf | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| Qwen3.8-27B-DFlash2-Q8_0.gguf | `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |
| Flash-Next UD-IQ4_XS shard 1/3 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| Flash-Next UD-IQ4_XS shard 2/3 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| Flash-Next UD-IQ4_XS shard 3/3 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |
| Flash-Next mmproj-BF16.gguf | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` |
| mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |

### Remaining failures and optional-phase decision

The Flash AR multiline failure was isolated through `/v1/completions`, using
the actual Qwen renderer on the retained required-call request. Baseline and
candidate raw requests and generated text are identical (375 prompt tokens,
37 output tokens, normal EOS). Both stop inside the JSON string after
`"new_text": "  after\nExample: `, before any literal tool marker or closing
JSON/frame. The parser cannot recover text that was not generated. Complete
synthetic versions pass the deterministic parser fixtures. Both raw captures
are retained separately in `build/tool-parser-runtime/raw-flash-baseline/`
and `build/tool-parser-runtime/raw-flash/`, including exact prompt/response
and server logs. This diagnoses the matched AR case; raw MTP and Responses
variants were not separately captured.

Generation-time constraints warrant a separate investigation for the premature
EOS case. They were not imported in this parser phase. A schema-valid call can
still contain a semantically wrong edit, as the agent file checks demonstrate;
strict structure alone does not establish task success. Do not promote this
record to complete structured-output or agent-quality qualification.

The subsequent [source investigation](tool-calling-followup.md) found that the
reference-derived Qwen renderer trimmed whitespace from tool results before
generation. The original agent failures therefore cannot be attributed solely
to model copying. That follow-up preserves tool-result bytes and separately
records remaining semantic errors, prompt controls and the Flash EOS trace.

Linux shared-code CI remains unrun: the available WSL installation lacks the
build toolchain, and no push was requested. Full matched-token logit/perplexity,
image inference and DeepSeek real-model checks were not run. No model arithmetic,
loading, sampler, speculative state or prompt-renderer implementation changed;
the GPU runs establish only the specific API/model behavior recorded above.

## Completion checklist

- [x] Reconfirm HEAD, clean/owned paths, worktree and current AGENTS instructions.
- [x] Verify which planned improvements are already present.
- [x] Capture baseline API fixture results and identify the test executable.
- [x] Implement/validate disabled-tool marker handling.
- [x] Implement/validate historical-call name handling.
- [x] Implement/validate bounded declared-type resolution.
- [x] Add multiline fixtures and resolve or explicitly defer error diagnostics.
- [x] Retain bounded agent and matched latency results; document coverage gaps.
- [x] Update provenance, server contracts and actual validation records.
- [x] Decide separately whether the optional inference phase is warranted.
