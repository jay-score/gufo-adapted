# PR #390 probe-and-exit adaptation

The text server exits immediately after a failed work unit's probe reports a
hard HIP error. The probe precedes request invalidation, including snapshot
waits and model-state reset. Successful requests do not probe. A scheduler
policy callback supplies the process-exit behavior from `gufo serve`; reusable
schedulers without that callback preserve request-only error handling.

The probe owns a private nonblocking stream and four-byte buffer allocated at
load. It clears the thread's prior error, enqueues a memset, and polls completion
for five seconds. Pending work at the deadline counts as usable. A failed probe
invokes the callback, which writes a bounded diagnostic directly to stderr and
calls `std::_Exit(75)` without model destruction or logging mutexes. Clients
receive a closed connection. The GUI reports a failed engine and supports
manual restart; automatic restart belongs to an external supervisor.

Source: [7d893a020372502757933ec80231927fc9d25a96](https://github.com/gufo-org/gufo/commit/7d893a020372502757933ec80231927fc9d25a96),
based on official main `23cacbb9379d5a8e8531535270891729ca6c00dd`.
Fork base: `c2ea1cf4dab0cf28791e7597a53c1db25217e430`; common ancestor:
`d9a84f13f35d1f98da22886a12eb25dc7062e392`. The wider upstream history was
not integrated. The comparison report is retained locally under
`build/upstream-sync/c2ea1cf4dab0-23cacbb9379d.json`.

## Validation

Windows 11 x64, gfx1151, TheRock 10.0.0, AMD clang 23.0.0git
(`8f497e0992fb7513f7f78a6f6b6f1056c375e961`), MSVC headers/SDK, pinned
manifest dependencies and RelWithDebInfo. Builds and logs use this T3
worktree's `build/`; the existing executable and launcher settings are preserved.

Focused commands (supply existing tool paths through the documented overrides):

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset cpu-test -Target text_generation_scheduler_test -Jobs 4
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset cpu-test -Target http_server_test -Jobs 4
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset cpu-test -Target openai_chat_test -Jobs 4
ctest --test-dir build/cpu-test -R '^(text_generation_scheduler_test|text_generation_device_loss_test|openai_chat_test|http_server_test)$' --output-on-failure --no-tests=error
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Preset gpu-test -Target gufo -Jobs 4
python tools/ci/check-format.py
python tools/ci/check-docs.py
python -m unittest discover -s tests/tools -p 'gui_*_test.py'
```

Scheduler and process-exit tests pass. The process fixture's negative control
exits 76 if invalidation runs; the production callback instead logs the reason
and exits 75. Fake-runner checks cover probe-before-cleanup ordering, healthy
and throwing probes retaining the original failure, recovery, successful work
never probing, classified output-limit errors, and an absent fatal handler.
The fresh GPU build and all four focused CPU tests pass, including Chat and
HTTP fixtures. The GUI suite passes all 34 tests. Repository formatting and
documentation checks pass.

The model baseline is build `9714c7d9caac`, with executable SHA-256
`98155c93a8956c3219508270b4a6b76dd39801810e7a422886acc240af7145a4`.
Its `src`, CMake and dependency manifests match fork base `c2ea1cf` exactly.
The candidate is build `c2ea1cf4dab0-dirty`, with executable SHA-256
`a42feabed64f42e6a8f02d25f453151ed99b9dafd14eaacbbc8dc160ca6863b7`.
Exact commands, executable identities, model
hashes, environment and HTTP outputs are retained locally under
`build/device-loss-validation/`.

Matched artifacts:

| Artifact | SHA-256 |
| --- | --- |
| Flash-Next UD-IQ4_XS shard 1/3 | `5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4` |
| Flash-Next UD-IQ4_XS shard 2/3 | `577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7` |
| Flash-Next UD-IQ4_XS shard 3/3 | `d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833` |
| Shared Q8_0 MTP | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| BF16 projector | `2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355` |

The corpus is the following single UTF-8 line with a trailing newline; its
SHA-256 is `5100c8ab6e0ec059b2dfaf7d0d488f1c157f97b33a910b4a711b36ef6cb904d7`.

```text
A GPU server schedules independent requests and preserves each conversation. The model reads a prompt, predicts the next token, and updates its state. A healthy request completes normally. If the execution device fails, a fresh process can load the model again. Reliable software makes failures visible and keeps successful computations unchanged.
```

Run both executables with the same paths and environment:

```powershell
& $baselineExe bench --model $model --speculative mtp --mtp-model $mtp --draft-tokens 7 --logit-eval build/device-loss-validation/corpus.txt --logit-schedules '1:1:2,4,8' --logit-out build/device-loss-validation/baseline
& $candidateExe bench --model $model --speculative mtp --mtp-model $mtp --draft-tokens 7 --logit-eval build/device-loss-validation/corpus.txt --logit-schedules '1:1:2,4,8' --logit-out build/device-loss-validation/candidate
python tools/bench/logit-eval.py build/device-loss-validation/baseline-s0.bin.sha256 build/device-loss-validation/candidate-s0.bin.sha256
python tools/bench/logit-eval.py build/device-loss-validation/baseline-s1.bin.sha256 build/device-loss-validation/candidate-s1.bin.sha256
python tools/bench/logit-eval.py build/device-loss-validation/baseline-s2.bin.sha256 build/device-loss-validation/candidate-s2.bin.sha256
```

Each schedule records 58 positions with perplexity 63.369153 for both builds.
All 174 complete raw-logit row hashes match exactly; the complete binary dumps
also match byte for byte. Teacher-forced logit checks use the model engine
directly; they do not exercise the serving runner's new probe allocation.

The serving comparison does exercise that allocation with IQ4_XS, Q8_0 MTP,
BF16 projector, context 4096, two sessions, greedy sampling, seed 31, thinking
off, 32-token limits and isolated in-memory caches. Both server commands use:

```powershell
& $exe serve llm --host 127.0.0.1 --port 18090 --model $model --mtp-model $mtp --speculative mtp --draft-tokens 7 --mmproj $projector --context 4096 --sessions 2 --max-tokens 32 --think off --temperature 0 --seed 31 --served-model-name device-loss-check
```

Eight cases cover Chat cold/retry/advancing history, Chat streaming, Responses
buffered/streaming, and two concurrent Chat requests. Both builds pass all eight
with identical text, finish reasons and recorded token counts, excluding
request IDs and timing fields. Model capabilities also match after excluding
the model creation timestamp, including loaded image support. This is bounded
regression evidence, not a performance benchmark or blanket model qualification.

## Limits

Actual Windows driver reset/TDR and Linux execution were not tested. The process
fixture injects device unavailability without resetting the display GPU. The
probe starts only once a generation error reaches the scheduler; cancelled or
expired requests may finish without probing. Idle failures and HIP API calls
that themselves hang need external monitoring. No HTTP status changes, new
wire error code, teardown watchdog, cache format migration, or audio/image/video
failure policy was imported.
