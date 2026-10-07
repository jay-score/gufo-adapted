# PR #473 disk staging qualification

October 7, 2026. Adapted official
[PR #473](https://github.com/gufo-org/gufo/pull/473) on fork base
`2083310ea97cfe1ed4a9d52f8684d44ca778ba36`, in branch
`fix/disk-staging-auto-pr473`. The implementation commit introduces this record.
Source provenance and deliberate omissions are in [UPSTREAM.md](../../UPSTREAM.md).

## Change and admission regression

Automatic staging now resolves to `min(options.capacity_bytes,
HostSnapshotBudgetBytes()/4)`. The helper leaves half the available host RAM,
so staging receives one eighth, sampled after model/session loading and bounded
by the disk budget. Nothing allocates the whole staging budget upfront.
Explicit limits, queue/read bounds, RAM checkpoints and model arithmetic retain
their behavior. No snapshot-sharing or `min_step` change is included.

The new disk-store fixture reserves a logical `1 GiB + 1 byte` payload through
`CanSave`/`ReserveCapture`, without creating a snapshot or allocating its payload.
It checks that pending captures consume the shared limit and release restores
admission. The branch runs when available host RAM exceeds about 8 GiB; otherwise
it prints a skip. It executed on this host. Temporarily restoring the old capped
calculation and rebuilding produced exit 1 at
`automatic staging admits snapshots larger than 1 GiB`. After restoring the fix
and rebuilding the disk-store object, the same fixture passed. Existing explicit
limit, in-flight capture, disk capacity and smaller-limit restart fixtures remain.

## Builds and focused checks

Windows 11 x64, Strix Halo gfx1151. TheRock ROCm 10.0.0, AMD clang 23.0.0git
(`8f497e0992fb7513f7f78a6f6b6f1056c375e961`), VS Build Tools/MSVC
14.51.36231 headers and libraries, Ninja, CMake, and the fork's pinned vcpkg
manifest. Dependencies were copied from the existing pinned install into this
worktree's ignored build directory; the build made no package upgrades.
The main production build and its runtime files were preserved.

Run from the isolated worktree (machine-specific overrides are qualification
inputs, not repository defaults):

```powershell
$BuildArgs = @('-Jobs', '4', '-Rocm', 'C:/TheRock/build',
  '-Vcpkg', 'C:/Users/pegas/Documents/Projects-2026/Strix-halo-optimizations/vcpkg',
  '-Ninja', 'C:/Strawberry/c/bin/ninja.exe')
./tools/windows/build.ps1 -Preset cpu-test -Target continuation_disk_store_test @BuildArgs
./tools/windows/build.ps1 -Preset cpu-test -Target text_model_runner_test @BuildArgs
./tools/windows/build.ps1 -Preset cpu-test -Target gufo @BuildArgs
./tools/windows/build.ps1 -Preset gpu-test -Target qwen27b_target_test @BuildArgs
./tools/windows/build.ps1 -Preset release -Target gufo @BuildArgs
ctest --test-dir build/cpu-test -R '^(continuation_disk_store_test|text_model_runner_test|serve_cli_test)$' --output-on-failure --no-tests=error
python tests/cli/serve_test.py "$PWD/build/release/gufo.exe"
python tools/ci/check-format.py
python tools/ci/check-docs.py
C:/P2027/gufo/build/gui-env/Scripts/python.exe -m unittest discover -s tests/tools -p 'gui_*_test.py'
git diff --check
```

All three fresh CPU CTest targets pass (1.91 seconds total). Release CLI checks
pass. GUI: 34 tests pass. clang-format **21.1.8** checks all 485 C++ files;
documentation and whitespace checks pass. Formatter executable came from
`C:/P2027/gufo/build/format-tools/clang_format/data/bin`. The disk-store test's
Windows root-symlink fixture skips because this session lacks creation privileges.
Initial build invocation with PowerShell stream redirection stopped on an existing
compiler warning; running the build script as an external PowerShell process
completed. Existing compiler warnings were not changed by this fix.

## Artifact and numerical identities

SHA256 values below identify the tested artifacts. Candidate production reports
`2083310ea97c-dirty`; its source is the implementation introducing this record.

| Artifact | SHA256 |
| --- | --- |
| Baseline `C:/P2027/gufo/build/release/gufo.exe` (`2083310ea97c`) | `27e59fccfe2e8f6088b32847d9ec8a2ce8fd1c9d56d55a98f2cc0bca3101308f` |
| Isolated candidate `build/release/gufo.exe` | `956c6087a3f81047b30399d8ba1eef8589a74bf3b6de505b3447a95cca4796ab` |
| `Qwen3.8-27B-UD-Q4_K_XL.gguf`, 17,559,178,144 bytes | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` |
| Compatible `mmproj-BF16.gguf`, 931,146,432 bytes | `83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53` |
| `tests/models/qwen27b/target_test.cpp` (fixed numerical corpus) | `90e1a42e27dbccc041cb7a9761578d53a8b9f10170a6c95a4d750372775f1fe9` |
| Fresh `build/gpu-test/qwen27b_target_test.exe` | `bd933c8d9b033b809c8c07a342afc0d3fb590e0d8bfda0eaafb2027df99a24fc` |
| Baseline and candidate full-logit files | `4a4972a1669fea578b24575b4b7f8a05bdc38b8ce9a0169ac5d82441ecd0da88` |

Model directory:
`C:/Users/pegas/Documents/Llama/LLama-GUI/models/unsloth_Qwen3.8-27B-GGUF`.
HTTP is AR text only, with speculative and thinking off; no draft is loaded.
The adjacent projector is available, but these requests contain no images.

```powershell
$Model = 'C:/Users/pegas/Documents/Llama/LLama-GUI/models/unsloth_Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf'
./build/gpu-test/qwen27b_target_test.exe $Model --capture-logits build/pr473/baseline-logits.bin
# Reconfigure/rebuild the same target for the candidate, then:
./build/gpu-test/qwen27b_target_test.exe $Model --capture-logits build/pr473/candidate-logits.bin
Get-FileHash build/pr473/baseline-logits.bin,build/pr473/candidate-logits.bin
```

Three predetermined prose/code token histories match. All 27 full FP32 rows
(248,320 logits each, 26,818,560 bytes total) are byte-identical; all printed row
hashes match. On the same 24 continuation labels, mean NLL is
**1.305064282821349** and perplexity is **3.6879261563836652** for both.
The target was initially built before the policy edit; candidate reconfiguration
found no model-target work because staging is outside that target and model code
is unchanged. Its executable hash is therefore the same. This check records the
unchanged arithmetic fixture; it does **not** validate disk serialization or
restore. The production HTTP checks below exercise those paths separately.

## Real checkpoint and matched-history restart

Local harness, logs, request/response JSON and raw logits are retained under
ignored `build/pr473/`. The harness selects the advertised `/v1/models` ID,
submits `/v1/chat/completions`, and starts each server in an owned hidden console.
Shutdown sends CTRL_C only to that console: every successful run logs
`shutdown_requested signal=2` and exits 0. No user server or GUI settings changed.

Exact server settings for each process:

```powershell
$Exe serve llm --model $Model --context 32768 --sessions 1 --speculative off --think off --prefill-chunk 512 --host 127.0.0.1 --port 5873 --cache-disk $Cache --cache-disk-bytes 8589934592 --cache-disk-staging-bytes $Staging
```

Request construction (UTF-8 JSON): `model` is the advertised ID,
`max_tokens=8`, `temperature=0`, `top_k=0`, `top_p=1`, `min_p=0`, `seed=473`.
One user message contains exactly:
`"The secret word is amber. Archive follows:" + " archive" * 14500 +
"\nReply with only the secret word."`.
This renders 14,529 prompt tokens. Actual cold serial-C1 prefill uses six chunks,
maximum 4,096 tokens; the scheduler's effective chunking differs from its
configured 512-token interleaving setting.

The live follow-up appends the returned assistant message verbatim, then a user
message `Repeat the secret word, with no other words.`. The identical JSON
history is replayed in a separate new process. All answers are `amber` with
finish reason `stop`, one generated token; content SHA256 is
`b1601f694b9d336c35fc456de5697dfde5e1b1ce4e8c40766fb6cb763aba91c7`.
Full `choices` objects match between baseline, candidate cold, restart retry,
and explicit-small fallback, and between live and restarted follow-up histories.

| Run | Staging bytes | Cached / prefill tokens | TTFT ms | Result |
| --- | ---: | ---: | ---: | --- |
| Baseline auto, empty isolated cache | 1,073,741,824 | 0 / 14,529 | 23,338.3 | Both large captures rejected; no files |
| Candidate auto, empty isolated cache | 1,870,011,904 | 0 / 14,529 | 23,113.9 | Queued and stored 1,111,625,411-byte checkpoint |
| Candidate live follow-up | same process | 14,530 / 24 | 684.7 | Memory hit; same answer |
| Candidate new process, same initial request/cache | 1,790,898,688 | 14,522 / 7 | 1,683.1 | Disk hit; same cold answer |
| Candidate another new process, matched follow-up history | 1,839,769,088 | 14,529 / 25 | 2,265.9 | Disk hit; same live continuation |
| Candidate explicit 1 GiB, empty isolated cache | 1,073,741,824 | 0 / 14,529 | 22,991.9 | Both captures rejected; correct fallback; no files |

Cold request wall times were 23.54 seconds baseline and 23.30 seconds candidate;
restart retry was 1.87 seconds and explicit-small fallback 23.15 seconds.
These single samples establish operation, not a general performance benchmark.
Restart restore times were 1,384.5 ms (retry) and 1,511.4 ms (follow-up).

For candidate cold, the first queued checkpoint occupied 1,111,625,411 bytes;
the second capture required 1,112,084,191 and was refused with
`reason=staging_capacity` while that first write was pending. Shutdown was logged
at 15:57:33; the accepted write completed at 15:57:34 (`write_ms=2420.59`),
demonstrating graceful drain. One file remained after that run. Subsequent restart
runs added two bounded files; final retained bytes were 3,336,973,513, below 8 GiB.

Logged host memory for candidate cold was 14,271 MiB available just before the
request, and 12,883 MiB at completion; RSS rose from 432 to 1,486 MiB while a
write was pending. GPU device allocation after loading was 21,816 MiB of
113,564 MiB. These are sampled process/machine observations, not peak-memory
measurements. Available RAM varies, so automatic budgets vary across processes.
The old and explicit-small rejection both reported the required checkpoint
sizes and zero staging use; ordinary generation remained successful.

## Coverage limits

Linux CI/Nix checks were not run on this Windows host; the shared calculation
continues using the existing host/cgroup helper. No full model sweep, image,
reasoning or speculative GPU suite was run. Existing runner state/fallback tests
pass, and those implementations are untouched. HTTP compares generated choices
on matched histories, not full disk-restored logit distributions; the numerical
fixture is separate and uses unchanged model code. Capture reuse, `min_step`
before reservation and early disk admission are deferred, not qualified here.
