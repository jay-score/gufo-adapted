# MTP catch-up boundary qualification

October 9, 2026. Fork baseline: `8cbff0135dad60f62840b7d206ddf7761b598450`.
Source: [PR #485](https://github.com/gufo-org/gufo/pull/485), merged as
[`fd747a51951ccd09eda20de5c34313670dc9b9d3`](https://github.com/gufo-org/gufo/commit/fd747a51951ccd09eda20de5c34313670dc9b9d3),
with reviewed head `1a2b49476d140146831b68f450ceac76cb0e8618`.
The comparison report pins official main at
`e6c850d3df61578bf878db2ad1fc96266d8bc07d` and merge base at
`d9a84f13f35d1f98da22886a12eb25dc7062e392`.

## Scope and acceptance

Adapt only the executor's last-only attention boundary split and final query-mask
coverage. Sparse groups can start at a different row after the dense prefix is
split off; score the final 16 rows instead of starting at the last aligned dense
tile. Skip a dense prefix whose results the final sparse tile does not consume.
Keep the existing fixed 512-query selector chunk and all kernel arithmetic.
No mapped n-gram memory, prefetch, GDN, HC, GEMM, routing or tuning changes.

The existing full predictor forward controls the headless catch-up path on the
same tokens, hidden input and retained KV/indexer prefix. A separate control
executor shares immutable weights but keeps selector/activation scratch private,
so the full forward cannot supply missing masks to headless catch-up. Attention,
FFN, carried hidden state, head mixer and candidate outputs must match exactly,
including the following recursive proposal. The focused probe adds widths
255/256/258 alongside 224/257/2047/2048 to cover all sparse group alignments.
The default independent scalar audit still runs separately; its tolerances are
unchanged.

Snapshot version 17 rejects earlier Flash-Next payloads: version 16 can contain
predictor state computed by the incorrect boundary route. The layout is unchanged.
Old disk entries fall back to prompt computation; no user cache is deleted.
Reverting to version 16 rejects new version-17 entries in the same way.

## Identities

Windows 11 x64, gfx1151; TheRock 10.0.0, clang 23.0.0git / LLVM
`8f497e0992fb7513f7f78a6f6b6f1056c375e961`, MSVC 14.51.36231,
Windows SDK 10.0.26100.0, build CMake 3.29.2, Ninja 1.12.0 and clang-format 21.1.8.
Fresh `gpu-test` builds (RelWithDebInfo, active test assertions), using the same
pinned vcpkg manifest
(SHA-256 `d711d2a008fd06e4d16438e033cbe463f44d5af82e82bf9f52b2c56fbe32bf3e`).
Baseline inference code is unchanged; both sides use the extended test harness.
Baseline and candidate Gufo report `8cbff0135dad-dirty`; the baseline's dirty
files are test-only and the candidate also contains the fix.

The IQ4_XS target's three shards and Q8_0 MTP sidecar were freshly hashed and
match the [PR #470 artifact identities](prefill-pr470-qualification.md#identities).
The image check uses that record's BF16 projector and the existing two screenshots.

| Binary | SHA-256 |
| --- | --- |
| Baseline Gufo | `34da1ca897a49cb44ec16ab9e76934bf91786aad4dfeb50fe529deceab1137fc` |
| Candidate Gufo | `7310a94d6879cd7ec7e766deb5b8c8e33bab2676cb02bd25efe59a16fc22409b` |
| Baseline session test | `a40c9d0e660fe5a1beaa8e10d02ced47f24338eb44d0a5a8acc6b61be832aa3b` |
| Candidate session test | `0d17f9cc1706c0d773ff464fe5ca7ae1f826fe15358053bcb4c158ac4ffe5142` |
| Baseline focused GPU probe (original shared-scratch audit) | `fb004d1e3011c4b9325b0990999284faab5048a050a41a8b4d651b70a6bfce97` |
| Candidate GPU probe (isolated-scratch audit) | `4e7a474644a8aa9404b29aa05b8bbdd7a089f87a7f96b224d190e49a4a7a54d2` |

## Commands and results

Run from the worktree with `MODEL`, `MTP` and `MMPROJ` set to the recorded artifacts.
Build each affected target with `tools/windows/build.ps1 -Preset gpu-test -Target
TARGET -Jobs 4` and the configured TheRock, vcpkg and Ninja paths. Build outputs
and runtime assets stay in this worktree. Baseline executables are preserved
beside the matching runtime files before rebuilding the candidate.

```powershell
build/gpu-test/qwen38_flash_next_gpu_probe.exe --model $MODEL --mtp-model $MTP --mtp-catchup-audit --batch 2048
ctest --test-dir build/gpu-test -R '^qwen38_flash_next\.(attention_ops|select_ops|mtp_ops)$' --output-on-failure --no-tests=error
build/gpu-test/qwen38_flash_next_session_test.exe --model $MODEL --mtp-model $MTP --capture-prefill-logits build/candidate-prefill.bin
build/gpu-test/qwen38_flash_next_session_test.exe --model $MODEL --mtp-model $MTP
build/gpu-test/qwen38_flash_next_snapshot_test.exe --model $MODEL --mtp-model $MTP
build/gpu-test/qwen38_flash_next_rollback_test.exe --model $MODEL --mtp-model $MTP
build/gpu-test/qwen38_flash_next_image_prefix_test.exe $MODEL $MTP $MMPROJ IMAGE_A IMAGE_B
build/gpu-test/qwen38_flash_next_gpu_probe.exe --model $MODEL --mtp-model $MTP --mtp-audit --batch 2048
python tools/ci/check-kernel-resources.py build/gpu-test/gufo.exe --bundler MATCHING_CLANG_OFFLOAD_BUNDLER
python tools/ci/check-format.py
python tools/ci/check-docs.py
```

The unchanged executor, using the original shared-scratch audit, fails at round 9,
start position 2025, width 224, at
`wide attention` (relative RMS `0.0000009`, maximum absolute difference
`0.0000045`; the catch-up contract permits zero difference). The candidate
passes all 55 rounds over seven widths with isolated executor scratch, including
recursive stages/candidates. A second negative control keeps the boundary split
fix but restores the old aligned `first_query`: it fails at round 7, start
position 1806, width 257, at `wide attention` (relative RMS `0.0000055`, maximum
absolute difference `0.0000242`). This confirms the regression detects missing
final sparse masks independently of the split correction.
All three focused GPU operator tests pass. The 57-kernel resource guard has zero
failures, and all 488 C++ files pass formatting.

The full independent scalar MTP audit fails identically on baseline and candidate
before reaching catch-up: `attention-state` relative RMS `0.0093689`, maximum absolute difference
`0.0474995`. It is not counted as a pass. The focused test entry point preserves
that audit and bypasses it only when explicitly checking catch-up against the
full quantized predictor execution.

All 270 captured full-vocabulary rows (67,046,400 logits, 268,185,600 bytes)
are byte-identical to baseline: SHA-256
`dfc7d5b755688838732f806676ce238af0b1ad461c6ced5d8aa3963ea38f1731`.
The unchanged three-fixture corpus records its token hashes in each capture log;
prefixes are 96/1023/1024/2048/4096 in AR and MTP modes, followed by eight fixed
Evaluate continuations. All 240 labels retain mean NLL
`0.25759154328132261` and perplexity `1.2938102451945588`.

The full session suite passes: 80 chunk-frontier rows, failure/cancellation
recovery, C2/C4/C6/C8 batching, sampled acceptance/rejection, fresh-load replay,
and all 23 serving sampling strategies in both AR/MTP modes. Snapshot round trips,
old-version rejection, all 105 rollback-prefix continuations and the BF16
image-prefix extension/replay/cancellation checks pass. The projector and both
screenshots were freshly hashed and match the PR #470 identities.

Linux build/CI, other quantizations, an independent full-model parity claim and
performance measurements are outside this qualification. Logs, commands,
model/build hashes and raw logits
are retained under ignored `build/` in the worktree.
