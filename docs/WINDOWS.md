# Windows integration status

Stage one, the transport/host increments, and persistence foundations in the
[integration plan](plans/windows-compatibility.md) build the current official
engine's **CPU CLI and selected host/server checks** natively on Windows x64.
This is an integration milestone, not a supported Windows inference release.
Native HIP builds, Windows GPU-budget admission, pinned-memory/transfer checks,
and selected Flash-Next live checks now pass. The Windows queue-policy boundary
is implemented; wider runtime qualification and the GUI remain pending.
Windows CPU CI is implemented; hosted results and the bounded-fixture correction
are recorded below. Disk checkpoint persistence is deferred from
the initial port. GGUF digest caching and private content tracing are enabled.

## Prerequisites and build

Use Windows 11 x64, Visual Studio C++ Build Tools with the Windows SDK, CMake 3.21
or newer, Ninja, Python, Git, and a bootstrapped vcpkg checkout. The initial
compiler for the HIP integration is the clang distributed with TheRock 10.0.0,
extracted under `C:\TheRock\build`. CPU builds can also use standalone LLVM
23.1.2 through `-CompilerBin`; neither CPU build needs ROCm runtime libraries or
a GPU. The pinned dependencies in the root `vcpkg.json` are installed into this
worktree's `build/vcpkg_installed`; the build does not reuse the fork's binaries.

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools/windows/check.ps1 -Vcpkg C:\vcpkg
powershell -ExecutionPolicy Bypass -File tools/windows/build.ps1 -Vcpkg C:\vcpkg
ctest --preset windows-cpu
./build/windows-cpu/gufo.exe --version
./build/windows-cpu/gufo.exe --help
```

Override `-Rocm`, `-CompilerBin`, `-Vcpkg`, or `-Ninja` for your local installation.
The default compiler directory remains `-Rocm` plus `lib/llvm/bin`. `-Jobs`
controls build parallelism and defaults to four. For focused iteration, supply
`-Target gguf_reader_test`, for example. The default `windows-cpu` build preset
builds the CLI, its `gufo-server.exe` copy, and the selected host/server tests.
Use this bounded preset on Windows; other test targets include Linux-specific
harnesses which have not yet been ported. The existing Linux presets remain
unchanged.

For the native HIP development build with TheRock 10.0.0:

```powershell
./tools/windows/build.ps1 -Preset windows-gpu -Vcpkg C:\vcpkg
./build/windows-gpu/gufo.exe probe
```

The script stages TheRock runtime DLLs and the BLAS kernel libraries beside the
executables. Windows kernel-bench profiler markers are disabled when TheRock
does not provide rocprofiler-sdk; Linux still requires that dependency. This
build remains a development milestone pending the qualification below.

The script imports the MSVC headers and libraries, invokes CMake, and stages
runtime DLLs beside the executables. CMake retains the official
`GUFO_RELEASE_VERSION` and `GUFO_REVISION` definitions, and creates the server
copy without requiring symlink privileges. Run the executable from the build
directory distribution; installation/redistributable packaging is not yet
qualified. The locally copied VS LLVM OpenMP DLL comes from `debug_nonredist`
and is for developer use only, not public binary redistribution. Standalone
LLVM builds instead use the OpenMP library and DLL shipped with that compiler.

## Windows CPU CI

The separate [Windows workflow](../.github/workflows/windows-cpu.yml) runs the
bounded `windows-cpu` build and all 15 checks on `windows-2025`, on pull requests
and pushes to `main` or `feat/windows-cpu-compat`, and through manual dispatch.
It retains build/configuration logs and CTest results; it does not publish
executables. The existing Linux CI workflow is unchanged.

[setup-ci.ps1](../tools/windows/setup-ci.ps1) downloads the official
[LLVM 23.1.2 release](https://github.com/llvm/llvm-project/releases/tag/llvmorg-23.1.2)
and verifies the Windows MSVC archive's SHA-256 before extraction. It bootstraps
an isolated vcpkg checkout at the root manifest's `builtin-baseline`. The job
caches the pinned compiler and vcpkg binary archives, leaving CMake build trees
and installed dependencies fresh. vcpkg's ABI keys account for toolchain changes;
the logs record compiler/build-tool and hosted-image versions. Visual Studio
headers/libraries and the Windows SDK come from the hosted image.

To reproduce that setup locally with the prerequisites above:

```powershell
$tools = & ./tools/windows/setup-ci.ps1
& ./tools/windows/build.ps1 -CompilerBin $tools.CompilerBin -Vcpkg $tools.Vcpkg
ctest --preset windows-cpu --no-tests=error
```

The helper's downloads and checkout stay under ignored `build/windows-ci`.
When changing compilers in an existing build, use `-CMakeArgs @('--fresh')`
with CMake 3.24 or newer to reset the compiler probes and preset variables.
This is a CPU contract gate; GPU/model correctness, numerical quality, and
performance require the later native HIP qualification.

Locally validated on 2026-10-12, building on `9d897f15`, with LLVM 23.1.2,
MSVC 14.51.36231, Windows SDK 10.0.26100.0, CMake 4.4.0, Python 3.14.6, and
PowerShell 7.6.6:

- Verified download, selective compiler extraction, isolated vcpkg bootstrap
  at the manifest baseline, and subsequent reuse of the extracted compiler.
- Successful native CPU build and all 14 checks passing in 17.26 seconds.
  The retained JUnit report records zero failures, disabled tests, or skips;
  the staged OpenMP DLL matches the compiler's runtime.
- actionlint 1.7.12, PowerShell syntax checks, and shared documentation and
  dependency checks pass.

These results use the local host's VS/SDK and existing manifest dependencies.
Linux, GPU/model, and performance qualification are not claimed by this increment.

The [hosted run at `afb05c1c`](https://github.com/thomas9120/gufo-adapted/actions/runs/38328358834/job/115072200657)
successfully configured and built, and passed 14 of 15 native checks.
`win32_compat_test` reached its 30-second timeout. Its four-byte write beyond
4 GiB used a regular Windows file, requiring allocation/zeroing of the gap and
a full-file flush. The original test emitted no phase markers, so the hosted
log cannot establish the exact stalled operation.

The fixture now uses `FSCTL_SET_SPARSE`, retaining the same production
`pwrite`/`pread`/`fsync` calls, 64-bit offset, logical-size and identity checks.
It also checks zero-filled hole reads and physical allocation of at most 1 MiB.
Flushed phase markers identify future stalls. No production filesystem behavior
or CTest timeout changed. Focused repeats pass locally in 0.03–0.07 seconds;
the full bounded suite passes. The correction awaits hosted verification.
Logs are retained under `build/windows-ci/compat-timeout-20261012-204754`, including
the original hosted artifacts and local build/CTest/format results.

## Scope and boundaries

Windows build wiring is in `cmake/Windows.cmake`. The CPU-only
`gufo_posix_compat` target supplies the shadow headers and Win32 implementations
to Gufo-owned targets; it links only Winsock, PSAPI, and Advapi32 security APIs,
with no HIP or DXGI
dependency. Linux does not include the module. Linux WebP discovery, Nix,
compiler flags, model framing, algorithms, and the disk format are retained.

Two portable C++ changes are needed to compile the current official code:
the JSON `size_t` constructor must tolerate Windows' identical
`unsigned long long` type, and GGUF hashing uses aligned C++ allocation instead
of `std::aligned_alloc`, which is absent from the Windows CRT. Small socket
option, descriptor-type, and fatal-signal adaptations initially made the CPU
CLI link; server transport validation is recorded below.

Both checkpoint-store implementations reject Windows initialization before
touching their cache directories. Private GGUF digest caching and content
tracing are now qualified by the focused CPU checks below. The filesystem shim
creates protected ACLs, verifies ownership/privacy, and uses native handles for
child-file operations. Unsupported flushes fail explicitly. Checkpoint layout
durability, store-level recovery, and their native fault harnesses are deferred;
the isolated filesystem checks do not establish those guarantees. The initial
Windows port will run without disk checkpoint persistence: RAM continuation
reuse remains in scope, but restarting the engine requires prompt recomputation.
Conversation history kept by a client or GUI is independent of these stores.

HIP configuration on Windows currently stops with an explicit stage-three
message. Keep using the existing `windows-port` checkout for its qualified
runtime paths while this integration proceeds.

## Stage-one validation

Validated on 2026-10-12 against official base
`8d4e1a2b74c94752b9b16b1dda153cbf52638443` with:

- TheRock 10.0.0, clang 23.0.0, LLD, MSVC 14.51.36231, Windows SDK 10.0.26100.0.
- CMake 4.4.0, Ninja, Python 3.14, and the manifest's pinned vcpkg baseline.
- Successful native CPU configuration and build; `gufo.exe` and its server
  copy run with their colocated DLLs.
- All nine `windows-cpu` checks pass: version, help, argument parser, JSON,
  JSON constraints, GGUF reader, GGUF identity, Windows compatibility, and the
  cache dependency boundary (including the shim's transitive links).
- Compatibility coverage includes spawning/waiting with spaced arguments,
  independent exclusive locks, offsets beyond 4 GiB, UTF-8/spaced directory
  names, directory-descriptor ownership, no-replace rename, file identity,
  file flush, explicit directory-flush failure, and disk-store rejection.
- GGUF checks cover mapped reads and full/parallel hashing, including edits
  with restored timestamps. Windows verifies that a mapped file cannot be
  shrunk; the Linux shrink-induced prefetch-failure test is retained on Linux.
- The shared formatting check passes with clang-format 21.1.8.

These stage-one checks did not qualify HTTP/WebSocket, disk-cache recovery,
Linux builds, GPU inference, performance, or numerical model behavior.

## Stage-two server transport validation

Validated on 2026-10-12 with the same native toolchain, building on stage-one
commit `bc7f8c64`. The bounded `windows-cpu` preset now runs **12 checks**, adding
`http_server_test`, `audio_websocket_test`, and `win32_server_shutdown_test`.
All 12 pass, along with the shared formatting, documentation, and dependency
checks. The CLI also rejects `--trace` before loading a model and creates no file.
The server checks use real local TCP connections and mock generation/audio
services; no GPU or model is needed.

The implementation retains native Winsock handles, maps receive/send errors,
uses millisecond socket timeouts, and closes sockets through Winsock. The
listener remains nonblocking; accepted Windows connections are restored to
blocking mode because they inherit the listener's properties, as described by
[Microsoft's accept documentation](https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-accept).
The listening address is exclusive on Windows. Response cleanup drains briefly
before closing, with no network wait under the worker mutex.

Server stop cancels pending socket I/O and joins workers; each worker retains
its handle until its I/O finishes. Native experiments and the subprocess tests demonstrated that
Winsock `shutdown` alone does not interrupt a pending receive.
[CancelIoEx](https://learn.microsoft.com/en-us/windows/win32/api/ioapiset/nf-ioapiset-cancelioex)
requests cancellation; the worker still owns the handle until its I/O finishes.
WebSocket readers observe an explicit server-stop token between bounded poll
waits, while retaining the idle limit for sessions receiving no input or output.

Coverage includes:

- Existing HTTP authorization, request parsing/framing, API responses, SSE
  heartbeats, deferred headers, streaming failures, and device-loss tests.
- Split HTTP heads/bodies, graceful and reset disconnects during buffered and
  streaming generation, and cancellation on server stop.
- Stop during incomplete HTTP requests and a blocked response write, without
  requiring the peer to send, read, or disconnect.
- Existing audio WebSocket handshakes, buffered/fragmented frames, ping/pong,
  streamed output, protocol-error close replies, and peer-close cancellation.
- Silent-session idle expiry, continued output keeping a listener alive,
  delayed consumption of a close reply with unread input, and local stop during
  incomplete frames and active audio work.
- Real Ctrl+C and Ctrl+Break events in six isolated hidden-console subprocesses:
  each event stops an idle listener, active generation, and an incomplete request
  through normal cleanup. Embedded `run(false)` keeps caller-owned handlers.
- Explicit Winsock timeout/error mapping, occupied-port rejection, and content
  trace refusal without creating a file.

Console-window closure, logoff, and forced process termination do not provide a
graceful-cleanup guarantee. Persistent-cache crash recovery is a separate
increment. The Linux test paths are
retained, including their signal subprocess tests, but a Linux build remains
pending: the toolchain is not installed in this host's WSL environment. Windows
CPU CI and live GPU/model correctness and timing qualification remain pending.

## Stage-two host integration validation

Validated on 2026-10-12 with the same native toolchain, building on transport
commit `7a2ab5bb`. All **13** bounded `windows-cpu` checks pass, including the new
`win32_host_test`. Shared formatting, documentation, and dependency checks pass.
The native test covers host queries, snapshot budget/ceiling arithmetic, live
memory log fields, explicit Linux-style fixtures, diagnostic verdicts, and cache
path selection with Unicode/spaced paths and empty/missing environment values.

The adaptations use the fork at
`ab4d98189a6d8103fce29d2537e4113da3da2420` as their reference. Windows reports
CPU names/vendor through CPUID and counts actual physical cores and NUMA nodes
through
[GetLogicalProcessorInformationEx](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex),
rather than assuming two threads per core. Logical processor counts span all
processor groups. Explicit fixture roots retain the existing `/proc` parser;
missing fixtures do not fall back to the live Windows host.

Host inventory, HTTP logs, and RAM snapshot admission now use native memory
queries. The process RSS field reports its Windows working set. Total/available
RAM comes from
[GlobalMemoryStatusEx](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-globalmemorystatusex).
It is OS-visible physical RAM, not installed RAM or a WDDM device budget.
Admission retains the existing half-available automatic budget and headroom
ceiling. Linux CMA/cgroup logic remains on Linux. Windows Job Object memory
limits and actual HIP allocations have not been qualified.

Live CLI JSON and human diagnostics were checked against Windows CIM on this
host: Ryzen AI MAX+ 395, 16 physical cores, 32 logical processors, one NUMA node,
Windows `10.0.26300`, and `33,981,009,920` OS-visible physical bytes. CPU-only
diagnostics report GPU/Windows driver compatibility as `unknown`, with GPU and
toolchain warnings; platform-only diagnostics pass. Compiler metadata identifies
the actual Clang build. They no longer inherit Linux's hard-coded GPU, amdgpu,
GCC, or ROCm values. Memory hardware type/unified classification is unavailable;
`memoryType` is `unknown`, `isUnified` remains false, and the unqueried Linux
`freeBytes` field is zero with its limitation identified in `source`.

The shared default-cache selector uses `XDG_CACHE_HOME/gufo`, then on Windows
`LOCALAPPDATA/gufo`, then `HOME/.cache/gufo`. With no usable root it returns no
path. Windows reads the native wide environment, preserving Unicode profile
paths. At that increment both stores and persistent GGUF identity caching
remained disabled before any directory creation. Opt-in continuation cache
directories remain caller-selected. The persistence-foundation increment below
qualifies the independent identity cache and tracing. Linux build verification,
Windows CI, HIP, and the GUI remain pending.

## Persistence foundations and unresolved cache contract

Validated on 2026-10-12 on local NTFS with the same native toolchain, building on
host-integration commit `e20cc86a`. That increment's bounded preset contains
**14 checks**, including `win32_filesystem_test`. All pass. This increment qualifies private
file operations, tracing, and identity caching; it does **not** complete the
checkpoint-persistence increment.

`compat/win32/filesystem.cpp` isolates native handle-relative operations. It
uses
[NtCreateFile](https://learn.microsoft.com/en-us/windows/win32/api/winternl/nf-winternl-ntcreatefile)
with the parent directory handle for `openat`/`mkdirat`; directory enumeration
also uses the handle. Rename, no-replace publication, unlink, and `fstatat` keep
that identity after the directory is renamed and its old path is replaced.
Replacement permits existing readers to finish with the old file. No-follow
opens reject intermediate and final reparse points. Native junction tests do
not require symbolic-link privileges, which are unavailable on this host.

New `0600`/`0700` objects receive a protected DACL granting access only to the
process's token user SID. Existing ownership and ACLs are inspected; synthetic
uid zero means that SID, while foreign/unknown ownership fails cache checks.
Mode bits are conservative privacy indicators, not complete Unix permission
emulation. `chmod`/`fchmod` implement private `0600`/`0700` ACLs and reject other
modes; they are no longer no-ops. Unknown or broader ACL shapes fail privacy
checks. Tests inspect the actual ACL and demonstrate access denial with a
restricted token. NTFS ChangeTime is required rather than falling back to
mtime, so restoring mtime cannot conceal edits from cached GGUF identities.

GGUF digest caching uses the unchanged upstream format, full-hash invalidation,
and malformed-digest fallback. Native cache files and directories are private.
`--trace` creates private files and rejects unprotected existing files, reparse
points, directories, and files with multiple hard links before recording data.
The HTTP trace content/authorization/streaming tests now run on Windows too.
These facilities are separate from both disabled checkpoint stores.

Filesystem tests force process death after payload write, file flush, rename,
and directory flush, then reopen and verify the namespace and complete payload.
Failure injection covers write/flush failure and errors before or after rename
and unlink take effect. Fault controls exist only in `BUILD_TESTING` builds;
production compilation of the filesystem/shim was also checked with those
controls absent. Process termination does not model power loss, and these are
operation tests, not checkpoint-store recovery tests. Linux's ELF-wrap/fork
harnesses remain unchanged.

Native experiments show
[FlushFileBuffers](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers)
succeeds on writable directory handles here and rejects a read-only handle.
The shim requests directory write access rather than ignoring a failed flush.
However, the new common store syncs every ancestor up to the filesystem root
when creating its layout. This unprivileged process cannot obtain that access
to `C:\Users` or `C:\`, although it can flush its own directories. A successful
rename or a process-crash test cannot establish the missing ancestor durability.

The selected contract preserves upstream guarantees: both checkpoint
guards stay intact. The user deferred disk checkpoint persistence to reduce
maintenance when merging upstream updates. It does not block native HIP or GUI
integration. A future port requires resolving the Windows
durability contract, qualifying the store algorithms and privacy checks, and
adding native restart/reclamation/corruption/failure tests. A weaker Windows
power-loss contract has not been selected or implemented. No Linux build,
live GPU/model cache behavior, numerical quality, or speed qualification is
claimed by these CPU checks.

## Initial native HIP live inference

On 2026-10-12, the native `windows-gpu` build based on `3137e712` plus the HIP
build-wiring changes passed a bounded live smoke check on the Radeon 8060S
(`gfx1151`). The compiler is TheRock 10.0.0; `probe` reports runtime
`7.15.26333` and 107.87 GiB device memory. The preset uses RelWithDebInfo with
assertions enabled. This evidence covers one model on this host.

Artifacts supplied locally were the four-part
`Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF, `mmproj-BF16.gguf`, and
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. With one session, a 4096-token context,
thinking off, and disk cache off:

- AR loaded in 23.839 seconds and MTP in 35.166 seconds. Health and model
  discovery succeeded, advertising text and image inputs.
- Both modes returned the expected buffered answer `42`, streamed answer `10`
  with a terminal SSE marker, and conversation follow-up `50`.
- Both follow-ups reported a RAM cache hit, 28 reused prompt tokens, and only
  29 new prefill tokens. AR reported zero drafts. Each MTP request actually ran
  one draft round, proposing three tokens and accepting one.
- An AR image request correctly identified a generated solid-red PNG as `Red`.

Requests, responses, per-request timings, server logs, and smoke scripts are
retained locally under `build/windows-gpu/live-flash-20261012-184954`. The test
servers were stopped after use. These short checks establish live execution,
streaming, a RAM restore, and one vision example; they do not establish numerical
quality, a speed comparison, long-context/cancellation behavior, allocation
pressure, or support for other models. The GPU-budget increment below addresses
WDDM admission. Pinned-memory contracts, broader model-owned checks, and the final
Linux/Windows qualification remain required. Disk checkpoint persistence remains
excluded from the initial port.

## Windows GPU-budget admission

The GPU-only [platform helper](../src/core/platform/device_memory.hpp) adapts
the fork's budget query against the current official source. On an integrated
Windows GPU it matches the calling thread's current HIP device to DXGI by LUID,
then queries the process's local and non-local segment budgets through
[QueryVideoMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo).
Available bytes are the sum of each segment's nonnegative `Budget - CurrentUsage`.
Total bytes are the summed OS budgets, not physical installed memory. These are
admission limits; allocation can still fail as budgets and usage change.

Adapter discovery is shared under a mutex, with an independent COM reference
for each query. Switching the current device cannot leave a caller using the
previous device's adapter. Budget values are refreshed on every query. TheRock
10.0.0 supplies a valid LUID but a zero node mask on this `gfx1151` APU; that known
single-node case uses node zero. Unknown zero/multiple-node masks and discrete
GPUs retain the HIP fallback. Failed/partial DXGI queries and overflowing sums
leave the complete HIP result intact. HIP errors return before DXGI is consulted;
null output pointers retain the runtime's partial-query behavior.

Six production admission sites use the helper: the Qwen, DeepSeek, and
Flash-Next resource claims; DeepSeek's FP16-cache and managed-KV decisions; and
Flash-Next's snapshot allocator. The loader's informational `hipMemGetInfo`
query remains raw HIP for comparison. Linux's helper directly forwards to
`hipMemGetInfo`. The Windows CMake module links DXGI only into the HIP build;
the CPU shim and common cache retain their reviewed dependencies.

Validation on 2026-10-12 uses the same TheRock/VS/SDK host as the initial HIP
record, based on `1e4c4333` plus this increment:

- The native HIP engine rebuild and the new GPU-only
  `win32_device_memory_test` pass. Its independent DXGI oracle matches the active
  HIP adapter before/after a 512 MiB allocation, after free, and on another thread.
  Synthetic checks cover reduced/exhausted budgets, failure of either segment,
  partial results, and overflow; real null-output calls match HIP. This host has
  one HIP device, so actual switching between multiple GPUs is not qualified.
- All 14 Windows CPU checks pass in 17.50 seconds. The cache dependency boundary,
  shared C++ formatting, and dependency-record checks pass. Two profiler no-op
  functions from the prior HIP increment were formatted to satisfy the shared
  check; their behavior is unchanged.
- Selected live cancellation/replay checks pass in AR and MTP, including repeated
  fresh-server runs. Both restore 317 cached tokens and prefill only 34 new tokens;
  the third turn retains the growing prefix. Seeded replay output and cache work
  match the earlier binary. AR performs zero drafts; interrupted MTP generations
  actually propose and accept nine tokens over two draft rounds.

Per-request/phase comparisons retain the repository's 5% and 3 ms limits.
Early comparisons using different executable names flagged a warm MTP replay's
wall/server duration. A temporary trace found no budget query during that replay;
the tracing code was removed. Alternating the binaries at the same executable
path did not reproduce that apparent slowdown: the warm replay's server duration
was 89.3-91.7 ms for the control and 85.5 ms for both candidate observations.
This does not establish the cause of the earlier difference. Other timings vary,
so overall AR/MTP performance evidence remains **inconclusive and unqualified**.
Earlier flags, repeated observations, and unchanged-binary controls are retained;
no timing margins were widened and no speed qualification is claimed.

The retained evidence is under `build/windows-gpu/wddm-20261012-190713`, including
binary identities, build/CTest logs, server commands, model file metadata,
request/phase timings, and every timing comparison and control. The earlier
binary's SHA-256 matches the initial smoke record, and its recorded build patch
matches `1e4c4333`. The selected live check uses `tests/functional/continuation.py`
with `content-preserve0-sampled0`, one session, context 4096, and a fixed 2 GiB RAM
cache. All test servers were stopped. Disk restart is excluded. Performance
qualification and the wider
Linux/model checks remain pending; these checks do not establish
a physical-memory-pressure or allocation-failure contract.

## Windows pinned backing and transfers

TheRock 10.0.0 on the local Radeon 8060S (`gfx1151`) passes the native
[`win32_pinned_transfer_test`](../tests/platform/win32_pinned_transfer_test.cpp).
The increment adds only Windows test wiring and a test; the production backing,
staging, transfer and model code is unchanged.

The test covers explicit `hipHostMallocCoherent` flags and zeroed pages,
precommitted block accounting, exhaustion/reuse, pins retaining physical memory
and accounting after pool destruction, and four concurrent borrowers. Ledger
reserve/convert failures exercise rollback, including failure after backing
allocation. Staging retains its allocation after the allocator and original
owner release it; all ledger charges return to zero after owners drain.

Real H2D/D2H transfers preserve a nonuniform payload with unaligned endpoints,
4,093-byte pieces, a partial tail and untouched guard bytes. The test also
covers stream/event lease exhaustion and reuse, repeated waits, sealed streams,
zero-byte copies, independently progressing streams with one pending host
callback, and a completion surviving pool destruction. Snapshot transfers
drain before releasing a borrowed stream and preserve row padding with different
source/destination pitches. The common cache dependency boundary passes.

Build and run the focused checks after the HIP development build above:

```powershell
./tools/windows/build.ps1 -Preset windows-gpu -Target win32_pinned_transfer_test -Vcpkg C:\vcpkg
ctest --test-dir build/windows-gpu -R '^win32_pinned_transfer_test$' --output-on-failure
./tools/windows/build.ps1 -Preset windows-gpu -Target qwen38_flash_next_continuation_probe -Vcpkg C:\vcpkg
./build/windows-gpu/qwen38_flash_next_continuation_probe.exe --model PATH --mtp-model MTP_PATH --mode ar --context 4096 --sessions 2 --backing-gib 1 --round-trip
./build/windows-gpu/qwen38_flash_next_continuation_probe.exe --model PATH --mtp-model MTP_PATH --mode mtp --context 4096 --sessions 2 --backing-gib 1 --round-trip
```

The Linux `cache_backing_test` and its ELF `--wrap` hooks are unchanged. The
native test does not observe every HIP allocation/copy call or inject HIP
allocation, submission, event or drain failures; those Linux fault cases remain
unqualified on Windows. Disk persistence remains disabled. Physical memory
pressure, larger backing capacities, broader numerical quality and speed
qualification are separate work.

The unchanged Flash-Next continuation probe passes AR and MTP cross-slot
round-trips with the supplied `Qwen3.8-Flash-Next-UD-Q4_K_XL` four-shard model,
the shared Q8 MTP sidecar, and the automatically discovered BF16 projector.
Each run prefills two different 4,096-token histories, allows 4,224 positions
for the next decode, and commits 1 GiB of coherent backing. Capture uses four
64 MiB blocks: AR captures 131,614,880 private bytes and 103,825,408 row bytes;
MTP captures 133,032,096 private bytes and 112,473,856 row bytes.

Restoration into the other slot reproduces token history and logits bit for
bit; readback verifies every captured device component. The next decode step
also matches tokens, logits and draft/acceptance statistics. Nonzero MTP drafting
is asserted during that post-restore step. These are focused state-integrity
checks, not a broader model-quality or timing qualification. The probe's capacity
printout still uses raw HIP memory reporting; it does not qualify Windows
capacity limits under pressure.

Evidence is retained under `build/windows-gpu/pinned-20261012-200729`: native
CTest/build logs, AR/MTP probe output, exact commands, source base and binary
hashes, CMake toolchain configuration, and model file sizes/modification times.
Full model-file hashes were not collected. Formatting and documentation checks
pass. No production source change was needed, and the earlier inconclusive
performance qualification remains unchanged.

## Windows GPU queue policy

The server's automatic queue planning is Linux-only. Its KFD cross-process
census and measured text/audio caps do not establish a Windows policy: absent
`/sys` files are not evidence that the GPU has no other queues. A 14-line Windows
branch at the server's sole planning call leaves `GPU_MAX_HW_QUEUES` untouched.
An inherited setting remains the operator's responsibility; otherwise HIP uses
its runtime default. No Windows tuning option or duplicate planner is added.

At INFO level Windows startup reports, for example:

```text
event=queue_budget kind=llm observed=unknown budget=unknown expected_max=unknown cap=runtime census=unavailable
```

A nonempty inherited setting reports `cap=operator`. The line intentionally
does not predict resident queues, apply Linux's eight-queue budget, or issue
Linux's power/clock warning. Help, invalid invocations and quieter log thresholds
still suppress the startup diagnostic. Linux's queue implementation, measured
caps and tests are unchanged.

The new Windows-only `win32_queue_policy_test` exercises all five server
modalities with absent, empty, numeric and nonnumeric inherited values, all four
log thresholds, help and invalid addresses before model loading. It passes in
the bounded CPU preset, which now has **15 checks**, all passing in 17.22 seconds.
The unchanged Linux planner's synthetic KFD fixture test also passes when built
on Windows; this is not a Linux-hosted CI qualification.
The queue-policy test passes against the native HIP executable too, in 13.96
seconds. Formatting, documentation and dependency checks pass.

```powershell
ctest --preset windows-cpu
ctest --test-dir build/windows-gpu -R '^win32_queue_policy_test$' --output-on-failure
```

Selected Flash-Next AR/MTP live checks pass against the preceding WDDM binary
using the supplied four-shard Q4_K_XL model, shared Q8 MTP sidecar and BF16
projector, at 4,096 context tokens, one session and 2 GiB RAM cache. The unchanged
`content-preserve0-sampled0` functional case covers seven generation requests,
including two interrupted streams, RAM resume/replay and third-turn reuse.
Seeded output, token counts and expected prefill/cache work match exactly; AR
does not draft and MTP performs actual drafts. These checks also pass with an
explicit inherited `GPU_MAX_HW_QUEUES=2` on the new binary.

Per-request/phase timing comparisons retain the strict 5%/3 ms thresholds.
Both default-policy pairs per mode, unchanged-baseline controls and the explicit
operator-setting diagnostic use the same `gufo.exe` path and fresh processes.
AR cancellation-handoff queue wait spans 15.05–38.52 ms on unchanged baseline
controls and 27.93–38.26 ms on the default-policy candidate. The flagged MTP
snapshot spans 7.55–16.16 ms on baseline controls and 15.12–18.43 ms on the
candidate. Controls reproduce the stalls; observations overlap. No confirmed
timing regression is established, but both modes remain **inconclusive and
unqualified for performance**. The operator-setting diagnostic does not replace
qualification of the runtime default or establish a Windows tuning policy.

Evidence is retained under `build/windows-gpu/queues-20261012-202345`: build/CTest
logs, commands, complete server and request measurements, all comparisons,
source patch, binary hashes, CMake configuration and model size/mtime identity.
Full model-file hashes were not collected. Windows queue residency, idle power,
multiple GPU processes, broader model quality and other modalities' inference
throughput remain unqualified. Linux-hosted checks remain pending. This change
does not enable disk checkpoint persistence.
