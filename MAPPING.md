# Step 1: Evidence-Based Discovery

Read-only discovery report for `./`, `../gufo-upstream`, and `../Flash-next-ssd`. The report records exact source references and explicitly identifies unverified or absent evidence.

## Section 1.1: Build System & Environment Audit

**[VERIFIED IN TARGET]** The project requires Windows or Linux x86-64, uses C++20, and selects HIP architecture `gfx1151` when HIP is enabled. On Windows it adds `HIP_PATH` to CMake’s prefix path. (`./CMakeLists.txt:3-5,15-17,60-89`.) The `release` preset uses Ninja, `RelWithDebInfo`, HIP enabled, and tests disabled; it builds under `build/release`. (`./CMakePresets.json:10-28,66-71`.)

**Machine/toolchain audit:** Windows 11 Home, build 26200; CMake 4.4.3; Ninja and Visual Studio C++ Build Tools are present. An initial default-path check reported TheRock missing at `C:\TheRock\build` and vcpkg missing at `C:\vcpkg`. Subsequent inspection found both in the user profile: TheRock 10.0.0 at `C:\Users\danie\AppData\Local\Gufo\dependencies\therock-10.0.0` and vcpkg at `C:\Users\danie\AppData\Local\Gufo\dependencies\vcpkg-9e2895bf6afb`. The TheRock version file says `10.0.0`; both clang executables, `vcpkg.exe`, and its CMake toolchain file exist there. Before Python was installed, the repository’s read-only check reported it missing; after installation, rerunning with explicit `-Rocm` and `-Vcpkg` paths reports all prerequisites found. The toolchain location and version discovery logic are in `./tools/windows/setup.ps1:14-16,67-73,101-114`; explicit build overrides are accepted by `./tools/windows/build.ps1:15-24,54-56,83-96`.

The repository’s Windows build script expects TheRock **10.0.0**; this differs from the spec’s ROCm 10.1 statement. (`./tools/windows/build.ps1:3-11,35-45`; `./docs/WINDOWS.md:87-103`.) The `HIP_PATH` environment variable points to an additional ROCm 7.2 installation; the user-profile TheRock 10.0.0 root is the validated repository toolchain.

The exact repository build command for the discovered user-profile installs (not executed during this read-only discovery) is:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset release -Jobs 4 -Rocm 'C:\Users\danie\AppData\Local\Gufo\dependencies\therock-10.0.0' -Vcpkg 'C:\Users\danie\AppData\Local\Gufo\dependencies\vcpkg-9e2895bf6afb'
```

The script configures with `cmake -S $root --preset $Preset`, passing TheRock’s clang as C/C++/HIP compilers, the vcpkg toolchain and manifest paths, the TheRock prefix, and LLD; it then runs `cmake --build $build --parallel 4`. (`./tools/windows/build.ps1:76-104`.) The executable location is `build\release\gufo.exe`. (`./tools/windows/build.ps1:76,128-155`.) The `release` preset uses `BUILD_TESTING=OFF`; the test preset is `gpu-test`. (`./CMakePresets.json:20-28,54-64`.)

**ROCm header probe:** `<hip/hip_file.h>` is **NOT FOUND** in either the active Windows SDK at `C:\Program Files\AMD\ROCm\7.2\include\hip\` or the validated user-profile TheRock 10.0.0 SDK at `C:\Users\danie\AppData\Local\Gufo\dependencies\therock-10.0.0\include\hip\`; `<hip/hip_runtime.h>` is present in both. No Linux ROCm SDK was inspected on this Windows host.

## Section 1.2: Cross-Repo Evidence Table

| Subsystem | Symbol / Struct / Function | Repo & file:line | Role | Callers | Memory / Sync Implications | Open Questions |
|---|---|---|---|---|---|---|
| [VERIFIED IN TARGET] GGUF loader and tensor mapping | `GgufReader::OpenFile`, `OpenSingleFile`, `Binder::Get`, `ModelWeights::Bind`, `TensorRef` | `./src/core/gguf_reader.cpp:123-194`; `./src/models/qwen38_flash_next/weights.cpp:59-118,255-310`; `./src/models/qwen38_flash_next/weights.hpp:38-64` | Opens and memory-maps GGUF shards, binds tensor metadata, and records each tensor’s shard and byte offset. | `Model::Load` calls `GgufReader::OpenFile` and `ModelWeights::Bind` (`./src/models/qwen38_flash_next/engine.cpp:87-95`). | Tensor references point into mapped data; mapped regions retain file descriptors (`./src/core/gguf_reader.cpp:163-194`). | Which target model artifact and split layout will be used for empirical offset measurement? |
| [VERIFIED IN TARGET] GPU/host allocation | `Uploader::Copy`, `Executor::Create`, HIP allocation calls | `./src/models/qwen38_flash_next/kernels/rocm/device_model.cpp:132-184`; `./src/models/qwen38_flash_next/kernels/rocm/executor.cpp:462-523`; `./src/models/qwen38_flash_next/kernels/rocm/batch.cpp:790-799` | `Uploader::Copy` uses `hipMalloc` for a tensor-sized allocation; executor/batch paths use `hipHostMalloc` for pinned host buffers. | `DeviceModel::Upload` uses `Uploader::Copy`; executor setup allocates pinned staging and scratch buffers. | Tensor allocations include the full `TensorRef::SizeBytes()`; pinned host buffers are used for staging and shared host/device access. No named GTT allocation wrapper was located. | Which HIP allocation mode on the target driver is suitable for direct NVMe reads into GPU-visible memory? |
| [VERIFIED IN TARGET] MoE router and expert dispatch | `RouterTopKKernel`, `RouterTopK`, `Executor::MoeBatch`, `Executor::Experts`, `qfn_mmq_moe_vec` | `./src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp:2627-2694,6302-6312`; `./src/models/qwen38_flash_next/kernels/rocm/batch.cpp:683-713`; `./src/models/qwen38_flash_next/kernels/rocm/executor.cpp:1117-1172`; `./src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp:351-488` | Computes router top-k and dispatches routed expert MLPs. Q4_K batch routing has a specialized path; the expert dispatcher includes IQ4_XS and Q4_K formats. | `Executor::MoeBatch` calls `RouterTopK` and Q4_K vector kernels; `Executor::Experts` dispatches by tensor type. | Routed expert tensors currently pass through tensor-sized HIP allocations and uploads; routing does not currently imply per-expert SSD residency. (`device_model.cpp:152-184`.) | How should routed expert IDs resolve into cache slots without changing existing kernel arithmetic or quantization contracts? |
| [VERIFIED IN TARGET] PLE / n-gram lookup | `ModelWeights::Bind`, `NgramTable::Open`, `HashNgramRows`, `NgramTable::ReadOne`, `Executor::PleFetch`, `Executor::WaitPle` | `./src/models/qwen38_flash_next/weights.cpp:288-300`; `./src/models/qwen38_flash_next/ngram.cpp:17-24,77-132,165-217`; `./src/models/qwen38_flash_next/kernels/rocm/executor.cpp:1381-1458`; `./src/models/qwen38_flash_next/kernels/rocm/batch.cpp:816-835,873-880` | Binds `per_layer_token_embd.weight`, hashes token histories, fetches rows from the GGUF shard, and transfers gathered embeddings for the PLE layer. | `Model::Load` opens the n-gram table (`./src/models/qwen38_flash_next/engine.cpp:114-121`); prefill/batch paths start and wait on gathers. | The table is read from disk rather than uploaded as a whole. `NgramTable` has a fixed 8 MiB compressed-row cache; Linux attempts `O_DIRECT` with 4 KiB alignment, while Windows uses `O_CONCURRENT_RANDOM` and cached reads. (`ngram.cpp:17-24,98-110,173-216`.) | The literal tensor name `.ngram_embedding` is **NOT FOUND** in the target; what target artifact sizes and PLE format must the budget planner support? |
| [VERIFIED IN TARGET] Speculative decoding | `Session::PrepareDecode`, `AppendLookup`, `StopChain`, `StopCalibrated`, `FinishDecode`, `SpeculativeVerifier::VerifyStep` | `./src/models/qwen38_flash_next/engine.cpp:801-847,849-1012,1015-1119`; `./src/cli/serve/serve.cpp:549-559`; `./src/core/speculative/speculative_verifier.hpp:264-346` | Flash-Next’s MTP decode supports prefix-head drafting, prompt lookup, confidence-based stops, and target verification. A separate `SpeculativeVerifier` supports DFlash-style target/draft verification for other model paths. | Server options set prompt lookup and speculative configuration (`serve.cpp:549-559,1166-1173`); `PrepareDecode` builds the MTP chain and `FinishDecode` verifies it. | Speculative decode tracks candidate logits, sampler state, and draft state; lookup proposals are appended to the chain and then verified. | Where should expert prefetch hook into the MTP and prompt-lookup chains while retaining rollback/verification semantics? |
| [VERIFIED IN TARGET] Conversation snapshot cache and state serialization | `--cache-ram-bytes`, `ContinuationCache::Lease::TryReserveSnapshot`, `SnapshotEventReasonName`, `Executor::WalkSnapshot`, `Executor::SaveSnapshot` | `./src/cli/serve/serve.cpp:585-588,1056-1059,1204-1206`; `./src/cli/serve/continuation_cache.cpp:562-599`; `./src/cli/serve/text_model_runner.cpp:207-240`; `./src/models/qwen38_flash_next/kernels/rocm/executor.cpp:2468-2519,2580-2625` | Bounds retained RAM snapshots and serializes session state. | Server config passes the byte cap into runner options; cache leases reserve/publish snapshots (`text_model_runner.cpp:721-836`). | Executor snapshots include recurrent convolution/state buffers, PLE history, full-attention K/V and QSA indexer keys; host/device transfers use `SnapshotTransfer`. Pending speculative batches cannot be snapshotted (`executor.cpp:2581-2588`). | Which snapshot regions should be included in the proposed append-only delta spill, and what state/version compatibility is required? |
| [VERIFIED IN TARGET] CLI parser | `ArgParser::AddOption`, `ArgParser::AddFlag`, `ArgParser::Parse` | `./src/cli/arg_parser.hpp:23-35,41-59,85-123,159-238`; `./src/cli/serve/serve.cpp:584-588,1055-1059,1073-1078` | Provides typed option/flag registration and parsing; command handlers register their own options. | `llm_parser.Parse` parses server LLM options (`serve.cpp:1073-1078`). | Option values are bound to command-owned fields; no global option registry is shown in these excerpts. | Should `--memory-budget-gb` be exposed on `serve`, `prompt`, or both? |
| [REFERENCE IMPLEMENTATION] PLE SSD reads and `--ple-shelf` | `llama_lazy_reader`, `llama_ple_shelf_io`, `llama_ple_shelf::plan`, `--ple-shelf` | `../Flash-next-ssd/src/llama-ple-shelf-io.h:3-20,37-63`; `../Flash-next-ssd/src/llama-ple-shelf-io.cpp:62-82,160-220`; `../Flash-next-ssd/src/llama-lazy-reader.cpp:43-68,70-74`; `../Flash-next-ssd/common/arg.cpp:2941-2966`; `../Flash-next-ssd/docs/flashnext/settings.md:190-201` | Reads the lazy PLE table from disk; default `auto` shelf is 128 MiB. Reference docs describe the table as 28.8 GB **in that quant**—the spec’s 51B size is not corroborated by this reference documentation. | `llama_model_base::add_lazy_reader` installs the shelf-backed lazy reader (`../Flash-next-ssd/src/llama-model.cpp:2077-2083`); graph inputs call `begin`/`wait` (`../Flash-next-ssd/src/llama-graph.cpp:102-123`). | Shelf rows are held in a fixed buffer; reference macOS reads use `F_NOCACHE` without an alignment rule for 90-byte rows, and the shelf buffer is 16 KiB-aligned and mlocked (`llama-ple-shelf-io.cpp:62-82`). | Which exact artifact/quant is meant by “51B PLE,” and should the target shelf hold raw or compressed rows? |
| [REFERENCE IMPLEMENTATION] Expert slot cache and in-flight protection | `llama_moe_stream_layer`, `pick_victim_locked`, `reserve_slot_locked`, `worker_loop` | `../Flash-next-ssd/src/llama-moe-stream.h:30-43,49-54`; `../Flash-next-ssd/src/llama-moe-stream-layer.h:110-136`; `../Flash-next-ssd/src/llama-moe-stream.cpp:310-363`; `../Flash-next-ssd/src/llama-moe-stream-io.cpp:154-167,261-268` | Streams experts into **per-layer** slots. Victim selection excludes `LOADING` slots and kept guesses; last slab completion publishes `RESIDENT`. The requested single shared LRU pool across all layers was **NOT FOUND**; `pure_lru` is optional, while default victim ranking considers route hotness and then LRU. | Remap/lookahead reserve slots; `worker_loop` performs reads and completion accounting. | State is guarded by the manager mutex; `slot_pending` and generation checks prevent consumers seeing partially filled or stale slots. Reference docs explicitly say concurrent contexts for one streamed model are unsupported (`llama-moe-stream.h:38-43`). | How should the target share capacity across layers and safely support its concurrent decode sessions? |
| [REFERENCE IMPLEMENTATION] Prefill “Reading Room” and lending | `llama_moe_room`, `llama_moe_room_size`, `llama_moe_room::begin_ubatch_locked`, `lend_take_back_locked` | `../Flash-next-ssd/src/llama-moe-room-size.h:6-23`; `../Flash-next-ssd/src/llama-moe-room-size.cpp:94-152`; `../Flash-next-ssd/src/llama-moe-room.cpp:142-174`; `../Flash-next-ssd/src/llama-moe-room-lend-ops.cpp:1-22,114-132`; `../Flash-next-ssd/src/llama-moe-room-lend.h:102-108` | A configurable ring/belt fetches missing books for sufficiently large read-in batches. Auto sizes the room to 1.25 floors; lending reuses the belt for evicted books between read-ins. Reference comments report about 1.44 GiB for 48 floors, rather than a universal fixed 1.4 GiB. | `llama_moe_room::take` selects batches; `begin_ubatch_locked` takes the belt back and schedules room reads (`llama-moe-room.cpp:120-174,208-253`). | Belt parts are returned after GPU use; lending waits for in-flight copies/restores before room reads reuse the storage (`llama-moe-room-lend-ops.cpp:10-22,114-132`). | How should Gufo’s HIP stream completion replace the reference scheduler/Metal part hand-back protocol? |
| [REFERENCE IMPLEMENTATION] Lookahead and async I/O synchronization | `llama_moe_stream_lookahead_run`, `llama_moe_stream_lookahead_issue`, `worker_loop` | `../Flash-next-ssd/src/llama-moe-stream-lookahead.cpp:55-89,96-103,208-239`; `../Flash-next-ssd/src/llama-moe-stream-io.cpp:100-129,134-145,261-268`; `../Flash-next-ssd/src/llama-moe-stream-layer.h:61-100` | Prefetches one and optionally two floors ahead; the L+2 path predicts from L’s input and can use a trained guesser. This is a concrete two-floor lookahead, not evidence of a target-specific N+2 predictor. | The remap callback triggers the lookahead; I/O workers process queued demand, room, and speculative work. | Separate queues prioritize demand over room and speculative reads; generation and pending-count checks gate slot publication. | Can Gufo’s router/MTP output predict expert IDs early enough, or is a fallback router-based two-layer predictor needed? |
| [REFERENCE IMPLEMENTATION] Zero-swap watchdog / memory brake | `tools/flashnext/start.sh` preflight; manual swap guidance | `../Flash-next-ssd/tools/flashnext/start.sh:81-90,145-153`; `../Flash-next-ssd/docs/flashnext/memory-sizing.md:19-30`; `../Flash-next-ssd/docs/Qwen3.8-Flash-Next.md:63-77` | An automated zero-swap runtime watchdog or hard memory brake was **NOT FOUND**. The launcher prevents a second server and advises on compressed memory/wired limits; docs instruct operators to check `sysctl vm.swapusage` and keep it at zero. | Launcher preflight runs before the server command. | The documented zero-swap control is operator observation, not a running process-level brake. | What OS-level telemetry is available on Windows and Linux for a true runtime swap/memory brake? |

## Section 1.3: Architectural Mismatches

1. **Different runtime and memory ownership.** The target builds C++20 HIP for `gfx1151` (`./CMakeLists.txt:15-17,60-89`). Flash-next-ssd’s streaming path is explicitly a GGML/Metal expert cache (`../Flash-next-ssd/src/llama-moe-stream.h:30-43`; `../Flash-next-ssd/src/llama-moe-room.cpp:24-49`). Its buffer allocation, host-pointer access, graph ops, and completion protocol cannot be treated as HIP equivalents without verification.

2. **Gufo currently allocates whole expert tensors.** `Uploader::Copy` allocates `t.SizeBytes()` with `hipMalloc` and copies the tensor (`./src/models/qwen38_flash_next/kernels/rocm/device_model.cpp:152-184`). The Flash reference instead represents one expert slab per slot per layer (`../Flash-next-ssd/src/llama-moe-stream-layer.h:23-30,110-136`). This is a loader and dispatch architecture change, not just a cache policy change.

3. **4096-byte expert-start alignment is NOT FOUND for actual model files.** The GGUF reader defaults alignment to 32 bytes, accepts file metadata alignment, and validates tensor offsets against that alignment—not 4096 (`./src/core/gguf_reader.hpp:245-247`; `./src/core/gguf_reader.cpp:683-688,755-769`). Gufo records a tensor’s shard file offset (`./src/models/qwen38_flash_next/weights.cpp:99-118`) and computes an expert’s in-tensor address as `data + RowBytes() * rows * e` (`./src/models/qwen38_flash_next/weights.hpp:55-63`). Thus an individual expert’s file offset is:

   ```text
   O_e = tensor_file_offset + e * (RowBytes(type) * tensor_rows)
   ```

   `RowBytes` is format-dependent (`./src/models/qwen38_flash_next/weights.cpp:247-253`), including IQ4_XS and Q4_K geometries (`weights.cpp:19-44`). No `.gguf` artifacts were found in the target checkout, so the actual IQ4_XS/Q4 expert offsets and their modulo-4096 values are **NOT FOUND**. Do not assume page alignment. Unbuffered reads require empirical confirmation and, where needed, aligned-window/sub-page offset handling. Gufo’s existing Linux PLE path already rounds direct-read windows to 4096 bytes and offsets into the aligned buffer; its Windows PLE path deliberately uses cached random reads (`./src/models/qwen38_flash_next/ngram.cpp:98-110,173-216`).

4. **The reference PLE shelf’s I/O assumptions differ.** Flash-next-ssd documents `F_NOCACHE` reads for 90-byte rows on macOS and explicitly says Linux `O_DIRECT` would require aligned whole-block reads (`../Flash-next-ssd/src/llama-ple-shelf-io.cpp:62-82`; `../Flash-next-ssd/src/llama-ple-shelf-io.h:7-16`). That is distinct from Gufo’s Windows cached-random PLE reader and Linux direct-read path.

5. **Pool and carousel sizing differ from the spec’s simplified description.** Flash-next-ssd uses per-layer expert slots, not one global shared slot pool (`../Flash-next-ssd/src/llama-moe-stream-layer.h:110-136`). Its room is size-resolved from a configurable floor count and batch threshold (`../Flash-next-ssd/src/llama-moe-room-size.h:6-23`; `../Flash-next-ssd/src/llama-moe-room-size.cpp:67-132`), and its reported ~1.44 GiB is a specific 48-floor reference observation (`../Flash-next-ssd/src/llama-moe-room-lend.h:102-108`).

6. **The “zero-swap watchdog” is not a reference runtime feature.** Flash-next-ssd provides manual swap guidance and launcher preflight notes, not a continuously enforced zero-swap brake (`../Flash-next-ssd/docs/flashnext/memory-sizing.md:19-30`; `../Flash-next-ssd/tools/flashnext/start.sh:145-153`).

7. **The target already has stateful snapshot machinery.** Gufo serializes recurrent state, PLE history, K/V, and QSA indexer keys (`./src/models/qwen38_flash_next/kernels/rocm/executor.cpp:2468-2519`). An append-only SSD delta format must preserve compatibility with this state layout and speculative rollback behavior; the Flash expert-streaming implementation does not provide that snapshot format.

## Section 1.4: File Modification Plan

**[PROPOSED DESIGN] Existing target files planned for modification:**

- Build and CLI wiring: `CMakeLists.txt`, `src/cli/serve/serve.cpp`, `src/cli/prompt/prompt.cpp`, `src/cli/serve/inference_backend.hpp`, `src/cli/serve/inference_backend.cpp`.
- Model options, file ranges, and PLE configuration: `src/models/qwen38_flash_next/engine.hpp`, `src/models/qwen38_flash_next/engine.cpp`, `src/models/qwen38_flash_next/weights.hpp`, `src/models/qwen38_flash_next/weights.cpp`, `src/models/qwen38_flash_next/ngram.hpp`, `src/models/qwen38_flash_next/ngram.cpp`.
- Device-resident expert dispatch: `src/models/qwen38_flash_next/kernels/rocm/device_model.hpp`, `src/models/qwen38_flash_next/kernels/rocm/device_model.cpp`, `src/models/qwen38_flash_next/kernels/rocm/executor.hpp`, `src/models/qwen38_flash_next/kernels/rocm/executor.cpp`, `src/models/qwen38_flash_next/kernels/rocm/batch.cpp`, `src/models/qwen38_flash_next/kernels/rocm/kernels.hpp`, `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp`, `src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.hip.cpp`, `src/models/qwen38_flash_next/kernels/rocm/mmq/moe_wave64.hip.cpp`.

**[PROPOSED DESIGN] New implementation and test files:**

- `tests/spike_ssd_gtt_dma.cpp`
- `src/storage/ssd_batch_reader.hpp`, `src/storage/ssd_batch_reader.cpp`
- `src/storage/memory_budget_planner.hpp`, `src/storage/memory_budget_planner.cpp`
- `src/storage/expert_cache_manager.hpp`, `src/storage/expert_cache_manager.cpp`
- `tests/storage/ssd_batch_reader_test.cpp`
- `tests/storage/memory_budget_planner_test.cpp`
- `tests/storage/expert_cache_manager_test.cpp`

Snapshot-disk-spill files are deferred to Phase 5 per the appended decision; no snapshot-spill files are in the Phase 0–4 modification plan.

`MAPPING.md` is the approval-gated report copy specified by Step 1.

### Appended Architectural Decisions

1. **Speculative Lattice / Multi-Token Candidate Prefetching (CRITICAL)**

   Prefetch experts for the entire speculative candidate batch/lattice, not just single tokens. Wire the lookahead prefetcher into both speculative verification paths identified in Section 1.2, Row 5:

   a. Qwen3.8-Flash-Next’s native speculative loop in `src/models/qwen38_flash_next/engine.cpp` (`Session::PrepareDecode`, `AppendLookup`, `StopChain`, and `StopCalibrated`) where candidate token chains/trees are assembled before target verification.

   b. The DFlash speculative verifier in `src/core/speculative/speculative_verifier.hpp` (`SpeculativeVerifier::VerifyStep`) when used.

   Operate in two stages during each speculative verification step:

   - **Stage A — Pre-Verification Token-Prior Warmup at PrepareDecode:** Once candidate token IDs `[T_1 ... T_k]` are finalized, compute Layer 0/1 routing (and any cached token-to-expert priors) across all `k` candidate tokens, pin their slots as `IN_FLIGHT_PINNED`, and fire an initial `SsdBatchReader` burst before Layer 0 begins.
   - **Stage B — Batched Multi-Token L+1 / L+2 Hidden-State Lookahead inside `Executor::MoeBatch`:** As the `k`-token verification batch executes Layer L, run the L+1 and L+2 lookahead router across the hidden states of all `k` speculative tokens simultaneously, deduplicate the union of required experts for Layers L+1/L+2, pin their slots, and stream them via `SsdBatchReader` while Layer L computes.

2. **Per-Layer Slot Pool + Indirection**

   Follow `../Flash-next-ssd`’s per-layer slot-pool design, with route-hotness plus LRU victim selection and an `expert_id -> slot_id` indirection table for `Executor::Experts` and `Executor::MoeBatch`.

3. **Native PLE Integration**

   Integrate the 128 MiB `--ple-shelf` directly into Gufo’s existing `NgramTable` in `src/models/qwen38_flash_next/ngram.cpp`.

4. **Defer Snapshot Disk Spill to Phase 5**

   Keep Phases 0–4 strictly focused on `SsdBatchReader`, `--memory-budget-gb`, `--ple-shelf`, the pinned expert cache plus carousel, and speculative batch/lattice prefetching. Defer snapshot disk spill to Phase 5.

### Phase 0 Hardware & API Verification Spike Results (2026-10-17)

**Host and artifact:** The probe ran on AMD Radeon(TM) 8060S Graphics, `gfx1151`, with TheRock 10.0.0. The fixture was a 256 MiB temporary file on `C:\`; Windows reported 512-byte logical sectors, and `Get-Disk`/`Get-PhysicalDisk` identified the volume as a Sandisk PC SN5100S NVMe SSD. Reads used `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`; each successful allocation path ran three 256 MiB reads.

| Allocation path | Direct read | HIP checksum of all read bytes | 4 KiB sub-page probe | Aggregate throughput |
|---|---|---|---|---:|
| `hipHostMalloc(..., hipHostMallocMapped)` | PASS | PASS | PASS | 5.47292 and 5.50557 GiB/s (each 805,306,368 bytes / 137.038 and 136.226 ms) |
| `hipHostMalloc(..., hipHostMallocUncached)` | NOT AVAILABLE: HIP returned `hipErrorInvalidValue` (1) at allocation | NOT RUN | NOT RUN | NOT MEASURED |
| `VirtualAlloc + hipHostRegister(..., hipHostRegisterMapped)` | PASS | PASS | PASS | 5.13038 and 5.29623 GiB/s (each 805,306,368 bytes / 146.188 and 141.610 ms) |

Both successful paths exceed 3.5 GB/s in two runs. For each candidate, `ReadFile` wrote directly into the candidate host pointer; the next operation was a HIP checksum kernel over its device-visible pointer, with no application CPU memcpy between the read and kernel. This measures Windows direct-I/O throughput and immediate HIP visibility; it is not an instrumented trace of the storage device’s DMA engine.

The unaligned probe used `O = 4219`, `N = 65519`: `align_down(O,4096) = 4096`, `delta = 123`, and `align_up(delta + N,4096) = 69632`. The overlapped direct read into the aligned base and HIP checksum at `device_ptr + delta` both passed. **Phase 0 decision:** use `hipHostMallocMapped` as the preferred Windows allocation path; retain `VirtualAlloc + hipHostRegisterMapped` as a viable alternate; do not use `hipHostMallocUncached` on Windows. The installed header documents `hipHostMallocUncached` as Linux-only on supported AMD devices (`C:\Users\danie\AppData\Local\Gufo\dependencies\therock-10.0.0\include\hip\hip_runtime_api.h:932-938`). The header `<hip/hip_file.h>` remains **NOT FOUND** in both inspected Windows SDKs; Linux remains uninspected.

The target compiled and linked. `tools/windows/build.ps1` then exited during post-build OpenMP runtime staging because it could not locate the local Visual Studio OpenMP runtime (`./tools/windows/build.ps1:148-153`): CMake selected `VC/Tools/MSVC/14.44.35207/lib/x64/libomp.lib`, while the installed redist directory is `VC/Redist/MSVC/14.44.35112`. The produced `build/gpu-test/spike_ssd_gtt_dma.exe` ran and reported the results above. The staging issue is separate from target compilation and spike execution.

The new `tests/spike_ssd_gtt_dma.cpp` passes `clang-format 19.1.5` in file-scoped dry-run mode. The repository-wide `python tools/ci/check-format.py` reports pre-existing formatting violations in `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp:3683,3686` and `src/models/qwen/hip/kernels/prefill_quant_gemm.hpp:139`; neither file was modified for this spike.

### Phase 1 Cross-Platform SsdBatchReader (2026-10-17)

**[PROPOSED DESIGN implemented]** Added `src/storage/ssd_batch_reader.hpp` and `.cpp`, a reusable `gufo_storage` CMake library, and `tests/storage/ssd_batch_reader_test.cpp`. `SsdBatchReader` accepts arbitrary file ranges and caller-owned 4096-byte-aligned destinations, rounds the read window down/up, and exposes the payload sub-page offset through `SsdReadRequest::Payload()`. Windows uses `CreateFileW` with `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED | FILE_FLAG_SEQUENTIAL_SCAN`, an IOCP completion port, and a worker pool. Linux uses `O_DIRECT`, `pread`, and a worker pool. Short final-sector reads are accepted only when all requested payload bytes arrived.

**[VERIFIED IN TARGET] Windows validation:** `ssd_batch_reader_test` passed in `build/gpu-test`, reading 64 unaligned mock expert blocks concurrently into `hipHostMallocMapped` slots and checking every block’s checksum with HIP kernels on the stream. The fixture ends on a non-4096-byte boundary to exercise final-block padding. The final CTest command and result were:

```powershell
ctest --test-dir build/gpu-test -R '^ssd_batch_reader_test$' --output-on-failure --no-tests=error
# 1/1 ssd_batch_reader_test passed
```

The full `gufo` target built and linked in the `gpu-test` preset (232/232 build steps). `build/gpu-test/gufo.exe --help` and `build/gpu-test/gufo.exe diagnose` both passed; diagnostics reported overall `[PASS]` on the Windows gfx1151 host. The build script still exits after compilation during OpenMP runtime staging because its CMake-selected MSVC runtime version (`14.44.35207`) has no exact-version redist directory; this is the same existing staging mismatch noted for Phase 0. `clang-format 19.1.5 --dry-run --Werror` passes for both new storage files and the new test. The repository-wide format check still reports only the two pre-existing files named above.

Linux implementation was added but **NOT BUILT OR RUN** on this Windows host.

### Phase 2 Memory Planner and PLE Shelf (2026-10-17)

**[VERIFIED IN TARGET] Implemented:** added `src/storage/memory_budget_planner.{hpp,cpp}`. `PlanMemoryBudget` validates finite positive GiB input, uses checked fixed-byte summation, computes the remaining expert budget without underflow, caps the planned expert cache at the routed-expert weight total, and formats the startup breakdown. `DeviceModel` now reports measured backbone, MTP drafter, and routed-expert upload bytes. `Model::Load` combines those measurements with vision bytes, actual PLE cache bytes, per-session state reservation times `decode_concurrency`, and deferred scratch, then stores and logs the plan when `memory_budget_gb > 0`.

The `--memory-budget-gb <GB>` option is wired into `gufo serve llm` and `gufo prompt`. `--ple-shelf` opts into a 128 MiB PLE row cache in the existing `NgramTable`; without that flag, the previous 8 MiB cache remains. Explicit shelf capacity is clamped to available table rows and `NgramTable::CacheBytes()` accounts for both row storage and cache-entry metadata.

**[VERIFIED IN TARGET] Focused validation:** `memory_budget_planner_test` and `qwen38_flash_next.ngram` passed in `build/gpu-test`. `gufo serve llm --help` and `gufo prompt --help` show both options. The full `gufo` target compiled and linked; its `--help` and `diagnose` smoke checks passed. The build wrapper again reports only its post-build OpenMP runtime staging mismatch (`14.44.35207` compiler path versus available `14.44.35112` redist).

**[PROPOSED DESIGN / PHASE 3 DEPENDENCY]** The planner’s `expert_cache_bytes` is currently a computed capacity recommendation. The existing `DeviceModel::Upload` still uploads all routed expert tensors; per-layer slots, indirection, and the allocation brake that enforces this capacity are Phase 3 work. A real-model startup plan was not exercised because no GGUF model artifact is present in this checkout. Do not treat the Phase 2 plan as an active bound on expert allocations until Phase 3 wires it into the expert cache.

The planner/PLE changes built into `build/gpu-test`; CTest `memory_budget_planner_test` and `qwen38_flash_next.ngram` both passed. The `gufo serve llm --help` and `gufo prompt --help` smoke outputs include `--memory-budget-gb` and `--ple-shelf`. `gufo diagnose` remains `[PASS]`. The same post-build OpenMP runtime staging mismatch remains after successful compilation/linking.

### Phase 3 Budgeted Expert Cache and Executor Indirection (2026-10-17)

**[IMPLEMENTED]** A positive `--memory-budget-gb` now defers routed expert uploads while keeping backbone and MTP weights resident. After the memory plan, `ExpertCacheManager` allocates packed, page-aligned `hipHostMallocMapped` arenas, with exact `expert_bytes` slot strides and bounded tail margins. `SsdBatchReader` reads missing experts from their recorded GGUF shard offsets into those mapped slots; unaligned payloads use aligned per-worker scratch. Persistent pinned host buffers capture router IDs, the host mapping table resolves expert IDs to cache slots, and the slot-ID transfer back to the existing `s_.ids` scratch keeps HIP MoE kernel interfaces unchanged. Each cache chunk waits for a HIP event before releasing its slots. Prefill and batched MoE dispatches split rows so each chunk's maximum active set is bounded by `floor(cache_slots / experts_used)`.

The budget's fixed scratch reservation includes the Executor's preallocated GPU scratch plus its bounded lazy batched-logit buffer, in addition to the previously recorded backbone, MTP drafter, PLE shelf, vision, and per-session reservations. This avoids presenting those existing allocations as capacity available to expert slots. No HIP kernel source/signature/stride file was changed.

**[VERIFIED IN TARGET] Focused validation:** `expert_cache_manager_test`, `ssd_batch_reader_test`, `memory_budget_planner_test`, `qwen38_flash_next.ngram`, `qwen38_flash_next.iq_experts_ops`, `qwen38_flash_next.moe_ids_ops`, and `qwen38_flash_next.moe_epilogue_ops` passed. The reader fixture exercised 64 concurrent unaligned payloads and compared every payload byte plus HIP checksums; the cache fixture checked every byte in selected packed slots, pin/release behavior, and LRU slot indirection. `gufo_qwen38_flash_next` and the full `gufo` executable compiled and linked. At the time of this validation, post-build OpenMP runtime staging failed; see the Windows build follow-up below.

**[VERIFIED IN TARGET] 32 GiB smoke run:** Swift-1.5 IQ4_XS, the shared Q8_0 MTP sidecar, `--speculative mtp`, greedy sampling, an 8-token output limit, and prompt `Reply with exactly OK.`. The startup plan reported 4.996 GiB backbone, 2.583 GiB drafter, 0.006 GiB PLE shelf, 0.384 GiB KV/state, 2.781 GiB Executor scratch, and 21.251 GiB expert cache (60.938 GiB total routed expert weights); fixed allocations fit within 32 GiB. The exact deterministic command uses `--raw --speculative mtp --mtp-model <shared-Q8_0-sidecar> --max-tokens 8 --seed 459`; the output trace is `434d749ab5504962fdc020206a41ab2c427b8c6b7f1e459d2e4c9228933f97d7`. The trace was reproduced on the Phase 3 baseline and Phase 4 at 32 GiB and 20 GiB. The previously recorded `adc0dfc7…` value was from an invocation whose exact flags were not retained and is superseded by this reproducible baseline/candidate comparison. This verifies output consistency and a basic MTP generation path, not a performance result.

A prior 14 GiB run left 3.251 GiB for the expert cache and loaded in 34.51 s, but its exact command was not retained. Do not use its historical trace as the canonical Phase 3/Phase 4 comparison; the reproducible 8-token command and trace are recorded above.

**[QUALITY VALIDATION PENDING]** No matched-token full-logit/perplexity comparison has been run for the Swift-1.5 artifact against a pre-change baseline. The successful generation and focused kernel/cache tests do not replace that check. Linux build/runtime validation also remains outstanding.

### Phase 4 synchronous Phase 3 baseline (2026-10-18)

**[VERIFIED IN TARGET]** Before enabling asynchronous prefetch, ran the telemetry-enabled synchronous Phase 3 binary on Windows/gfx1151 with the Swift-1.5 IQ4_XS target and shared Q8_0 MTP sidecar. The deterministic prompt is passed raw and tokenized to exactly 64 tokens by the artifact tokenizer:

```text
The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. [repeat three times] Reply with exactly OK.! More
```

Both runs use greedy decoding, seed 459, MTP, 64 generated-token maximum, `--ssd-stats`, and the same prompt. The Windows command environment explicitly sets `HIP_PATH` and `ROCM_PATH` to TheRock 10.0.0 and prepends its `bin` directory to `PATH`.

| Budget | Load (s) | Prefill (tok/s) | Decode (tok/s) | Peak working set (GiB) | OS swap delta (MiB) | Expert requests | Cache hit % | Prefetched-in-time % | Hard stall % / ms | PLE hit % | SSD bytes read | NVMe GiB/s | Session bytes written | 64-token output trace SHA-256 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 GiB | 38.31 | 6.17 | 6.65 | 35.823 | +793.762 | 35,316 | 71.064 | 0.000 | 28.936 / 11,974.781 | 13.930 | 27,331,212,944 | 1.681 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 20 GiB | 36.14 | 4.99 | 4.49 | 23.766 | +557.680 | 40,991 | 57.213 | 0.000 | 42.787 / 19,477.911 | 13.930 | 46,908,909,200 | 2.221 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |

These are short, single-request synchronous baselines for the requested 64-token prompt, not production pp2048/tg128 performance results. The peak working set is an OS process measurement and can exceed the configured device-memory budget; it is reported separately from the Gufo planner reservation. “Hard stall %” is the share of unique expert requests that required a blocking read. The separate 8-token smoke prompt has trace hash `434d749a…`. Token hashes are output-consistency checks, not full-logit/perplexity equivalence.

### Phase 4 asynchronous lookahead results and trace equivalence (2026-10-18)

**[VERIFIED IN TARGET]** Repeated the same raw 64-token prompt, Swift-1.5 IQ4_XS model, shared Q8_0 MTP sidecar, greedy sampling, and seed 459 with asynchronous Stage A/Stage B prefetch enabled. Both synchronous and asynchronous measurement runs used `--ssd-stats --verbose`; Phase 4 explicitly set `--cache-prior-bias 0`, which matches the synchronous default. The no-I/O arithmetic path remains unchanged. Both generated-output traces match the Phase 3 baseline at 32 GiB and 20 GiB.

| Budget | Phase | Load (s) | Prefill (tok/s) | Decode (tok/s) | Peak working set (GiB) | OS swap delta (MiB) | Expert requests | Cache hit % | Resident hits | Prefetch requests | Prefetched in time % / hits | Late waits | Cold misses | Hard stall % / ms | PLE hit % | SSD bytes read | NVMe GiB/s | Session bytes written | 64-token trace SHA-256 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 GiB | Phase 3 synchronous | 38.31 | 6.17 | 6.65 | 35.823 | +793.762 | 35,316 | 71.064 | — | 0 | 0.000% / 0 | 0 | — | 28.936 / 11,974.781 | 13.930 | 27,331,212,944 | 1.681 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 32 GiB | Phase 4 async | 16.69 | 7.83 | 10.04 | 35.872 | +609.742 | 35,365 | 75.555 | 25,214 | 1,952 | 77.152% / 1,506 | 35 | 8,610 | 24.445 / 10,031.927 | 13.930 | 28,248,573,584 | 1.752 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 20 GiB | Phase 3 synchronous | 36.14 | 4.99 | 4.49 | 23.766 | +557.680 | 40,991 | 57.213 | — | 0 | 0.000% / 0 | 0 | — | 42.787 / 19,477.911 | 13.930 | 46,908,909,200 | 2.221 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 20 GiB | Phase 4 async | 13.75 | 6.22 | 5.57 | 23.895 | +170.090 | 42,386 | 64.977 | 24,672 | 3,829 | 74.928% / 2,869 | 105 | 14,740 | 35.023 / 17,417.259 | 13.930 | 49,663,649,424 | 1.740 | 0 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |

The final 8-token Phase 3/Phase 4 trace is `434d749ab5504962fdc020206a41ab2c427b8c6b7f1e459d2e4c9228933f97d7` for the exact prompt `Reply with exactly OK.` and seed 459. The 64-token trace is the `c71e47bb…` value shown above. For both checks, the canonical Gufo draft option is `--mtp-model` (not `--draft-model`). The model and sidecar used were:

```powershell
$model = 'C:\Users\danie\AppData\Local\Gufo\models\swift-1.5-qwen38-flash-next-iq4-xs\IQ4_XS\Swift-1.5-Qwen3.8-Flash-Next-IQ4_XS-00001-of-00003.gguf'
$mtp = 'C:\Users\danie\.cache\huggingface\hub\models--unsloth--Qwen3.8-Flash-Next-GGUF\snapshots\38bb39ee97821de2c9009abb7e93950eec396e66\MTP\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'
$short = 'Reply with exactly OK.'
$long = 'The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. Reply with exactly OK.! More'

# Substitute --memory-budget-gb 20 for 32 to repeat at the smaller budget.
build\gpu-test\gufo.exe prompt --model $model --raw --speculative mtp --mtp-model $mtp `
  --memory-budget-gb 32 --verbose --max-tokens 8 --seed 459 --prompt $short
build\gpu-test\gufo.exe prompt --model $model --raw --speculative mtp --mtp-model $mtp `
  --memory-budget-gb 32 --cache-prior-bias 0 --ssd-stats --verbose `
  --max-tokens 64 --seed 459 --prompt $long
```

The 8-token run was repeated with `--ssd-stats`; the trace was unchanged. The Phase 4 64-token benchmark table reports the telemetry-enabled runs; their SSD reads use Windows `FILE_FLAG_NO_BUFFERING`, so those expert transfers bypass the Windows OS page cache. Phase 4 may read extra speculative bytes when lookahead experts are not selected by the true next-layer router; this is expected background I/O, traded for reduced synchronous stall time. These short single-request results are not production pp2048/tg128 measurements, and token hashes do not replace matched-logit/perplexity validation.

### Phase 4 validation status (2026-10-18)

**Focused tests:** 7/7 passed: `ssd_stats_test`, `expert_cache_manager_test`, `ssd_batch_reader_test`, `qwen38_flash_next.ngram`, `qwen38_flash_next.moe_epilogue_ops`, `qwen38_flash_next.moe_ids_ops`, and `qwen38_flash_next.iq_experts_ops`.

**Full GPU selection:** 75 tests were selected; 45 passed, 28 were skipped because optional model artifacts were absent, and 2 failed (73/75 did not fail). The two failures were `minimax_h3_dit_analytic` and `qwen_ssm_ops_test`, both unrelated to Phase 4. On Windows `gfx1151`, those upstream failures can fault the MES compute ring; subsequent HIP allocations then fail with `unspecified launch failure`. Exclude these faulting tests on this host when running the GPU suite:

```powershell
ctest --test-dir build/gpu-test -L gpu -E "minimax_h3_dit_analytic|qwen_ssm_ops_test" --output-on-failure --no-tests=error
```

The excluded full-suite rerun was not performed. The focused storage/Qwen3.8 tests and model trace checks above completed before the MES-ring fault.

### Phase 5 streamed-prefill and I/O scheduling (2026-10-18)

**Reference review:** `../Flash-next-ssd/src/llama-moe-room.cpp` builds a floor plan from the whole read-in router set, reserves its missing union, then pipelines reading-room fills while the GPU consumes the same floor. Its worker ordering is demand, room, then speculative; queued guesses can be promoted, while stale queued room work is dropped at hand-back. Gufo's regular `ForwardBody` path calls `Moe` → `StreamedMoeRows`; the former `MoeBatch` path also limited streamed dispatch to eight rows. The new resolver scans actual router IDs, selects the largest contiguous union that fits the layer's unpinned slots, resolves that union once, then keeps the previous compute-group sizes. `MoeBatch` passes its complete routed batch into this path. This avoids using `floor(slots / top_k)` as the prefill window while preserving the original eight-row batched group and the regular path's cache-bounded group.

`ExpertCacheManager::Resolve` now pins all current-window residents before selecting victims, counts each expert's route frequency, and chooses cold/old victims without evicting a hot expert needed later in the same window. A demand hit promotes its queued per-shard prefetch batch; releasing a layer cancels its still-queued speculative batches. The host cache has separate demand and prefetch job lanes. Windows IOCP keeps eight completion workers and a model-wide 32-read/scratch budget spread across expert shard readers; it sorts by aligned file offset and coalesces overlapping padded ranges into one bounce-buffer read. The cache config sizes scratch for two adjacent expert slabs (about 32–64 MiB for this target's shard count). Linux retains its eight-worker `O_DIRECT` path and sorts read batches.

The current Qwen MTP cap is seven draft proposals per cycle plus the anchor (maximum verification width eight). `cap = min(request output budget, remaining context, executor verification capacity)`; optional prompt lookup fills remaining width after an MTP proposal. The 64-output candidate runs at both budgets reported 17 cycles, 55 MTP proposals, 47 accepted (85.45%); prompt lookup was off, so lookup proposals were zero. `prompt --verbose` now prints MTP/lookup proposal and acceptance counts. During verification, all proposed rows are run through target layers before rollback, so rejected tail rows have already contributed router assignments and cache demand; this cost is traded against the additional accepted-prefix opportunity. No memory-budget-specific draft cap was retained: a matched long-form acceptance/cap sweep did not complete after the HIP allocation fault below.

#### Matched 64-output control, Phase 4 baseline vs Phase 5 candidate

Same Windows gfx1151/TheRock 10.0.0 machine, Swift-1.5 IQ4_XS/Q8_0 MTP artifacts, raw 63-token prompt, greedy sampling, seed 459, 64-token output limit and `--ssd-stats`. Single run per cell; model load and filesystem/device warm state vary. The 8-token trace also remains `434d749ab5504962fdc020206a41ab2c427b8c6b7f1e459d2e4c9228933f97d7`; the 64-token trace is unchanged at `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d`.

| Budget | Version | Prefill (tok/s) | Decode (tok/s) | Effective NVMe (GiB/s) | Prefetched in time % | Hard stall ms | SSD bytes read | 64-token trace |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 GiB | Phase 4 | 7.921 | 11.160 | 1.775 | 76.275 | 9,867.548 | 28,170,908,608 | `c71e47bb…` |
| 32 GiB | Phase 5 | 9.993 | 11.469 | 2.563 | 77.241 | 8,722.453 | 28,159,734,720 | `c71e47bb…` |
| 40 GiB | Phase 4 | 8.330 | 11.996 | 1.820 | 76.857 | 9,434.749 | 26,839,004,096 | `c71e47bb…` |
| 40 GiB | Phase 5 | 10.061 | 12.484 | 2.584 | 77.819 | 8,208.794 | 26,818,950,080 | `c71e47bb…` |

Single-run changes: at 32 GiB, prefill +26.2%, decode +2.8%, NVMe throughput +44.4%, and prefetched-in-time +0.97 percentage points; at 40 GiB, prefill +20.8%, decode +4.1%, NVMe throughput +42.0%, and prefetched-in-time +0.96 points. `expert_requests` and cache-hit percentages are **not directly comparable across phases**: `Resolve` now batches a larger row union and counts each unique expert once per union. Direct expert reads use `FILE_FLAG_NO_BUFFERING` and bypass the Windows file cache.

The 40 GiB/44 GiB long-form baseline supplied for this pass reported 1.23 million expert requests, 262 GB read, 1.77 GiB/s and 37.6% prefetched in time. A reproducible raw 78-token prompt at 40 GiB, `--max-tokens 128`, MTP, seed 459 and prompt lookup off generated 123 tokens before EOS: prefill 8.938 tok/s, decode 12.396 tok/s, 1.849 GiB/s, 71.662% prefetched in time. The Phase 5 `--prompt-lookup` run for that workload failed while creating the session (`hipMalloc of 40960 bytes failed`) after the preceding successful GPU work; it produced no candidate timing or acceptance sample. **Do not run further GPU commands in this session:** the device returned a HIP allocation failure and may have entered the Windows gfx1151 MES-ring fault state. Long-form before/after, MTP+Lookup acceptance and cap sweeps remain unmeasured.

The seven focused checks passed after the window, hotness, IOCP-priority and range-coalescing changes: `ssd_stats_test`, `expert_cache_manager_test`, `ssd_batch_reader_test`, `qwen38_flash_next.ngram`, `qwen38_flash_next.moe_epilogue_ops`, `qwen38_flash_next.moe_ids_ops`, and `qwen38_flash_next.iq_experts_ops`. `gpu-test` and `release` `gufo` targets compiled successfully. The faulting MiniMax H3 analytic and Qwen SSM tests were not run in Phase 5.

The MTP controller remains capped at seven proposals plus its anchor (verification width at most eight). `PrepareDecode` chooses a draft width from the output budget, remaining context and executor capacity; when enabled, prompt lookup appends copied context tokens only into unused width. A target verification pass processes the complete proposed chain before rollback, so rejected tail rows still run router selection and contribute expert demand. The new verbose prompt counters separate MTP drafts/accepts from lookup drafts/accepts. On the 64-output no-lookup candidate, the recorded sample was 17 cycles, 55 MTP proposals, 47 accepted (85.45%). A long-form lookup/cap sweep was blocked by the HIP allocation fault; the cap was not adjusted without that evidence.

### Phase 5 matched 64-output measurement (2026-10-18)

One run per cell on Windows/gfx1151, TheRock 10.0.0, Swift-1.5 IQ4_XS and the shared Q8_0 MTP sidecar, raw 63-token input, greedy/seed 459, 64 output limit. The Phase 5 run uses the actual-union cache windows and IOCP priority/coalescing. Prefill and decode changes are single-run observations; SSD reads use `FILE_FLAG_NO_BUFFERING`.

| Budget | Phase | Prefill (tok/s) | Decode (tok/s) | Effective NVMe GiB/s | Prefetched in time % | Expert requests | Cache hit % | Cold misses | Hard stall ms | SSD bytes | 64-output trace SHA-256 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 GiB | Phase 4 | 7.921 | 11.160 | 1.775 | 76.275 | 35,761 | 75.988 | 8,552 | 9,867.548 | 28,170,908,608 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 32 GiB | Phase 5 | 9.993 | 11.469 | 2.563 | 77.241 | 29,527 | 70.993 | 8,548 | 8,722.453 | 28,159,734,720 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 40 GiB | Phase 4 | 8.330 | 11.996 | 1.820 | 76.857 | 34,191 | 76.035 | 8,164 | 9,434.749 | 26,839,004,096 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |
| 40 GiB | Phase 5 | 10.061 | 12.484 | 2.584 | 77.819 | 28,906 | 71.722 | 8,162 | 8,208.794 | 26,818,950,080 | `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d` |

The 8-output exact command also retains `434d749ab5504962fdc020206a41ab2c427b8c6b7f1e459d2e4c9228933f97d7`. Across the short runs, Phase 5 raises prefill +26.2%/+20.8%, decode +2.8%/+4.1%, effective NVMe +44.4%/+42.0%, and prefetched-in-time +0.97/+0.96 percentage points at 32/40 GiB. The `expert_requests` and cache-hit units change with the larger resolution windows: expert IDs repeated across more rows are now resolved/count once per window. Do not compare those counters as raw token-route assignments.

The user's prior 40/44 GiB long-form telemetry was 1.23M expert requests, 262 GB read, 1.77 GiB/s and 37.6% prefetched in time. The reproducible 78-token baseline prompt at 40 GiB/128-token maximum generated 123 tokens before EOS (prefill 8.938 tok/s, decode 12.396 tok/s, 1.849 GiB/s, 71.662% prefetched in time). Adding `--prompt-lookup` failed at session creation with `hipMalloc of 40960 bytes failed`; no long-form Phase 5 result or MTP+Lookup acceptance sample was collected. The system had just faulted during GPU allocation, so no further GPU commands were run.

The 64-output candidate acceptance sample is reproducible with the Phase 4 `$long` prompt and exact `--speculative mtp --mtp-model $mtp --max-tokens 64 --seed 459` settings above; the Phase 5 `--verbose` report shows 17 cycles, 55 proposals and 47 accepted (lookup off). The long-form baseline prompt and attempted lookup run were:

```powershell
$long78 = 'The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. Reply with exactly OK.! More. Batch expert union across tokens, and preserve the eight-row compute groups.'
build\gpu-test\gufo.exe prompt --model $model --raw --speculative mtp --mtp-model $mtp `
  --memory-budget-gb 40 --cache-prior-bias 0 --ssd-stats --verbose `
  --max-tokens 128 --seed 459 --prompt $long78
# Phase 5 lookup attempt added --prompt-lookup and failed at session hipMalloc before inference.
```

`SsdStats.expert_requests` is unique expert demand per `Resolve` window. Its count
therefore falls when repeated row assignments are deduplicated across a wider
window; use `expert route hits` in the manager for frequency, and compare cold
misses/bytes/latency alongside it rather than treating Phase 4/5 request totals
as the same unit.

Reproduce with the exact 64-token raw prompt and `--memory-budget-gb 32` or `20`:

```powershell
$therock = 'C:\Users\danie\AppData\Local\Gufo\dependencies\therock-10.0.0'
$env:HIP_PATH = $therock
$env:ROCM_PATH = $therock
$env:PATH = "$therock\bin;$env:PATH"
build\gpu-test\gufo.exe prompt --model $model --raw --speculative mtp --mtp-model $mtp `
  --memory-budget-gb 32 --ssd-stats --verbose --max-tokens 64 --seed 459 `
  --prompt 'The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. Reply with exactly OK.! More'
```

### Windows build and formatter follow-up (2026-10-18)

The CMake cache points to `VC/Tools/MSVC/14.44.35207/lib/x64/libomp.lib`; its LLVM OpenMP dependency is `libomp140.x86_64.dll`, not `vcomp140.dll`. The installed Visual Studio redist is `14.44.35112`. `tools/windows/build.ps1` now checks the exact toolset redist first, then tries installed `VC/Redist/MSVC/14.*` versions from newest to oldest until it finds the LLVM OpenMP DLL. A full `-Target gufo` build completed with exit code 0 (240/240 steps) and staged the DLL from `14.44.35112`; `gufo.exe --version` ran successfully.

Visual Studio's LLVM tools include `clang-format 19.1.5` at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\Llvm\bin\clang-format.exe`; TheRock does not include it. Changed C++ files now pass the repository format check. `python tools/ci/check-format.py` reports only the two pre-existing violations in `src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp` and `src/models/qwen/hip/kernels/prefill_quant_gemm.hpp`, neither of which was changed.

### Phase 6 lazy context, shared reader memory, and warm cache (2026-10-19)

**Implemented:** `SsdScratchPool` now owns one aligned pool shared by every expert-shard reader; leases span each overlapped read through completion/scatter, and demand waiters bypass queued prefetch waiters. The budget planner includes reader scratch, a 4,096-token initial session reservation, a continuation snapshot reserve, one-layer pageable migration staging, and `max(1 GiB, 5% of target)` safety margin. The planner caps expert capacity against WDDM's reported available bytes after the model/executor allocations are resident.

At model load, each streamed layer pre-warms slot experts `0..slots-1`; this load is reported separately as `expert_cache_prewarm bytes=... elapsed_ms=...` and does not enter inference `SsdStats`. Router resolution now accepts future-window expert hints and uses them when selecting cache victims. Session K/V and block-key storage starts at up to 4,096 tokens and grows geometrically; growth synchronizes/invalidate graphs, preserves live prefixes, and can shrink mapped expert arenas one layer at a time by staging the retained prefix to pageable memory, freeing the old arenas before allocating smaller ones. Out-of-core serving exposes only a single serial execution plan/state, and prefetch is drained around session snapshots.

**Focused validation:** after the user rebooted the Windows host, the seven approved tests passed on TheRock 10.0.0 / `gfx1151`: `memory_budget_planner_test`, `expert_cache_manager_test`, `ssd_batch_reader_test`, `qwen38_flash_next.ngram`, `qwen38_flash_next.iq_experts_ops`, `qwen38_flash_next.moe_ids_ops`, and `qwen38_flash_next.moe_epilogue_ops`. New fixtures cover shared scratch leases, pre-warm byte reporting, future-use victim protection, and free-before-allocate arena shrink. `gpu-test` and `release` `gufo` targets compiled and linked. A final conservative WDDM-check adjustment below was compile-verified after these tests; it does not change the covered test paths.

**40 GiB prompt/model checks:** load reported a 26.065 GiB expert-cache plan and pre-warmed 27,923,197,952 bytes in 6.7–6.8 s. The 40 GiB 78-token, 128-output MTP+prompt-lookup request completed (prefill 16.21 tok/s, decode 12.51 tok/s, 123 output tokens); this prompt produced zero lookup proposals, so it verifies successful session creation after the earlier 40,960-byte allocation failure, not a lookup hit. A separate 561-token repeated C++ coding prompt prefills at 43.01 tok/s. The 63-token matched prompt generated 64 tokens at 13.62 tok/s and reproduced the saved trace `c71e47bb7dccaf37a42ddba0ceda952bdaf95f41f17db02a8c43fff697ebc20d`.

**Serving and context growth:** `gufo serve llm` ran at 40 GiB with an 8,192-token context and a 4,275-token first turn. Both chat requests returned HTTP 200 on `plan=serial-c1`; the follow-up restored 4,270 cached tokens (`cache=memory`, 11.2 ms restore) and prefetched only the final 26 tokens. This crossed the 4,096-token initial session capacity and exercised snapshot replay.

**WDDM pressure follow-up:** the pre-reboot 80 GiB attempt used a retry loop that shrank the cache down to 17 slots/layer, then hit `unspecified launch failure` / device-lost status 75 during session-state copy. After reboot, the one-shrink version measured 3,515,740,160 WDDM free bytes, released 1,174,118,400 cache bytes, and projected 4,689,858,560 available bytes; the subsequent 8 MiB `hipMalloc` still failed with out of memory. The final code no longer credits projected bytes and trusts only DXGI's post-shrink reading, rejecting growth if WDDM does not confirm headroom. This safer guard compiles but has not been exercised at runtime. The 40 GiB context-growth/snapshot-serving case and the manager's standalone arena-shrink test passed.

**Remaining validation:** matched-token full-logit/perplexity validation is not yet available for the new allocation/cache lifecycle. The 80 GiB context-growth request did not complete under this host's WDDM/UMA pressure, and the post-shrink reporting guard remains runtime-unverified. No additional GPU runtime commands were run after the 8 MiB allocation failure. The local `clang-format 19.1.5` whole-repository check reports pre-existing differences across multiple files; all task-modified C++ files pass its direct formatting check, without reformatting unrelated sources.

### Phase 6.1 full-context reservation, LRU and Desk/Room prefill (2026-10-20)

**Implementation correction:** each session now allocates attention K/V, sparse-index keys and MTP K/V/block keys for the complete configured `--context` ceiling at `CreateSession`; its request-specific context remains the logical limit. Rollback rows are reserved once at session creation. Reset and rollback trimming retain buffers and only invalidate captured graphs. The byte planner reserves the complete configured session/snapshot footprint before expert-cache sizing; migration staging and geometric context/cache shrink paths are removed. If the fixed reserve does not fit, load reports the requested context and suggests `--context 65536` or `32768`.

QFN's device arena and routed-ID scratch are reserved and frozen before inference. The HIP allocation-site audit found `hipMalloc`/`hipHostMalloc` only in model/executor/session/cache creation paths; QFN growth after freeze fails rather than allocating. Runtime API interception was not added, so this is source-path verification rather than a measured zero-allocation counter.

`SsdStats` now separates Prefill/Decode request, hit, miss, stall, PLE, demand-read and prefetch-read counters. Pure LRU is the default victim policy; cumulative and decayed-hotness policies remain available to the host replay tool. `GUFO_EXPERT_ROUTE_TRACE` writes JSONL router assignments, actual Resolve windows/passes, protected desk IDs and cache capacity. `tools/bench/expert_cache_replay.py` compares LRU, hotness, prefill-reset/decayed hotness and greedy packing. Replay models demand routing and desk reservation, but excludes prefetch completions/timing and token-prior predictions.

For long prefill windows above 64 rows, the executor estimates Room capacity as `ceil(max(top_k, mean_unique - max_unique/2))`, retains the most frequently routed experts in protected Desk slots, and reserves the rest for Room passes. Each pass masks inactive assignments with negative IDs, recomputes counts/tile hints from remapped slot IDs, writes owned rows at their original `(token, route)` indices, and runs the epilogue once per original compute group. Slot leases and pinned ID/tile staging are held through stream completion. While a Desk/Room pass executes, bounded asynchronous prefetch reads the next Room group into the reserved slots.

**Build and focused validation:** `gpu-test` and production `release` `gufo` targets build with TheRock 10.0.0 / `gfx1151`. The eight focused storage/Qwen tests pass, including a new masked gate/up and down-projection fixture that compares split passes bit-for-bit with the complete assignment matrix and checks one post-pass epilogue; five route-replay Python tests pass. The 8-token `434d749a…` and 64-token `c71e47bb…` traces match the recorded controls. A 32 GiB server with `--context 32768`, one session and MTP loaded with 144 slots/layer; an HTTP chat request returned `OK`. `--context 65536` was not exercised.

**Long-prefill control:** the raw repeated-text fixture tokenized to 1,602 tokens and generated 8 tokens at the same `434d749a…` hash as the pre-change release binary. A single same-prompt comparison measured 14.91 tok/s prefill on that release binary versus 16.89 tok/s on the current release (94.83 s); this is an exploratory single-request result, not pp2048/tg128 qualification. The old `build/release/gufo.exe` was built on October 19 and was overwritten by the candidate release build, so the baseline executable is not retained and this timing comparison is not reproducible. The current run read 132.4 GB (66.2 GB speculative Room-prefetch bytes) and reported 34.3 s hard stalls; no-prefetch Desk/Room control on the same build reported 102.6 s prefill and 45.9 s stalls. The offline trace replay used 157 slots/layer and predicted 41,748 LRU misses, 44,663 cumulative-hotness misses and 42,974 decayed-hotness misses; runtime recorded 20,103 synchronous cold misses plus 3,183 late-prefetch waits. These differences are expected because replay intentionally omits asynchronous I/O and speculative prefetch.

**Matched logits and perplexity:** a GPU probe captured the final 64 complete FP32 vocabulary rows for the same 1,602-token prompt using the existing streamed path with a 20-slot cache (Desk disabled) and the new Desk/Room path with 157 slots. The 15,892,480 values are byte-identical (SHA-256 `156744c3b467a4b449f4a508ab73832ec03600748f3b687c43d75cd69a984d85`); NLL over the 63 in-prompt labels is `0.47005817058939048` and perplexity is `1.6000872685296976` for both. This directly checks streamed/cache/Desk arithmetic, but it is a same-build no-Desk control rather than a preserved pre-change binary. The old release executable used for the token-hash and timing comparison was overwritten before full logits were captured.

The control is reproducible with `build/gpu-test/qwen38_flash_next_gpu_probe.exe`, `--batch 2048 --context 4096 --logit-rows 64`, and the prompt `('The quick brown fox jumps over the lazy dog. Strix Halo executes this deterministic benchmark sequence. ' * 84) + ' Reply with exactly OK.'`. Use `--expert-cache-bytes 2600000000` for 20 slots or `--expert-cache-bytes 20140175392` for 157 slots, plus `--dump <file>`. `qwen38_flash_next_reference_probe.exe --tokenize-only` supplies the 1,602 token IDs; the raw dump holds the last 64 rows in token-major order. The binary dumps and 63-label NLL calculation are in ignored `build/gpu-test/` files.

**Remaining quality limits:** full-logit equivalence to the historical pre-change Phase 6 binary is not available. All C++ files changed for Phases 6.1/6.2 pass the Windows LLVM 19.1.5 `clang-format --dry-run --Werror` check after formatting; `ssd_stats.cpp` keeps the Windows SDK-required `windows.h` before `psapi.h` in a formatter-disabled include block. The repository-pinned formatter could not be run because Nix is unavailable on this host.

### Phase 6.2 whole-chunk Desk/Room validation (2026-10-20)

**Formatting and build:** Ran Visual Studio LLVM `clang-format 19.1.5 -i` on the Phase 6.1/6.2 C++ files. The first Release rebuild caught the formatter's reordering of `psapi.h` ahead of `windows.h`; the Windows include block in `src/storage/ssd_stats.cpp` is now explicitly protected in the required order. The subsequent `release` `gufo` build and `gpu-test` reference-probe build passed with TheRock 10.0.0 / `gfx1151`. The formatted files pass the matching `clang-format --dry-run --Werror` check.

**Untraced timing:** `build/release/gufo.exe prompt` (`gufo version 611267bf8583-dirty`; TheRock 10.0.0 / `gfx1151`) read the first 160 lines of the formatted `src/storage/expert_cache_manager.cpp` as raw text (4,932 bytes; 1,510 model tokens), with the Swift-1.5 Qwen3.8-Flash-Next IQ4_XS weights, shared Q8_0 MTP sidecar, greedy sampling, seed 459, PLE shelf, `--memory-budget-gb 38`, `--ssd-stats --verbose`, and a 128-token output cap. The model planner selected 204 expert slots/layer (24.364 GiB expert cache); prewarm loaded 26,129,965,056 bytes in 6.277 s. Prefill took 10.8581 s (139.067 tok/s); all 128 output tokens were generated at 11.2657 tok/s. The generated-token SHA-256 is `73998e6652f74c52cfe5702dca13b8d96a509200499454af20f4058904332961`. MTP accepted 105/124 proposals (84.677%).

The untraced request recorded 60,419 expert requests, 75.360% cache hits, 4,962 prefetch requests, 69.629% prefetched in time, 159 late-prefetch waits, 14,728 synchronous cold misses, 15,167.545 ms hard stalls, 52,642,375,070 SSD bytes read and 2.665 GiB/s effective NVMe. Prefill accounted for 8,312 synchronous cold misses, 7,488.996 ms hard stalls, 22,214,822,112 demand-read bytes and 7,411,970,048 prefetch-read bytes. `SsdStats` reported a 39.394 GiB peak working set; independent 500 ms process sampling measured 39.393 GiB working set and 37.308 GiB private bytes. This stayed below the user's 40 GB process ceiling at a 38 GiB planner budget.

**Separate route-trace validation:** Repeated the same prompt, artifacts, sampling, budget and output cap in a separate process with `GUFO_EXPERT_ROUTE_TRACE=build/phase6-2-route-trace-38g-20261020.jsonl`. It produced the same 128-token SHA. Trace analysis covered 48 prefill layer-chunks at 1,510 rows each (724,800 routed assignments), with 7,161 resident and 10,611 absent unique expert IDs summed across layers. The exact assignment multiset was covered by the recorded passes in every layer; Pass 0 matched the resident set, and every absent routed expert appeared in exactly one Room Resolve pass (195 Resolve passes total, at most five per layer-chunk). Since each Room pass is disjoint and `ExpertCacheManager::Resolve` deduplicates repeated IDs within a pass, each absent routed expert can trigger at most one demand resolution/read per layer-chunk. Aggregate trace-run prefill stats were 8,312 synchronous cold misses and 7,411,970,048 prefetched bytes. The trace verifies Resolve-pass ownership; byte/read counters remain aggregate rather than per-expert.

The trace-enabled timing (11.7538 s prefill, 128.469 tok/s; 10.7793 tok/s decode) is reported only to identify the separate verification run, not as the timing result. Its process peak was 39.394 GiB working set / 37.308 GiB private bytes. Logs, prompt fixture and trace are retained under ignored `build/` as `phase6-2-timing-38g.*`, `phase6-2-trace-38g.*`, `phase6-2-expert-cache-manager-first160.txt` and `phase6-2-route-trace-38g-20261020.jsonl`.

### Linux Handoff & Verification Checklist

This checklist is for the Linux validation pass; it records commands, not a Linux pass. Windows Phase 6.2 has a measured 38 GiB-budget timing run and a separate successful route-ownership trace check above. Reproduce the tests and matched-token quality controls on Linux before handoff sign-off.

From the repository root on the Linux gfx1151 host:

```bash
nix develop -c python3 tools/ci/check-format.py
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target gufo qwen38_flash_next_tests expert_cache_manager_test
nix develop -c ctest --preset gpu-full -R '^(expert_cache_manager_test|qwen38_flash_next\.(iq_experts_ops|moe_ids_ops|moe_epilogue_ops|routed_wmma_ops))$' --output-on-failure
nix develop -c python3 tests/tools/test_expert_cache_replay.py
build/gpu-test/gufo --version
build/gpu-test/gufo diagnose
```

Before Linux performance claims, run the matched-token logit/perplexity comparison against the recorded baseline with identical model artifacts, prompt bytes, toolchain, cache budget, context and sampler settings. Record trace, benchmark outputs, build identity and peak process memory under `build/`; keep the measured process use within the host's safe limit. The 8/64-token checks and the 1,602-token same-build logit control above remain the existing Windows controls; repeat them on Linux before handoff sign-off.

### Phase 6.3 streamed epilogue, priority lanes, and PLE accounting (2026-10-21)

**Implementation:** `StreamedMoeRows` now defers the MoE epilogue across all windows, restores base scratch, and applies one epilogue to the full output. Desk/Room leaves the pending F16 epilogue intact for `Combine` and clears it on failures. Streamed calls reject unsupported deferred-epilogue and route-count flags. Desk/Room still requires `slots > 2 * experts_used`; the shared boundary predicate and debug assertion preserve the fallback at equality. The regression test uses synthetic F16 expert rows and verifies the accumulated-window/single-epilogue result, pending-state consumption, supported Q4_K/Q5_K plus Q5_1/Q8_0 format combinations, and the slot boundary without loading a model.

Lookahead router predictions are now captured before current-layer dispatch, but their disk requests are submitted from the streamed window/Desk/Room paths only after the current demand Resolve. The reader and shared scratch pool prioritize demand, then Room, then speculative work. The manager caps outstanding speculative layer-prediction transactions at two, excludes Room reads, retires credits on completion/cancel/failure/FinishPrefetch, and skips capped speculation without failing inference. The direct host mapping read in Desk/Room is documented as relying on the single-scheduler-thread executor invariant.

Linux `SsdBatchReader` now falls back to buffered open on `EINVAL`/`EOPNOTSUPP`, warns once per file, coalesces overlapping/adjacent aligned ranges, scatters payloads through the existing bounce-buffer path, and reports physical bytes transferred. PLE disk bytes and read time have dedicated overall and per-phase counters; expert demand counters no longer include PLE reads. `docs/CLI.md` now documents `--memory-budget-gb`, `--ple-shelf`, `--ssd-stats`, and `--cache-prior-bias`, including the option's GB label versus GiB planner conversion.

**Windows validation:** TheRock 10.0.0 / `gfx1151`; both `gpu-test` (`gufo`) and `release` (`gufo`) builds passed. Focused CTests passed: `ssd_stats_test`, `expert_cache_manager_test` (including the two-transaction cap/release), `ssd_batch_reader_test` (overlap/adjacency and byte accounting), `qwen38_flash_next.moe_epilogue_ops`, and `qwen38_flash_next.routed_wmma_ops`. `python tests/tools/test_expert_cache_replay.py` passed all six tests. The changed C++ files pass the repository clang-format 19.1.5 dry-run check. The repository-wide `tools/ci/check-format.py` still reports formatting violations in unrelated existing CLI, scheduler, and kernel files; those files were not reformatted as part of this change.

**IQ4_XS smoke and Desk/Room trace:** `build/release/gufo.exe` (`8fc4ab6fd3ad-dirty`) ran the exact local Swift IQ4_XS shard and Q8_0 MTP sidecar with `--memory-budget-gb 38 --ple-shelf --ssd-stats --cache-prior-bias 0.5`, 1,561 prompt tokens, greedy sampling, and a 16-token output limit. The model loaded in 46.8226 s; prewarm selected 205 slots/layer and read 26,258,354,176 bytes in 6.798 s. Prefill was 22.3204 s (69.936 tok/s), generation 16 tokens at 4.343 tok/s, output SHA-256 `21e2a037c61a5baa8e16e27a1b9e86badd05204156f14960e023668371ab985f`; MTP accepted 8/10 proposals. The 38-GiB planner budget reported 39.486 GiB peak working set and 369.805 MiB swap delta.

The trace `build/phase6-3-route-trace-final-20261021-204316.jsonl` covered 48 prefill layer-chunks of 1,561 rows. All 48 had nonempty Desk sets; across them Pass 0 matched the resident expert set, all routed assignments were covered exactly once, and Room passes did not overlap (149 Room Resolves total). This confirms the IQ4_XS run exercised Desk/Room. PLE stats reported 14,424 requests, 1,298,160 bytes and 1,669.966 ms; the total SSD byte count equals the expert demand/prefetch bytes plus the dedicated PLE bytes, confirming the counters are separated.

**Linux status:** The POSIX fallback/coalescing implementation is present, but Linux compilation and CTests remain unverified on this Windows host: Nix and WSL are unavailable. Run the Linux handoff commands above before Linux sign-off; no Linux pass or matched-logit/perplexity result is claimed here.
