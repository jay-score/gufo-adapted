# Adding a Model to Gufo (this fork: Gufo + native Windows/gfx1151)

This is the step-by-step procedure for adding a new model to this
repository. It is synthesized from the local checkout (`windows-port`
branch): `docs/DEVELOPMENT.md`, `docs/TESTING.md`, `docs/WINDOWS.md`,
`docs/SERVER.md`, `docs/CLI.md`, `docs/models/README.md`, the root
`CMakeLists.txt`, `src/cli/serve/inference_backend.cpp`,
`src/cli/serve/serve.cpp`, and the existing packages under
`src/models/`.

Upstream is `gufo-org/gufo`. This fork adds native Windows 11 x64 on
Strix Halo `gfx1151` (TheRock ROCm 10.0.0, `compat/win32`, `tools/windows/`,
GUI launcher). Every new model must keep Linux working: Linux CI remains
required (see Step 12).

## 0. Understand the two model families

Gufo does not have a generic “add any GGUF” path. Each architecture is a
model-private package. Pick your family first:

| Family | Weight container | Examples in tree | Loader entry |
| --- | --- | --- | --- |
| GGUF LLM | one or more `.gguf` shards, `general.architecture` selects the backend | `qwen` (Qwen3.8-27B, fallback), `qwen38_flash_next` (`qwen4exp`), `deepseek_v4_flash` (`deepseek4`), DFlash draft (`dflash`) | `core::GgufReader::OpenFile`, `Config::FromGguf`, `Model::Load(model_path, ModelOptions{}, ...)` |
| Safetensors directory | model-specific layout: audio config + shards, or diffusion component directories; manifest when pinned | `qwen3_asr`, `qwen3_tts`, `qwen_image_21`, `minimax_h3` | Audio examples: `LoadModelDirectory(dir)`, `ParseModelConfig` / `LoadModelConfigFromPath`, `LooksLike…` + `IsSupported…`; diffusion uses package-specific weights/inventory loaders |

These are examples of model-owned APIs, not one shared loader contract.
Qwen-Image, for example, requires a Diffusers directory with
`model_index.json` and component configurations such as
`transformer/config.json`, `text_encoder/config.json`, and `vae/config.json`.

Dispatch reality for GGUF text today
(`src/cli/serve/inference_backend.cpp::InferenceBackend::load`,
mirrored in `src/cli/bench/bench.cpp` and `src/cli/prompt/prompt.cpp`):

- `general.architecture == "deepseek4"` → `deepseek_v4_flash::Model::Load`
- `general.architecture == "qwen4exp"` → `qwen38_flash_next::Model::Load`
- anything else → `hip::QwenGpuModel::CreateFromGguf` (the Qwen 3.x/3.8 path)

So a new LLM either (a) extends the fallback Qwen path when the block
structure really matches, or (b) — the normal case — gets a new
`architecture` string plus a new `src/models/<model>` package and a new
dispatch branch in those three call sites.

Modality reality for non-LLM models (`src/cli/serve/serve.cpp`):

- `serve llm` → `InferenceBackend` (GGUF text)
- `serve image` → `ImageService` (`qwen_image_21`)
- `serve video` → `VideoJobService` (`minimax_h3`)
- `serve tts` → `TtsService` (`qwen3_tts`)
- `serve asr` → `AsrService` (`qwen3_asr`)

A new audio/image/video model needs a service + a `serve <modality>`
wiring, not just weights. A new text LLM reuses `serve llm`.

Philosophy constraints (from `README.md` / `DEVELOPMENT.md`):

- Keep model code, kernels, tests, tools, and numerical contracts **with
  their model**. Do not reuse kernels across different models; that
  limits blast radius.
- One canonical long option and backend name per behavior; no aliases.
- Successful optimizations become the default path, no extra switches.
- Production paths retain quality; quality gates pass before performance
  counts.

## 1. Survey the closest existing model and copy its skeleton

Do not start from scratch. Pick the nearest template:

- Dense text LLM or DFlash-style speculation → `src/models/qwen/`
  (`weights.cpp`, `forward.hpp/cpp`, `hip/model_loader.cpp`,
  `hip/executor.cpp`, `dflash_weights.*`, `hip/dflash/`, `tokenizer.cpp`,
  `chat_template.cpp`, vision under `vision/`). Reuse its implementation
  for a Qwen-family variant only when the model graph and tensor contract
  match; otherwise use it as a structural example.
- Hybrid recurrent/QSA MoE text LLM with MTP →
  `src/models/qwen38_flash_next/` (`config.hpp/cpp`, `weights.hpp/cpp`,
  `engine.hpp/cpp`, `kernels/rocm/`, `reference.cpp`, `cpu_ops.cpp`,
  `ngram.cpp`, `prompt_lookup.hpp`). Flash-Next supports MTP, not DFlash.
- New MoE / special attention LLM → also read
  `src/models/deepseek_v4_flash/` (`engine.hpp/cpp`, `chat_template.*`,
  `kernels/rocm/`, `runtime/`, `UPSTREAM.md`, `LICENSE.ds4`).
- Speech / TTS from safetensors → `src/models/qwen3_asr/` or
  `src/models/qwen3_tts/` (`config.hpp/cpp`, `loader.hpp/cpp`,
  `tensor.hpp/cpp`, `tokenizer.cpp`, `prompt.cpp`, `hip/*_runtime.cpp`,
  `reference/run_official.py`, `tools/benchmark.py`).
- Image / video diffusion → `src/models/qwen_image_21/` or
  `src/models/minimax_h3/` (note: H3 sources are currently compiled into
  `gufo_core` in the root `CMakeLists.txt`; the image model has its own
  `gufo_qwen_image_21` library with a `model_stub.cpp` for non-HIP builds).

List the template directory recursively before copying:

```sh
ls -R src/models/qwen38_flash_next | head -n 80
ls -R tests/models/qwen38_flash_next | head -n 40
```

## 2. Scaffold `src/models/<model>`

Create `src/models/<model>/CMakeLists.txt` exporting `gufo_<model>`.
This GGUF example includes a scalar CPU reference; omit that target if
validation uses other independent references (see Step 6).

```cmake
add_library(gufo_mymodel STATIC
  config.cpp
  weights.cpp
  tokenizer.cpp
  chat_template.cpp
)
target_include_directories(gufo_mymodel PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_mymodel PUBLIC gufo_core Threads::Threads)

# If using a heavy scalar CPU oracle, exclude it from routine builds.
add_library(gufo_mymodel_reference STATIC EXCLUDE_FROM_ALL
  cpu_ops.cpp reference.cpp)
target_link_libraries(gufo_mymodel_reference PUBLIC gufo_mymodel)
set_source_files_properties(cpu_ops.cpp reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if(ENGINE_ENABLE_HIP)
  target_sources(gufo_mymodel PRIVATE
    engine.cpp
    kernels/rocm/kernels.hip.cpp
    kernels/rocm/executor.cpp)
  set_source_files_properties(kernels/rocm/kernels.hip.cpp
    PROPERTIES LANGUAGE HIP)
  target_compile_definitions(gufo_mymodel PRIVATE MYMODEL_ROCM_BUILD=1)
  target_link_libraries(gufo_mymodel PUBLIC hip::host roc::hipblas roc::hipblaslt)
endif()
```

Follow the per-model precedents for details:

- `qwen3_asr/CMakeLists.txt`: small static lib + `gufo_<model>_register_tests()`
  function with `cpu`/`gpu` labels, `SKIP_RETURN_CODE 77` for
  external-model tests.
- `qwen38_flash_next/CMakeLists.txt`: separate vendored MMQ lib
  (`gufo_<model>_mmq`, HIP C++17, `-mwavefrontsize64` where needed,
  `GGML_HIP_NO_VMM`), system includes for `hipcub`/`rocprim`/`rocwmma`.
- `qwen_image_21/CMakeLists.txt`: `model_stub.cpp` for the
  non-`ENGINE_ENABLE_HIP` build; HIP-only `-O3 -ffp-contract=off` where
  arithmetic boundaries matter.

Then attach it in the root `CMakeLists.txt` with `add_subdirectory`:

```cmake
add_subdirectory(src/models/mymodel)
```

and link it where it is consumed (`gufo_core`, `gufo_http`, `gufo`,
`gufo_llm_cli` — mirror the existing `target_link_libraries` lines for
your family; e.g. image links into `gufo_http`, DeepSeek links into
`gufo_llm_cli` + `gufo` under `ENGINE_ENABLE_HIP`).

Stage new files in Git early: Nix flakes only see tracked files.

## 3. Implement config parsing (fail closed)

GGUF path — implement `Config::FromGguf(reader, require_trunk, error)`:

- See `src/models/qwen38_flash_next/config.hpp/cpp`. Every value is read
  from the file; fixed limits only bound what the runtime was written
  and validated for.
- Reject on wrong `general.architecture` (e.g. `qwen4exp` check).
- Read `<arch>.*` keys with typed helpers (`U32`, `F32`, `U64Array`);
  validate shapes (`block_count` vs `nextn_predict_layers`,
  `rope.dimension_sections`, head dims, MoE counts).
- For MTP/draft sidecars, implement `MtpMatches(trunk)` so a mismatched
  draft is rejected at load, not at decode.

Safetensors path — follow the checkpoint's directory layout and implement
model-specific configuration validation:

- Audio example: `ParseModelConfig` + `IsSupported…` in
  `src/models/qwen3_asr/config.hpp/cpp`: structs for each sub-model,
  JSON parse, then `IsSupportedModelConfig` that accepts **only** the
  qualified checkpoint (ASR accepts only 1.7B; 0.6B and forced-aligner
  are intentionally out).
- Record `model_type`, `architecture`, `dtype`, token IDs, and supported
  languages/voices as applicable to that model.
- Diffusion example: `src/models/qwen_image_21/weights.cpp` validates
  `model_index.json` and the component configurations; do not assume a
  single root `config.json` describes the pipeline.

Add `tests/models/<model>/*config_test.cpp` (label `cpu`) covering valid,
missing-field, and wrong-checkpoint inputs.

## 4. Implement weight loading

GGUF path — implement `weights.hpp/cpp`:

- Define a non-owning `TensorRef` view (`data`, `GgmlType`, `cols`/`rows`/
  `experts`, `file_offset`, `shard`, `name`) following
  `qwen38_flash_next/weights.hpp`.
- Bind every expected GGUF tensor name explicitly; missing or
  wrong-shape tensors are load errors with the tensor name in the
  message.
- Handle quantization at load deliberately (e.g. Flash-Next documents
  that Q6_K embedding/head rows are re-encoded to Q8_0, and that
  UD-IQ4_XS expert paths run on vendored MMQ/MMVQ). Never silently
  re-quantize without documenting it in the model README + QUALITY.
- Multi-shard: the loader discovers remaining shards from the first
  shard path (document this in the model README load section).

Safetensors path — implement a loader for the model's layout:

- Audio example in `qwen3_asr/loader.hpp`: `LoadModelDirectory(dir)` returning
  `LoadResult{ok, error, config, store, mappings, mapped_regions}`,
  plus `LooksLikeQwen3Asr(dir)` for fast probing and
  `RegionsFor({prefixes…})` to restrict mapped shards per component
  while preserving payload alignment for GPU upload.
- Validate inventory against a manifest when the model needs pinning
  (see `minimax_h3/MINIMAX_H3_FL2VA_BF16.source-manifest.json` and the
  `video --manifest` override).

## 5. Implement tokenizer + chat template + output parsing

- Reuse `tokenization::QwenTokenizer::CreateFromGguf` when the artifact
  carries a Qwen vocabulary/merges (see `src/models/qwen/tokenizer.hpp`).
  Otherwise implement a model-private tokenizer (see `qwen3_asr/bpe.cpp`,
  `minimax_h3/tokenizer.cpp`) with exact-token-ID tests against
  independent fixtures.
- Implement `chat_template.cpp/hpp` with `ValidateGgufTemplate(reader,
  error)` and rendering that matches the official template. The
  `InferenceBackend::load` path rejects unsupported templates before
  allocating the model — keep that behavior for the new arch.
- Cover malformed/truncated input, special tokens, multilingual/edge
  cases, and (for reasoning models) thinking/effort handling. Mirror
  `tests/models/qwen/tokenization/chat_template_test.cpp` and the
  `chat_template_hf_token_golden_test`.
- Check the model's output dialect as well as its input template.
  `src/cli/serve/openai_chat.cpp` currently recognizes `<think>` reasoning
  and Qwen/DeepSeek tool-call formats through `ParseGeneration`,
  `ParseQwenCalls`, `ParseDsmlCalls`, and `StreamingTextFilter`. A different
  dialect needs corresponding parsing support; a loader and runner alone
  do not provide it. Cover buffered and streaming Chat Completions and
  Responses, including reasoning boundaries, tool arguments, and replay.

For image/API changes, `/v1/models` and `/props` must reflect actual
loaded image support; harnesses cache these capabilities. Cover direct
attachments, images in tool results, history replay, and streaming.

## 6. Establish independent correctness references before optimizing

Choose references that cover the model's operators and end-to-end behavior.
A complete scalar CPU implementation is one useful approach, not a universal
repository requirement. Pinned official runtimes and hand-checkable analytic
fixtures are also used; agreement between two Gufo execution paths alone does
not establish independent correctness. See `docs/TESTING.md`.

- Scalar example: `qwen38_flash_next/reference.hpp` and `cpu_ops.*`
  implement the single-token graph with float32 formulas. Keep a heavy
  CPU reference behind `EXCLUDE_FROM_ALL`. Flash-Next uses `-O2` even in
  debug because its reference decodes every weight on the fly.
- Where useful, provide stage-trace hooks to isolate operator errors from
  accumulated numerical drift. Flash-Next's MTP examples are `MtpStep`
  and `MtpTrace`; these are not required APIs for other models.
- Official-runtime example: `qwen3_asr/reference/run_official.py` runs
  the pinned checkpoint with PyTorch and captures deterministic artifacts.
- Small operator tests can use analytic fixtures and synthetic tensors
  without a full-model CPU implementation or downloaded weights.

Record which independent reference each check uses, its numerical limits,
and any gaps in the model's `QUALITY.md`.

## 7. Implement the HIP runtime (`Model` / `Session`)

Follow `qwen38_flash_next/engine.hpp` as the API shape:

- `struct ModelOptions{max_context, draft/sidecar paths, vision path,
  decode_concurrency, policy opt-ins}`.
- `class Model : enable_shared_from_this<Model>` with static
  `Load(path, options, error)`, `CreateSession(mode, max_context, error)`,
  `Tokenize/Decode`, `EosToken/IsStopToken`, `ResidentBytes/SessionBytes`,
  `config()/tokenizer()/VisionEncoder()`.
- `class Session` with `Sync(prompt)`, `Evaluate(token)`,
  `TeacherForce(tokens, rows, …)` for numerical A/B,
  `DecodeStep/DecodeBatch/EvaluateBatch`, `Reset`,
  `SaveSnapshot/RestoreSnapshot`, `SnapshotBytes`,
  `kSnapshotPayloadVersion` (bump on any payload or arithmetic change).
- Keep kernels under `kernels/` (or `kernels/rocm/`, `hip/`), attention
  under testable units, BLAS via hipBLAS/hipBLASLt, quantized GEMV/MMVQ
  vendored per-model when needed (never shared across models).
- Image state participates in prefill, decoding, verification,
  multi-turn reuse, and disk-cache identity; image snapshots require
  the matching immutable prompt first (pixel data is not serialized).
- Speculation (MTP/DSpark/DFlash/prompt-lookup) keeps verification
  unchanged so greedy text stays byte-identical; policy opt-ins change
  only proposals. Seeded replay is checked within the same execution
  configuration.

Keep production HIP behind `ENGINE_ENABLE_HIP`; provide a CPU stub when
the package must link without HIP (see `qwen_image_21/model_stub.cpp`).

## 8. Wire the CLI + server dispatch

Wire a new GGUF `architecture` into these entry points, then update flags
and output parsing where needed:

1. `src/cli/serve/inference_backend.cpp` — add the
   `general.architecture == "<myarch>"` branch: validate speculative
   config (which backends are allowed, required `--*-model` sidecar,
   draft-token limits), validate the chat template, call
   `mymodel::Model::Load`, fingerprint artifacts for disk cache, then
   delegate to the matching `load(shared_ptr<Model>, …)` overload.
   Add the `load(shared_ptr<mymodel::Model>, …)` overload +
   runner-state class following the DeepSeek / Flash-Next / Qwen
   runners in the same file (fingerprint strings like
   `model_kind=<myarch>`, `chat_template=…`, `state_abi=…`,
   `payload_layout=…`).
2. `src/cli/bench/bench.cpp` + `src/cli/prompt/prompt.cpp` — same
   architecture check so `gufo bench` / `gufo prompt` / `gufo chat`
   route to the new model (see the existing `deepseek4` / `qwen4exp`
   branches).
3. `src/cli/serve/serve.cpp` — only if the model needs new flags
   (e.g. `--mtp-model`, `--mmproj`, `--think`, speculative policies).
   One canonical long option per behavior; no aliases. Register help
   text in `PrintServeHelp` and parsing in `RunServe` together so they
   cannot drift.
4. `src/cli/serve/openai_chat.cpp` — if the model introduces a different
   reasoning/tool-call dialect, update output parsing and streaming as
   described in Step 5, with Chat Completions and Responses coverage.

A new safetensors modality additionally needs:

- A service class (`AsrService` / `TtsService` / `ImageService` /
  `VideoJobService` pattern in `src/cli/serve/`).
- A `serve <modality>` subcommand in `serve.cpp` with `--model <DIR>`,
  `--served-model-name`, context/queue/storage options, plus
  `HttpServer` wiring and (where applicable) a `gufo transcribe` /
  `gufo video` CLI.
- Capability reporting (`/v1/models`, `/props`, `supports_image_input`)
  reflecting what is actually loaded.

Vision rule: vision needs a compatible loaded encoder/projector. Do not
assume an F16 projector is interchangeable with BF16; document the exact
qualified sidecar (Flash-Next uses `mmproj-BF16.gguf`, auto-discovered
beside the model or via `--mmproj`).

## 9. Add tests (smallest covering check first)

Create `tests/models/<model>/` and register in `CMakeLists.txt`
(mirror `gufo_qwen3_asr_register_tests()` /
`gufo_qwen_image_21_register_tests()` / `tests/models/qwen38_flash_next/CMakeLists.txt`):

- CPU (no GPU): config parsing, loader validation,
  tokenizer exact IDs, prompt/template rendering, API contracts.
  Use fixtures without downloaded weights for hosted checks; mark tests
  requiring external artifacts with `external-model` and
  `SKIP_RETURN_CODE 77`. Labels: `cpu;models;<model>;…`. Add to
  `gufo_pr_targets` in `cmake/Checks.cmake` only if it belongs in every
  hosted PR.
- HIP operator tests (need a GPU; downloaded weights only when the test
  uses them): one test per kernel family against an independent reference
  or analytic fixture. Synthetic tensors suffice for many checks. Label
  `gpu` (+ `hip;gfx1151`), `external-model` + `SKIP_RETURN_CODE 77`
  when weights are required, `RUN_SERIAL TRUE` for device tests.
- State tests: long-context boundaries, snapshot/restore,
  fresh-versus-reused state, rollback, seeded replay, EOS commit,
  cancellation.
- Server tests: `openai_chat_test` + `http_server_test` for API
  changes, Qwen template tests when changing prompt rendering,
  direct-vs-HTTP equivalence, batched independent requests,
  multi-turn reuse, and buffered/streaming reasoning and tool-call parsing
  for the model's dialect. Responses history is stateless — preserve
  tool-call identity and image order; use isolated client
  settings/cache for integration tests.
- Matched-token validation (required when touching kernels, loading,
  tokenization, prefill/decode, sampling, caching, speculation):
  use the model's validation harness with identical token histories.
  Require exact full-logit equality when arithmetic is unchanged, or
  declared numerical limits when it intentionally changes. Record
  commands, build identities, model/sidecar identities, corpus hash,
  and results. **Flash-Next example:**
  `gufo bench --logit-eval corpus.txt --logit-schedules "…"`
  with model/MTP options as needed and separate `--logit-out` prefixes,
  then `python tools/bench/logit-eval.py LEFT.bin.sha256 RIGHT.bin.sha256`.
  This command is implemented only for Flash-Next; a new model needs its
  own harness or an explicit extension of that implementation. The Python
  comparator checks exact raw-logit hashes only. Intentional numerical
  changes need a comparator that measures errors against the declared
  limits; hashes cannot do that. See
  `docs/TESTING.md#matched-token-and-layer-comparisons`. A missing-model
  skip is not a pass — document blocked coverage explicitly.

## 10. Add the four required documents

Create the four documents and extend the checker as described below.
`python tools/ci/check-docs.py` (or `nix build .#checks.x86_64-linux.docs`)
checks required files and validates links/anchors/JSON offline:

- `docs/models/<model>/README.md` — model card, acquisition (exact
  repo/revision/include list), usage and modes, sidecars, limits.
- `docs/models/<model>/BENCHMARKS.md` — retained measurements only,
  scope/date per cell (see `docs/BENCHMARKS.md` methodology).
- `docs/models/<model>/QUALITY.md` — independent numerical checks,
  reference/scope, replay guarantees, unresolved gaps.
- `docs/models/<model>/EXPERIMENTS.md` — short retained/rejected
  decisions.
- Add a row to `docs/models/README.md` (weights/inputs + guide link).
- Add the model's documentation directory name to the fixed model list
  in `REQUIRED_DOC_FILES` in `tools/ci/check-docs.py`. The checker does
  not automatically require the four documents for newly added models.
- Keep executable fixtures with tests; new logit/trace dumps and local
  profiles go in ignored top-level `artifacts/`; commit only small
  independent reference evidence under `docs/models/<model>/artifacts/`.
- Record new third-party code in `THIRD_PARTY_NOTICES.md`
  (`tools/ci/check-dependencies.py` validates it).

## 11. Linux build / test loop (reference)

```sh
nix build                              # production package
./result/bin/gufo diagnose
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
nix build .#checks.x86_64-linux.pr     # hosted CPU/repository checks
```

Without Nix: `cmake --preset release`, `cmake --build --preset release`.
Presets: `release` (prod, no tests), `gpu-test` (HIP + assertions),
`cpu-test` (host-only), `cpu-sanitizer` (ASan/UBSan). See
`docs/DEVELOPMENT.md`.

## 12. Windows fork checklist (required for every new model)

Linux CI is still required for shared code. On top of Step 11, do this
on Windows 11 x64 / gfx1151 (see `docs/WINDOWS.md`):

1. Toolchain: TheRock ROCm **10.0.0** for Windows/gfx1151, VS C++ Build
   Tools (MSVC headers/libs + Windows SDK), CMake, Ninja, vcpkg in
   manifest mode (`vcpkg.json`). TheRock clang compiles the engine; do
   not substitute MSVC or another ROCm release without validating it.
2. Check + build from the repo root:
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1
   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Target gufo -Jobs 4
   powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target <test-target> -Jobs 4
   ctest --test-dir build/gpu-test -R '^<test-name>$' --output-on-failure --no-tests=error
   ```
   Prefer `build.ps1`: it imports the MSVC env, configures CMake,
   embeds the Git revision, and stages runtime DLLs + ROCm kernel dirs
   next to `gufo.exe`. Outputs are `build\release\gufo.exe` or
   `build\gpu-test\gufo.exe` — keep the exe with its DLLs; check the
   GUI’s selected executable before testing a rebuild.
3. Keep Windows adaptations in `compat/win32`, platform helpers
   (`src/core/platform/`), or `_WIN32` branches. Never hard-code a
   developer’s machine paths; `-Vcpkg`/`-Rocm`/`-Ninja` override
   install locations. New POSIX-isms need a `compat/win32/include`
   shim + `posix.cpp`/`spawn.cpp` implementation (note
   `O_DIRECT`/`O_CONCURRENT_RANDOM`, `flock`, `renameat`,
   `posix_spawnp`, socket semantics in
   `src/core/platform/socket.hpp`, 8 MiB stack).
4. HIP specifics: `CMAKE_HIP_ARCHITECTURES` must stay `gfx1151`;
   `compat/win32/hip_include/cmath` works around the MSVC 14.5x
   `isgreater` collision — include new HIP TUs in the same build or
   they will fail on Windows first.
   Validate weight placement at the real model's size. Unified physical
   memory does not make Linux host-registration strategies portable to
   Windows: a model-sized `hipHostRegister` can return success and still
   hang the next stream creation or fail a GPU read. Qwen27B uses
   `hipMalloc` plus the bounded `hip::WeightUpload` pipeline on Windows;
   keep the source reader alive for tensor metadata, drain uploads before
   freeing destinations on failure, and verify unchanged bytes and logits.
   See [the reproduced Windows memory failure](WINDOWS.md#qwen27b-weight-memory).
5. GUI + API: `launch-gui.bat` wraps `tools/windows/gui.ps1` (Flask/
   Waitress panel at `http://127.0.0.1:8090`, engine at
   `http://127.0.0.1:8080/v1`). Keep the panel thin; settings live in
   `%LOCALAPPDATA%\Gufo\launcher.json`, saving is explicit, and opening
   the GUI must not load a model. If the model has settings (context,
   sidecars, speculative flags), add them to the curated GUI list and
   run `.\build\gui-env\Scripts\python.exe -m unittest discover -s
   tests/tools -p "gui_*_test.py"`.
6. Stop the instance using that build before replacing its exe/DLLs
   (GUI Stop/Exit). Record `gufo.exe --version`, toolchain,
   model/quantization, sidecars, context, sampling and speculative
   settings with every benchmark; distinguish build success, CPU
   tests, GPU correctness, and measured performance.
7. Docs/tests: `python tools/ci/check-docs.py`,
   `python tests/tools/windows_launcher_test.py`,
   `python tests/tools/windows_setup_test.py` as relevant. C++ changes:
   run `python tools/ci/check-format.py` with CI’s clang-format on
   PATH. Bounded Windows CPU CI is the reference (`ci.yml`
   `cpu-test`); hosted Windows CI does not validate GPU kernels or
   real-model quality — run `openai_chat_test`/`http_server_test`
   plus real-model smoke tests for loader/vision/inference changes
   locally.

Existing Windows-validated behaviors use `GUFO_PLATFORM_TUNING`
(`src/core/platform/tuning.hpp`). Most switches are read explicitly by
Flash-Next's engine, executor, and uploader; `prompt_checkpoint` is read
by Qwen prompt preparation. A new model does not automatically inherit
these behaviors: adopt only applicable changes in its implementation and
validate them independently. Existing switches default on for Windows
and off elsewhere. Qualify new platform-specific changes before choosing
their defaults, preserve Linux behavior, and record measurements in
`docs/WINDOWS.md#windows-validated-switches`.

## 13. Qualify before calling it done

- All Step 9 checks relevant to the change pass on both Linux
  (`gpu-full` focused targets) and Windows (`gpu-test` build of the
  affected target).
- `QUALITY.md` records agreement with an **independent**
  implementation (official torch runtime or hand-checkable analytic),
  not just self-consistency; greedy speculation reproduces AR.
- `BENCHMARKS.md` cells carry scope/date/settings; compare against the
  reference project’s server where applicable (see the
  `benchmark-model` skill).
- Commit per AGENTS.md: Conventional Commits, single-line message, one
  focused commit per improvement, no push unless requested, no
  formatting churn mixed into the fix.

## Quick checklist (copy into the PR)

- [ ] `src/models/<model>/` package + `add_subdirectory` + link targets
- [ ] Config parsing rejects wrong arch/checkpoint with named errors
- [ ] Weights/loader binds every tensor, documents any re-quantization
- [ ] Tokenizer + chat template validated against official behavior;
      reasoning/tool-call output dialect covered in buffered and streaming APIs
- [ ] Independent references (scalar CPU, official runtime, analytic fixtures
      as appropriate); heavy CPU oracles use `EXCLUDE_FROM_ALL`
- [ ] HIP `Model`/`Session`/snapshot + per-model kernels, no shared kernels
- [ ] `inference_backend.cpp` + `bench.cpp` + `prompt.cpp` dispatch
      (+ `serve.cpp` service/flags for new modalities, `/v1/models`+`/props` honest)
- [ ] CPU + HIP + server + model-specific numerical checks, labels + skip codes correct
- [ ] Four docs + `docs/models/README.md` row + checker model-list entry + `THIRD_PARTY_NOTICES.md`
- [ ] `check-format.py` + `check-docs.py` clean; hosted `pr` selection passes
- [ ] Windows: `check.ps1` + `build.ps1` clean, DLLs staged, GUI selection
      verified, `gufo.exe --version` recorded, no Linux regression
