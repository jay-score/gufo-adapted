# AGENTS.md

## Platform and build

This fork supports native Windows 11 x64 on AMD Strix Halo (`gfx1151`) alongside
the upstream Linux target. Preserve Linux compatibility when changing shared
code. CMake owns compiler flags, dependencies and installation; Nix supplies
the pinned Linux toolchain.
Linux source-build prerequisites are in [README.md](README.md#build-from-source).

```sh
nix build                              # production package, no tests/tools
./result/bin/gufo diagnose
nix build .#checks.x86_64-linux.pr      # bounded hosted CPU/repository checks
nix develop                            # GPU development and reference tools

# Same production build without Nix, with the documented dependencies installed
cmake --preset release
cmake --build --preset release --parallel 4
```

Stage only task-owned paths before Nix builds; flakes include tracked files.
Measure performance with `result/bin/gufo` or `build/release/gufo`. Preserve
compiler/dependency versions when comparing results.

## Windows build and runtime

Use [docs/WINDOWS.md](docs/WINDOWS.md) for prerequisites and port details.
The validated GPU toolchain is TheRock ROCm 10.0.0 for Windows/gfx1151, VS C++
Build Tools (MSVC headers/libraries and Windows SDK), CMake, Ninja and vcpkg.
TheRock clang compiles the engine; do not substitute MSVC or another ROCm
release without explicitly validating it. Dependencies are pinned by
`vcpkg.json` and installed in manifest mode.

Run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File tools\windows\check.ps1
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Target gufo -Jobs 4
powershell -ExecutionPolicy Bypass -File tools\windows\build.ps1 -Preset gpu-test -Target openai_chat_test -Jobs 4
ctest --test-dir build/gpu-test -R '^openai_chat_test$' --output-on-failure --no-tests=error
```

- Pass `-Vcpkg`, `-Rocm` or `-Ninja` when installations differ from the script
  defaults. Do not hard-code a developer's machine paths in tracked files.
- Prefer `build.ps1`: it imports the MSVC environment, configures CMake,
  embeds the Git revision and stages runtime dependencies. Direct incremental
  CMake builds still need the compiler environment used during configuration.
- Outputs are `build\release\gufo.exe` or `build\gpu-test\gufo.exe` according
  to the preset. Keep the executable with its runtime DLLs and ROCm kernel
  directories. Check the GUI's selected executable before testing a rebuild.
- Stop the instance using that build before replacing its executable or DLLs.
  Use the GUI's Stop/Exit controls or stop only processes owned by the test.
- Record `gufo.exe --version`, toolchain, model/quantization, sidecars, context,
  sampling and speculative settings with benchmark results. Distinguish a
  successful build, CPU tests, GPU correctness and measured performance.
- Keep Windows adaptations in `compat/win32`, platform helpers or `_WIN32`
  branches where appropriate. Shared API/model changes still need Linux CI.

## Windows GUI and API compatibility

- [launch-gui.bat](launch-gui.bat) wraps `tools/windows/gui.ps1`. Keep it thin;
  Python environment setup belongs in the PowerShell launcher. The first run
  creates `build/gui-env` using Python 3.10+; later runs reuse it.
- The GUI is a small local Flask/Waitress control panel, not a chat client.
  Keep its curated settings and process controls simple. See
  [tools/gui/README.md](tools/gui/README.md) for the full behavior.
- GUI: `http://127.0.0.1:8090`; default engine API:
  `http://127.0.0.1:8080/v1`. These are separate processes and ports.
  Settings live in `%LOCALAPPDATA%\Gufo\launcher.json`; saving is explicit.
  Preserve existing settings and do not load a model just by opening the GUI.
- Gufo has its own CLI and API contracts; do not assume llama.cpp argument
  compatibility. Check [docs/CLI.md](docs/CLI.md) and
  [docs/SERVER.md](docs/SERVER.md), and keep request-level sampler/reasoning
  overrides working alongside saved server defaults.
- Vision needs a compatible loaded encoder/projector. Flash-Next has been
  exercised here with IQ4_XS weights, a Q8_0 MTP sidecar and `mmproj-BF16.gguf`;
  the original UD-Q4_K_XL benchmark is not a requirement for all loads.
  Do not assume an F16 projector is interchangeable with BF16.
- `/v1/models` and `/props` must reflect actual loaded image support.
  Harnesses may cache these capabilities. For image/API changes, cover direct
  attachments, images in tool results, history replay and streaming as relevant.
  Responses history is stateless; preserve tool-call identity and image order.
  Use isolated client settings/cache for integration tests.

## Focused tests

Formatting and Python repository checks may run on the editing host; they do
not need the remote GPU. Before committing C++ changes, run the shared CI check:

```sh
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
# Add --fix to apply formatting, then rerun the check.
```

Run the smallest check covering the change. `gpu-test` is RelWithDebInfo with
assertions enabled. Build only the affected target during iteration:

```sh
nix develop -c cmake --preset gpu-test
nix develop -c cmake --build --preset gpu-test --target <test-target>
nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
```

The same CMake/CTest commands work outside Nix. `cmake --build --preset pr`
runs the hosted contract suite after configuring `cpu-test`. Full CPU checks,
sanitisers, GPU/operator/model checks and H3 quality tools remain available
locally; see [docs/TESTING.md](docs/TESTING.md). Do not run full model sweeps,
video generation or duplicate suites for routine edits. A missing-model skip
is not a quality pass. Broaden checks when shared behavior changes or failures
expose risk.

On Windows, use the bounded CPU setup in
[.github/workflows/ci.yml](.github/workflows/ci.yml) as the reference for
`cpu-test`; it does not require a GPU. Build the affected targets before
running CTest so an old executable cannot produce a misleading pass.
For API changes, start with `openai_chat_test` and `http_server_test`; include
Qwen template tests when changing prompt rendering. Real-model smoke tests
are appropriate for loader, vision and inference changes, not routine GUI or
documentation edits.

```powershell
# GUI checks after the launcher has created its environment:
.\build\gui-env\Scripts\python.exe -m unittest discover -s tests/tools -p "gui_*_test.py"
# Flash-Next PowerShell launcher checks:
python tests/tools/windows_launcher_test.py
# Documentation changes:
python tools/ci/check-docs.py
# C++ changes: put the repository-compatible clang-format on PATH first.
python tools/ci/check-format.py
```

Keep `clang-format` aligned with CI's toolchain; an unrelated formatter version
can rewrite large amounts of code. Do not mix formatting churn into a fix.
For the batch launcher, smoke-test startup from a different working directory,
argument forwarding and clean shutdown. A GUI-only change does not require
rebuilding the engine.

## Profiling and kernels

Apply [.agents/skills/optimize-kernel/SKILL.md](.agents/skills/optimize-kernel/SKILL.md).
Use `tools/bench/build.sh` for standalone HIP experiments,
`tools/bench/gfx1151_peak.hip` for measured hardware ceilings,
`tools/prof/prof.py` for pipeline/wall-time profiles, and
`tools/prof/isa_mix.py` for instruction analysis. Production paths must retain
quality; successful optimizations become the default, without extra switches.

## Development

- Keep model code, tests, tools and numerical contracts with their model.
- Use one canonical long option and backend name per behavior; avoid aliases.
- Use `gh` for GitHub operations after checking `gh auth status`.
- Follow Conventional Commits with a single-line message.
- Keep improvements small and independently reviewable, with one focused commit
  per improvement. Do not push unless requested.
- Confirm the checkout, branch and working tree before editing; multiple Gufo
  clones/worktrees may exist. Preserve unrelated work and local configuration.
- If `.codegraph/` exists, use CodeGraph before searching or reading code to
  locate an implementation. Creating an index is the user's decision.
- Prefer `jj` when available (`jj version`); otherwise use Git.
- Follow the user's remote workflow and preserve unrelated work.
