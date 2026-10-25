# Gufo Engine — Build, Pinning & Runtime Guide (CachyOS / Strix Halo)

This document serves as the source of truth for building, pinning, and serving `gufo` on Linux (specifically CachyOS / Arch on AMD Strix Halo `gfx1151`).

> **Note for AI Agents & Contributors:**  
> **DO NOT** use imperative `cmake -B build` + `patchelf` hacks or `LD_LIBRARY_PATH` symlinks for production/persistent binaries. Furthermore, **DO NOT** modify `flake.nix` or `.devops/nix/package.nix` without running the dependency inventory checks (`tools/ci/check-dependencies.py`), as CI strictly audits those files.

---

## 1. Why We Use `nix build` Instead of `nix develop` + Manual CMake

When building on a non-NixOS host like CachyOS:
1. **Unpinned Dev Shells Break on Reboot:** Building manually inside `nix develop` (`./build/release/gufo`) leaves the binary untracked by Nix's Garbage Collector. Upon reboot or GC sweep, `/nix/store` runtime dependencies (`icu`, `libwebp`, etc.) can be deleted, and shell-only `LD_LIBRARY_PATH` entries vanish.
2. **`nix develop` Pulls Heavy Python/PyTorch Closures:** The default `devShell` builds `pythonTools` (including `torchWithRocm` and `torchvisionRocm`). You do **not** need this just to compile the C++/HIP `gufo` engine.
3. **`nix build` Wraps Everything Automatically:** Running `nix build` executes `.devops/nix/package.nix`, which:
   * Compiles `gufo` in a clean sandbox.
   * Automatically shrinks and bakes all transitive library `RPATH` entries into the ELF binary.
   * Wraps `$out/bin/gufo` via `wrapProgram` (`postFixup`) to inject the custom-patched Strix Halo `rocmRuntime` (`rocr-fixed-cwsr.patch`).
   * Creates a `./result` symlink in the repository that registers a **hard Nix Garbage Collection (GC) root**, ensuring reboots never delete required runtime libraries.

---

## 2. Standard Build & Installation Workflow

Run these commands from the repository root whenever pulling upstream updates or compiling local C++/HIP commits:

### Step 1: Build and Register the Nix GC Root
```fish
nix build -L -o result --no-accept-flake-config
```
* `-L`: Streams full build logs.
* `-o result`: Creates/updates the `./result` symlink in the repo directory, which locks the package closure in `/nix/var/nix/gcroots/auto/` so it survives reboots. *(Never delete `./result` unless you want Nix GC to reclaim the build).*
* `--no-accept-flake-config`: Skips the interactive prompt asking to trust `https://gufo.cachix.org` and builds cleanly using official `cache.nixos.org` packages plus local compilation for patched ROCr and `gufo`.

*(Optional Fast Incremental C++ Rebuilds: Ensure `/tmp/gufo-ccache` exists and is writable (`mkdir -p /tmp/gufo-ccache && chmod 777 /tmp/gufo-ccache`). `.devops/nix/package.nix` automatically detects `/tmp/gufo-ccache` if exposed to the Nix sandbox).*

### Step 2: Symlink Globally to `~/.local/bin` (One-Time Setup)
```fish
mkdir -p ~/.local/bin
ln -sf "$PWD/result/bin/gufo" ~/.local/bin/gufo
ln -sf "$PWD/result/bin/gufo-server" ~/.local/bin/gufo-server
```
Because `~/.local/bin/gufo` points to `$PWD/result/bin/gufo`, subsequent `nix build -L -o result` runs automatically update the binary in-place.

### Step 3: Verify Standalone Execution
Verify that the binary runs without any inherited shell environment variables (simulating execution from an external proxy, `systemd` unit, or different working directory):
```fish
env -i ~/.local/bin/gufo --version
```

---

## 3. Strix Halo (64 GB Unified RAM) Kernel Prerequisites

To use large out-of-core MoE memory budgets (e.g., `--memory-budget-gb 47`), the Linux kernel must permit `amdgpu` and `ttm` to allocate beyond the default 50% RAM (~32 GiB) GTT ceiling.

Ensure `/etc/modprobe.d/amdgpu-strix-halo.conf` contains:
```ini
# Allow amdgpu/ttm to allocate up to ~56 GiB of unified memory (in 4KiB pages)
options amdgpu gttsize=57344
options ttm pages_limit=14680064
options ttm page_pool_size=14680064
```
After editing, regenerate initramfs (`sudo mkinitcpio -P` on CachyOS) and reboot.

---

## 4. Production Runtime Commands (Qwen3.8-Flash-Next Out-of-Core MoE)

### A. OpenAI/Anthropic HTTP Server (100k Context, Proxy-Ready)
```fish
~/.local/bin/gufo serve llm \
  --model $model \
  --speculative mtp \
  --mtp-model $mtp \
  --served-model-name gufo \
  --sessions 1 \
  --context 102400 \
  --memory-budget-gb 47 \
  --cache-ram-bytes 2147483648 \
  --ple-shelf \
  --cache-prior-bias 0.05 \
  --cache-policy lru \
  --log-progress \
  --ssd-stats \
  --port 8080
```

**Why these specific flags matter on 64 GB Strix Halo:**
* `--sessions 1`: Preallocates only a single session's KV and Gated DeltaNet recurrent state so extra unused session slots don't steal GiBs from the expert cache.
* `--context 102400`: Fits comfortably within native 262k context (~2.3 GiB KV across the 12 Qwen Sparse Attention layers) without needing `--enable-yarn`.
* `--memory-budget-gb 47`: Maximizes resident MoE experts in the GTT pool while leaving safe OS headroom.
* `--cache-ram-bytes 2147483648` (2 GiB): **Critical.** Leaving this at `0` (auto) allows Flash-Next to progressively pin up to 32 GiB of host RAM for session prefix snapshots, which will collide with the 47 GiB GTT budget and trigger an OOM.
* `--ple-shelf` & `--cache-prior-bias 0.05`: Keeps Per-Layer Embeddings in a 128 MiB RAM shelf and weights router priors into expert retention (~97.4% expert cache hit rate, ~24.4 tok/s).
* *Note:* `--prompt-lookup` is a `gufo prompt` CLI flag only; do not pass it to `gufo serve llm`.

### B. Direct CLI Prompting / Benchmarking
```fish
~/.local/bin/gufo prompt \
  --model $model \
  --speculative mtp \
  --mtp-model $mtp \
  --prompt-lookup \
  --memory-budget-gb 47 \
  --ple-shelf \
  --cache-prior-bias 0.05 \
  --cache-policy lru \
  --ssd-stats \
  -v \
  -n 512 \
  -p "Your prompt here"
```