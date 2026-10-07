# Other engines

How other engines reuse prompt state across requests, as of 2026-10-07. The
llama.cpp section is from source (local checkout `41abbfd59`, 2026-09-14). The
others are from public documentation, so check details against the source
before relying on them.

## Summary

| Engine | Storage across requests | Hybrid and recurrent models |
| --- | --- | --- |
| llama.cpp | Live slot, plus full-copy cache entries in host RAM. No automatic disk tier | Checkpoints of recurrent or SWA state only, inside the live slot |
| vLLM | Hash-keyed KV blocks shared across requests | Mamba state cached only at block boundaries, experimental |
| SGLang | Token radix tree sharing KV, extended to host RAM and storage by HiCache | Recurrent state as checkpoints on tree nodes; one unified tree since 2026-08 |
| LMCache | 256-token chunks keyed by hash, in RAM, on local disk or remote | Gated DeltaNet hybrids with vLLM `align` mode |
| ds4 | Disk KV cache with checkpoint-interval settings | Files appear to be full snapshots (inferred) |

The large serving engines all share KV and treat recurrent state as sparse
checkpoints. The local engines (llama.cpp, ds4, gufo today) store full
snapshots.

## llama.cpp

- **Live slot.** A request reuses the longest common prefix with its slot's
  tokens. Attention KV past the divergence point is truncated
  (`llama_memory_seq_rm`), not copied, so a conversation's history exists once.
- **Context checkpoints.**
  - Limits: `--ctx-checkpoints`, default 32 per slot (`common/arg.cpp:1695`,
    `common/common.h:629`). At least `--checkpoint-min-step` tokens apart,
    default 8,192 (`common/common.h:631`), plus one at the last user message.
  - Contents: only the state that cannot be truncated, meaning recurrent state
    or the sliding window, via `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`
    (`include/llama.h:912`, `tools/server/server-context.cpp:2363`).
  - Rollback restores the nearest checkpoint and truncates KV.
- **Host prompt cache.** `--cache-ram`, default 8 GiB (`common/arg.cpp:1713`).
  - When a slot switches conversation, its whole state is copied into one
    entry: all KV plus its checkpoints (`tools/server/server-task.cpp:1711`).
  - Inserting an entry deletes every cached prompt it fully contains.
    Eviction is oldest first.
  - A request takes the entry with the best common prefix, which must keep at
    least 25% of that entry. The state moves back into the slot and the entry
    is deleted (`:1793`).
- **Disk.** Manual only: `/slots/{id}?action=save|restore` with
  `--slot-save-path` writes a full state file.
- **`--cache-reuse N`.** Reuses KV chunks that are not part of the prefix by
  shifting positions. Attention-only.

## vLLM

- **Automatic prefix caching.** KV is held in fixed-size blocks keyed by a hash
  of the block's tokens and its parent. Requests share blocks.
- **Hybrid KV cache manager.** Each attention type is a group. A request's hit
  is the intersection of the hits of each group.
- **Mamba and GDN state.**
  - `--mamba-cache-mode align` stores the state only at block boundaries, so a
    hit resumes only at a block boundary.
  - The block size is forced large, so an attention page is at least as big as
    a Mamba state page: 784 tokens for Qwen3.6-27B.
  - `--enable-mamba-fine-grained-prefix-cache` adds a checkpoint at a shared
    prefix junction.
  - Marked experimental.

Sources:
[hybrid KV cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/),
[cache config](https://docs.vllm.ai/en/v0.22.1/api/vllm/config/cache/).

## SGLang

- **RadixAttention.** A radix tree keyed by tokens. Each node owns the KV of
  its span, and shared prefixes share nodes.
- **MambaRadixCache.** It keeps recurrent state as checkpoints on tree nodes.
  The SGLang team gives three reasons this is hard:
  - recurrent state is updated in place, so it cannot be rolled back to a
    prefix;
  - it is "orders of magnitude larger than the KVs of a single token";
  - it is reusable only "all or nothing".

  It keeps two LRU lists: KV is evicted from the leaves inward, while
  recurrent checkpoints can be evicted from any node.
- **Unified Radix Cache (2026-08).** One tree with components:
  - FULL: KV for the whole path;
  - SWA: only the trailing window;
  - MAMBA: checkpoints at exact positions.

  It replaced a growing set of special-case cache classes. Session-aware
  eviction uses a client `session_id` and reported 2.9–16.6% lower time to
  first token than LRU on SWE-bench agent traces.
- **HiCache.** Three tiers: GPU (L1), host RAM private to each instance (L2),
  and shared storage (L3: Mooncake, 3FS, NIXL, AIBrix, or a file backend).
  - Data moves in pages. Small pages mean more metadata and I/O; large pages
    mean fewer partial hits.
  - Write policies: `write_through`, `write_through_selective`, `write_back`.
  - Prefetch from L3 stops on `best_effort`, `wait_complete` or `timeout`.

Sources:
[HiCache design](https://docs.sglang.io/advanced_features/hicache_design.html),
[hybrid models in SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/),
[Unified Radix Cache](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache).

## LMCache

- KV in hash-keyed chunks of 256 tokens by default. Backends: CPU RAM, local
  disk, Redis or Valkey, remote.
- Gated DeltaNet hybrids are supported, with conditions:
  - the chunk size must be a multiple of vLLM's unified block size;
  - recurrent layers get their own cache objects;
  - vLLM must run Mamba `align` mode.
- Caveats from its documentation:
  - not bit-exact against a fresh run;
  - cached pages are opaque bytes and cannot be shared across different
    attention backends or kernel block sizes.

Sources:
[hybrid models](https://docs.lmcache.ai/mp/hybrid_models.html),
[storage backend](https://docs.lmcache.ai/advanced/lmcache.storage_backend.html).

## ds4

- A DeepSeek V4 Flash engine by antirez. gufo's DeepSeek V4 model code is
  adapted from it, without its disk-cache front end
  ([UPSTREAM.md](../../src/models/deepseek_v4_flash/UPSTREAM.md)).
- `--kv-disk-dir` and `--kv-disk-space-mb` keep prefixes across slot reuse and
  restarts.
- `--kv-cache-min-tokens`, `--kv-cache-cold-max-tokens`,
  `--kv-cache-continued-interval-tokens`, `--kv-cache-boundary-trim-tokens`
  and `--kv-cache-boundary-align-tokens` choose which boundaries are saved.
- Its documentation describes cache files as holding prompt text and model
  state, with no sharing between files. That suggests full snapshots
  (inferred). DeepSeek V4's compressed KV keeps them small: under 3.5 GiB at
  256k tokens, according to its README.

Sources: [README](https://github.com/antirez/ds4),
[SERVER.md](https://raw.githubusercontent.com/antirez/ds4/main/docs/SERVER.md).
