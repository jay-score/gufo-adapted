# Continuation cache redesign

Research started 2026-10-07. Nothing in this directory changes server
behaviour; it records the question, the evidence and the decision.

**Question.** Every cached checkpoint today is a full, self-contained copy of
the model state, in RAM and on disk. Should gufo move to a design that stores
shared KV once and keeps only per-checkpoint state separately, and if so, how
far should it go?

| Document | Contents |
| --- | --- |
| [Current design](current-design.md) | What the continuation cache does today, with source references and known costs |
| [Other engines](external-engines.md) | llama.cpp, vLLM, SGLang, LMCache and ds4, with links |
| [Options](options.md) | Candidate designs, pros and cons, and the features still to decide |
| [Experiments](experiments.md) | Measurements needed to decide, and their results |

## Status

1. Write down the discussion and external references. Done.
2. Run the experiments in [experiments.md](experiments.md). Not started.
3. Agree on the required features (see [Options](options.md)).
4. Choose a design, or keep the current one.

## Decision log

### 2026-10-07

**#275 and #409.** #275 reports that one long conversation fills the disk
budget and evicts every other conversation. A plan was posted on the issue
([comment](https://github.com/gufo-org/gufo/issues/275#issuecomment-6041775357)):
close it when [#409](https://github.com/gufo-org/gufo/pull/409) merges, and
decline eager K=2 supersession, a per-conversation byte cap, per-lineage
logging and a checkpoint-interval flag. The fixed 2,048-token disk spacing from
#348 is a good performance tradeoff.

#409 evicts covered intermediate disk checkpoints before falling back to LRU
and logs them as `reason=superseded`. It was rebased onto main `b39c530e`, and
review fixes were pushed to the contributor's branch (head `223e8515`):
learned shared-prefix checkpoints are never treated as covered, the eviction
scan walks the prefix tree by node instead of by token, and the log-line table
gained the `superseded` row. **Its merge is on hold**: if the redesign goes
ahead, chunk reference counting would replace its policy.

**Conclusions from the discussion.**

- Each checkpoint is a full snapshot of the model state at one position. A
  Flash-Next RAM checkpoint borrows the append-only KV rows of its live
  session (#445), but the RAM budget still counts its full size. Disk files are
  always full copies.
- No design document explains why disk files are full copies. The code
  explains it: snapshots are opaque to the cache, each file is verified,
  published and evicted on its own, and several processes may share one
  directory.
- Hashing the prompt at each turn would not add anything. The existing token
  prefix tree already finds the longest stored prefix exactly. The gain of a
  "diff" design comes from storing shared KV once.
- Recurrent state cannot be diffed. Any design keeps a full copy of it for
  every position it can restore.
- llama.cpp uses the diff idea only inside a live slot. Its cache across
  conversations stores full copies and deletes prompts contained in a newer
  one.
- vLLM, SGLang and LMCache share KV in blocks or radix-tree nodes and keep
  recurrent state as sparse checkpoints. Most of their machinery (per-token
  paging, host and remote tiers, prefetch policies, elastic pools, session
  IDs) addresses a scale gufo does not have: at most 8 concurrent requests,
  one machine, and unified memory.
- The smallest useful candidate: KV in 2,048-token chunks shared across
  checkpoints, a full copy of fixed state per checkpoint, the existing prefix
  tree and eviction ranks, and budgets that count each chunk once.
- The gain grows with context length. For Flash-Next, a checkpoint 2,048
  tokens past its parent costs about 2× less than a full copy at 8k tokens
  and about 24× less at 145k.

**Next.** Measure before choosing (see [Experiments](experiments.md)), then
discuss the required features.
