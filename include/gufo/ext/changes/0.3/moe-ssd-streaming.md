# MoE out-of-core SSD streaming subsystem and telemetry

Add `<gufo/ext/experimental/moe_streaming.hpp>` exposing the model-agnostic MoE
out-of-core NVMe expert cache manager (`gufo::storage::ExpertCacheManager`),
3-tier priority SSD batch reader (`gufo::storage::SsdBatchReader`), memory
budget planner (`gufo::storage::PlanMemoryBudget`), Desk/Room prefetch planner,
and SSD telemetry (`gufo::storage::SsdStats`). Extend `ext::TextDecodeStep` with
verified draft token counts and prefix/tail cold-miss and stall telemetry, and
add `ext::TextModelRunner::SsdStatsReport()`.
