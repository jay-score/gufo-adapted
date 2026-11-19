# Marginal expert-cache slot allocation

Add `storage::SlotMarginalPoint` and `storage::SlotMarginalCurve`, a monotone
piecewise-linear per-layer decode cold-miss curve in milli-percent units.

`ExpertCacheManager::ScaleSlotProfile` and `ExpertCacheManager::Create` accept an
optional `std::span<const SlotMarginalCurve>` and, when supplied, distribute the
budget-derived slot total with a deterministic greedy marginal allocator instead
of linear scaling above the floor. Extra slots go to the layer with the largest
measured cold-miss reduction, ties break by layer index, and residual slots
spill round-robin toward `num_experts` once every curve saturates. The exact-sum,
floor/cap, dense-layer, and fallback contracts are unchanged, and omitting the
curves keeps the previous linear behavior.
