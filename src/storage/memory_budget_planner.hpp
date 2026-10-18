#ifndef GUFO_STORAGE_MEMORY_BUDGET_PLANNER_HPP_
#define GUFO_STORAGE_MEMORY_BUDGET_PLANNER_HPP_

#include <cstddef>
#include <string>

namespace gufo::storage {

struct MemoryBudgetInputs {
  double budget_gb{0.0};
  std::size_t backbone_bytes{0};
  std::size_t drafter_bytes{0};
  std::size_t ple_shelf_bytes{0};
  std::size_t kv_reserved_bytes{0};
  std::size_t scratch_bytes{0};
  std::size_t routed_expert_bytes{0};
};

struct MemoryBudgetPlan {
  std::size_t budget_bytes{0};
  std::size_t backbone_bytes{0};
  std::size_t drafter_bytes{0};
  std::size_t ple_shelf_bytes{0};
  std::size_t kv_reserved_bytes{0};
  std::size_t scratch_bytes{0};
  std::size_t fixed_bytes{0};
  std::size_t routed_expert_bytes{0};
  std::size_t available_for_experts_bytes{0};
  std::size_t expert_cache_bytes{0};
  bool fits_fixed{false};

  [[nodiscard]] std::string Format() const;
};

/// Plans a top-down expert-cache capacity. The budget uses binary GiB.
[[nodiscard]] bool PlanMemoryBudget(const MemoryBudgetInputs& inputs,
                                    MemoryBudgetPlan* plan,
                                    std::string* error_msg = nullptr);

}  // namespace gufo::storage

#endif  // GUFO_STORAGE_MEMORY_BUDGET_PLANNER_HPP_
