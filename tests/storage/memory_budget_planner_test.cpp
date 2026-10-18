#include "src/storage/memory_budget_planner.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr std::size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
int failures = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

void TestBudgetArithmetic() {
  const gufo::storage::MemoryBudgetInputs inputs{
      .budget_gb = 8.0,
      .backbone_bytes = 1 * kGiB,
      .drafter_bytes = 2 * kGiB,
      .ple_shelf_bytes = kGiB / 8,
      .kv_reserved_bytes = 1 * kGiB,
      .scratch_bytes = kGiB / 4,
      .routed_expert_bytes = 10 * kGiB,
  };
  gufo::storage::MemoryBudgetPlan plan;
  std::string error;
  Check(gufo::storage::PlanMemoryBudget(inputs, &plan, &error),
        "valid budget should plan");
  Check(plan.fits_fixed, "fixed allocations fit in 8 GiB");
  Check(plan.fixed_bytes == (35 * kGiB) / 8,
        "fixed allocation components sum without unit conversion drift");
  Check(plan.available_for_experts_bytes == (29 * kGiB) / 8,
        "remaining budget is reserved for experts");
  Check(plan.expert_cache_bytes == plan.available_for_experts_bytes,
        "cache capacity is clamped to remaining budget");
  Check(plan.Format().find("expert cache capacity") != std::string::npos,
        "startup breakdown contains cache capacity");

  auto small_weights = inputs;
  small_weights.routed_expert_bytes = kGiB;
  Check(gufo::storage::PlanMemoryBudget(small_weights, &plan, &error),
        "budget with smaller expert weights should plan");
  Check(plan.expert_cache_bytes == kGiB,
        "cache capacity is capped at the total routed expert bytes");
}

void TestBudgetUnderflowAndInputs() {
  gufo::storage::MemoryBudgetPlan plan;
  std::string error;
  const gufo::storage::MemoryBudgetInputs too_small{
      .budget_gb = 1.0,
      .backbone_bytes = 2 * kGiB,
      .routed_expert_bytes = 8 * kGiB,
  };
  Check(gufo::storage::PlanMemoryBudget(too_small, &plan, &error),
        "a too-small but valid budget returns a diagnostic plan");
  Check(!plan.fits_fixed && plan.available_for_experts_bytes == 0 &&
            plan.expert_cache_bytes == 0,
        "fixed-allocation overflow cannot underflow the expert capacity");

  for (const double invalid :
       {0.0, -1.0, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    auto inputs = too_small;
    inputs.budget_gb = invalid;
    Check(!gufo::storage::PlanMemoryBudget(inputs, &plan, &error),
          "non-positive and non-finite budgets are rejected");
    Check(!error.empty(), "invalid budget returns an explanation");
  }

  auto overflow = too_small;
  overflow.budget_gb = 32.0;
  overflow.backbone_bytes = std::numeric_limits<std::size_t>::max();
  overflow.drafter_bytes = 1;
  Check(!gufo::storage::PlanMemoryBudget(overflow, &plan, &error),
        "fixed component overflow is rejected");
}

}  // namespace

int main() {
  TestBudgetArithmetic();
  TestBudgetUnderflowAndInputs();
  if (failures != 0) {
    std::cerr << failures << " memory budget planner check(s) failed\n";
    return 1;
  }
  std::cout << "memory_budget_planner_test: all checks passed\n";
  return 0;
}
