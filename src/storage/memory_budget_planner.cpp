#include "src/storage/memory_budget_planner.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace gufo::storage {
namespace {

constexpr long double kBytesPerGiB = 1024.0L * 1024.0L * 1024.0L;
constexpr double kBytesPerGiBDouble = 1024.0 * 1024.0 * 1024.0;

bool Add(std::size_t lhs, std::size_t rhs, std::size_t* sum) {
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    return false;
  *sum = lhs + rhs;
  return true;
}

double GiB(std::size_t bytes) {
  return static_cast<double>(bytes) / kBytesPerGiBDouble;
}

}  // namespace

std::string MemoryBudgetPlan::Format() const {
  std::ostringstream out;
  out << std::fixed << std::setprecision(3);
  out << "Memory budget (GiB)\n"
      << "  target:                 " << GiB(budget_bytes) << '\n'
      << "  backbone:               " << GiB(backbone_bytes) << '\n'
      << "  drafter:                " << GiB(drafter_bytes) << '\n'
      << "  PLE shelf:              " << GiB(ple_shelf_bytes) << '\n'
      << "  KV/state reserved:      " << GiB(kv_reserved_bytes) << '\n'
      << "  executor scratch:       " << GiB(scratch_bytes) << '\n'
      << "  fixed total:            " << GiB(fixed_bytes) << '\n'
      << "  available for experts:  " << GiB(available_for_experts_bytes)
      << '\n'
      << "  routed expert weights:  " << GiB(routed_expert_bytes) << '\n'
      << "  expert cache capacity:  " << GiB(expert_cache_bytes) << '\n'
      << "  fixed allocations fit:  " << (fits_fixed ? "yes" : "no") << '\n';
  return out.str();
}

bool PlanMemoryBudget(const MemoryBudgetInputs& inputs, MemoryBudgetPlan* plan,
                      std::string* error_msg) {
  if (error_msg != nullptr)
    error_msg->clear();
  if (plan == nullptr) {
    if (error_msg != nullptr)
      *error_msg = "memory budget plan output is null";
    return false;
  }
  *plan = {};
  if (!std::isfinite(inputs.budget_gb) || inputs.budget_gb <= 0.0) {
    if (error_msg != nullptr)
      *error_msg = "memory budget must be a finite positive number of GiB";
    return false;
  }
  const long double target =
      static_cast<long double>(inputs.budget_gb) * kBytesPerGiB;
  if (target >
      static_cast<long double>(std::numeric_limits<std::size_t>::max())) {
    if (error_msg != nullptr)
      *error_msg = "memory budget exceeds the host size range";
    return false;
  }

  plan->budget_bytes = static_cast<std::size_t>(target);
  plan->backbone_bytes = inputs.backbone_bytes;
  plan->drafter_bytes = inputs.drafter_bytes;
  plan->ple_shelf_bytes = inputs.ple_shelf_bytes;
  plan->kv_reserved_bytes = inputs.kv_reserved_bytes;
  plan->scratch_bytes = inputs.scratch_bytes;
  plan->routed_expert_bytes = inputs.routed_expert_bytes;
  std::size_t fixed = 0;
  if (!Add(fixed, inputs.backbone_bytes, &fixed) ||
      !Add(fixed, inputs.drafter_bytes, &fixed) ||
      !Add(fixed, inputs.ple_shelf_bytes, &fixed) ||
      !Add(fixed, inputs.kv_reserved_bytes, &fixed) ||
      !Add(fixed, inputs.scratch_bytes, &fixed)) {
    if (error_msg != nullptr)
      *error_msg = "fixed memory components overflow the host size range";
    *plan = {};
    return false;
  }
  plan->fixed_bytes = fixed;
  plan->fits_fixed = fixed <= plan->budget_bytes;
  if (!plan->fits_fixed)
    return true;

  plan->available_for_experts_bytes = plan->budget_bytes - fixed;
  plan->expert_cache_bytes =
      std::min(plan->available_for_experts_bytes, inputs.routed_expert_bytes);
  return true;
}

}  // namespace gufo::storage
