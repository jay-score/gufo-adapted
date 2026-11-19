#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_

#include <array>
#include <cstdint>

#include "src/storage/cache_policy.hpp"

namespace gufo::models::qwen38_flash_next {

/// Validated per-layer expert-cache slots for 48 streamed layers. The total is
/// 13,920 slots, the same total as a uniform 290 slots per layer. The storage
/// layer scales this shape to the budget-derived total when the cache is
/// smaller or larger.
inline constexpr std::array<std::uint32_t, 48> kSkewedSlotProfile = {
    460, 436, 456, 264, 312, 356, 380, 388, 332, 248, 312, 224,
    288, 304, 264, 156, 184, 296, 340, 240, 272, 364, 408, 276,
    292, 252, 272, 260, 284, 268, 284, 184, 220, 280, 288, 240,
    248, 300, 364, 228, 220, 260, 244, 276, 276, 288, 244, 288,
};

/// Measured decode cold-miss curves for the 48 streamed layers, used by the
/// storage layer's marginal slot allocator. Each row holds three
/// (slots, rate_milli_percent) points taken from the same router/bias/MTP
/// configuration at three expert-cache budgets:
///
///   - 34 GiB / 3 GiB / 32K:
///     benchmark_logs/probe-34gb-slot-curve.log
///   - 44 GiB / 6 GiB / 64K:
///     benchmark_logs/gufo-upstream-44gb-crb6g-c64k-rb6144.log
///   - 47 GiB / 3 GiB / 32K:
///     benchmark_logs/gufo-upstream-47gb-crb3g-rb6144.log
///
/// `rate_milli_percent` is cold misses per expert request in 0.001 % units.
/// The allocator interpolates within a row, extrapolates with the nearest
/// segment, and clamps at zero, so rows must stay monotone; a unit test pins
/// the anchor allocations.
inline constexpr std::array<gufo::storage::SlotMarginalCurve, 48>
    kSlotMarginalCurves = {{
        {{233, 7585}, {380, 400}, {444, 10}},
        {{225, 4034}, {362, 239}, {421, 35}},
        {{231, 3938}, {377, 196}, {440, 13}},
        {{171, 1757}, {231, 313}, {257, 143}},
        {{186, 3243}, {268, 286}, {303, 84}},
        {{200, 3550}, {301, 670}, {345, 256}},
        {{207, 4819}, {319, 1133}, {368, 400}},
        {{210, 6409}, {325, 1353}, {375, 476}},
        {{192, 4230}, {283, 1127}, {322, 642}},
        {{166, 4979}, {219, 1997}, {242, 1485}},
        {{186, 6926}, {268, 2564}, {303, 1743}},
        {{158, 5283}, {201, 2741}, {219, 2260}},
        {{178, 6802}, {250, 2915}, {280, 2102}},
        {{183, 7470}, {262, 3077}, {295, 2214}},
        {{171, 4833}, {231, 2432}, {257, 1636}},
        {{137, 1089}, {149, 630}, {155, 566}},
        {{146, 2064}, {171, 1051}, {181, 928}},
        {{181, 3903}, {256, 1276}, {288, 906}},
        {{195, 5916}, {289, 2186}, {330, 1369}},
        {{163, 4236}, {213, 1961}, {234, 1531}},
        {{173, 2880}, {237, 1139}, {265, 858}},
        {{202, 3629}, {307, 1176}, {352, 600}},
        {{216, 5006}, {341, 913}, {394, 207}},
        {{175, 1297}, {240, 494}, {269, 336}},
        {{180, 4307}, {253, 1550}, {284, 1078}},
        {{167, 2912}, {222, 1471}, {246, 1066}},
        {{173, 5942}, {237, 2382}, {265, 1807}},
        {{170, 3110}, {228, 1047}, {254, 692}},
        {{177, 5173}, {246, 2129}, {276, 1530}},
        {{172, 5953}, {234, 2705}, {261, 2043}},
        {{177, 3395}, {246, 1047}, {276, 654}},
        {{146, 432}, {171, 132}, {181, 124}},
        {{157, 982}, {198, 230}, {215, 165}},
        {{176, 3258}, {243, 808}, {273, 434}},
        {{178, 6334}, {250, 2689}, {280, 1684}},
        {{163, 2641}, {213, 891}, {234, 675}},
        {{166, 1282}, {219, 374}, {242, 227}},
        {{182, 2575}, {259, 823}, {292, 391}},
        {{202, 4310}, {307, 820}, {352, 239}},
        {{159, 395}, {204, 93}, {223, 61}},
        {{157, 506}, {198, 137}, {215, 98}},
        {{170, 999}, {228, 158}, {254, 93}},
        {{165, 2211}, {216, 698}, {238, 372}},
        {{175, 369}, {240, 75}, {269, 45}},
        {{175, 673}, {240, 81}, {269, 38}},
        {{178, 771}, {250, 100}, {280, 52}},
        {{164, 1346}, {216, 393}, {238, 199}},
        {{178, 1702}, {250, 234}, {280, 73}},
    }};

}  // namespace gufo::models::qwen38_flash_next

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_SLOT_PROFILE_HPP_
