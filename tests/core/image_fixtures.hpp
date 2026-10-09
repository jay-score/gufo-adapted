#ifndef GUFO_TESTS_CORE_IMAGE_FIXTURES_HPP_
#define GUFO_TESTS_CORE_IMAGE_FIXTURES_HPP_

#include <string_view>

namespace gufo::test {

// Synthetic 2x3 fixtures encoded with Pillow 12.2.0 / libwebp.
// Lossless RGBA rows: red/green, blue/yellow, magenta/cyan.
// Alpha values 255/128/64 verify straight RGB without compositing.
inline constexpr std::string_view kPng =
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAADCAYAAAC56t6BAAAAHUlEQVR4nAXBAQEAIAzAIDTZbW"
    "bz"
    "HUTy0ZRuzNFbkDAKPDywkK8AAAAASUVORK5CYII=";

inline constexpr std::string_view kLosslessWebP =
    "data:image/webp;base64,"
    "UklGRjwAAABXRUJQVlA4TDAAAAAvAYAAEC8gEEjaH3oNAUGR/6MJCIr8H40g26Zouz/"
    "Uwz70gnWB"
    "QCANQaQw0BH9jwU=";

inline constexpr std::string_view kLossyWebP =
    "data:image/webp;base64,"
    "UklGRkAAAABXRUJQVlA4IDQAAAAwAgCdASoCAAMAAMASJaACdLoB+AH4AARoAAD++iGX/"
    "3easNN3"
    "9a3/9aOfron+tHP/WVgA";

inline constexpr std::string_view kOrientedWebP =
    "data:image/webp;base64,"
    "UklGRnAAAABXRUJQVlA4WAoAAAAYAAAAAQAAAgAAVlA4TDAAAAAvAYAAEC8gEEjaH3oNAUGR/"
    "6MJ"
    "CIr8H40g26Zouz/"
    "Uwz70gnWBQCANQaQw0BH9jwVFWElGGgAAAE1NACoAAAAIAAEBEgADAAAAAQAG"
    "AAAAAAAA";

inline constexpr std::string_view kAnimatedWebP =
    "data:image/webp;base64,"
    "UklGRp4AAABXRUJQVlA4WAoAAAASAAAAAQAAAgAAQU5JTQYAAAAAAAAAAABBTk1GQgAAAAAAAA"
    "AA"
    "AAEAAAIAAGQAAAJWUDhMKgAAAC8BgAAQLyAQSNofeo35FxAU+T+"
    "aQCAJbbPVgxtURw8IBNIQRAoD"
    "HdH/aEFOTUYoAAAAAAAAAAAAAQAAAgAAZAAAAFZQOEwPAAAALwGAAAAHENH//gciov8BAA==";

}  // namespace gufo::test
#endif
