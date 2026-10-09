#include "src/core/image.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "tests/core/image_fixtures.hpp"

namespace {
using gufo::core::DecodeImage;
using gufo::core::ReadImageUrl;

void ExpectRejected(const std::vector<std::uint8_t>& bytes,
                    std::string_view reason) {
  try {
    (void)DecodeImage(bytes);
    assert(false && "invalid image accepted");
  } catch (const std::invalid_argument& error) {
    assert(std::string_view(error.what()).find(reason) !=
           std::string_view::npos);
  }
}

void TestWebP() {
  const auto bytes = ReadImageUrl(gufo::test::kLosslessWebP);
  const auto image = DecodeImage(bytes);
  const auto png = DecodeImage(ReadImageUrl(gufo::test::kPng));
  const std::vector<std::uint8_t> expected{
      255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 255, 0, 255, 0, 255, 255};
  assert(image.width == 2 && image.height == 3);
  assert(image.pixels == expected && image.pixels == png.pixels);

  const auto lossy = DecodeImage(ReadImageUrl(gufo::test::kLossyWebP));
  assert(lossy.width == 2 && lossy.height == 3);
  for (std::size_t i = 0; i < lossy.pixels.size(); ++i)
    assert(std::abs(int(lossy.pixels[i]) - (i % 3 == 0 ? 255 : 0)) <= 3);

  const auto oriented = DecodeImage(ReadImageUrl(gufo::test::kOrientedWebP));
  const std::vector<std::uint8_t> rotated{
      255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0};
  assert(oriented.width == 3 && oriented.height == 2);
  assert(oriented.pixels == rotated);

  ExpectRejected(ReadImageUrl(gufo::test::kAnimatedWebP), "animated WebP");
  auto truncated = bytes;
  truncated.pop_back();
  ExpectRejected(truncated, "truncated WebP");
  auto corrupt = bytes;
  corrupt[20] = 0;  // Invalid VP8L signature in an otherwise intact RIFF.
  ExpectRejected(corrupt, "WebP");
  auto short_payload = bytes;
  short_payload.resize(26);
  short_payload[4] = 18;  // Complete RIFF and VP8L headers, missing pixel data.
  short_payload[16] = 6;
  ExpectRejected(short_payload, "WebP image");
  auto oversized = bytes;
  // VP8L width/height occupy the low 28 bits after its signature. The header
  // is valid but must be rejected before allocating a 16384x16384 RGB buffer.
  oversized[21] = oversized[22] = oversized[23] = 0xff;
  oversized[24] |= 0x0f;
  ExpectRejected(oversized, "pixel limit");

  auto bad_chunk = bytes;
  bad_chunk[16] = 0xff;
  ExpectRejected(bad_chunk, "WebP chunk");
  auto truncated_exif = ReadImageUrl(gufo::test::kOrientedWebP);
  truncated_exif.pop_back();
  ExpectRejected(truncated_exif, "truncated WebP");
}

void TestWebPTransportBudget() {
  const auto bytes = ReadImageUrl(gufo::test::kLosslessWebP);
  gufo::core::ImageReadBudget budget;
  budget.remaining_bytes = bytes.size();
  budget.remaining_images = 1;
  assert(ReadImageUrl(gufo::test::kLosslessWebP, budget) == bytes);
  assert(budget.remaining_bytes == 0 && budget.remaining_images == 0);
  try {
    (void)ReadImageUrl(gufo::test::kLosslessWebP, budget);
    assert(false && "request image budget bypassed");
  } catch (const std::invalid_argument&) {
  }
  for (const auto url : {"data:image/webp;base64,!!!!", "data:image/webp,RIFF",
                         "data:image/gif;base64,AQID"}) {
    try {
      (void)ReadImageUrl(url);
      assert(false && "invalid data URL accepted");
    } catch (const std::invalid_argument&) {
    }
  }
}
}  // namespace

int main() {
  TestWebP();
  TestWebPTransportBudget();
  std::cout << "image_test: passed\n";
}
