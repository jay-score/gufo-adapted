// Exercise every kept prefix and subsequent trunk logits. TeacherForce does
// not advance the MTP head; snapshot_test covers normal speculative snapshots.
// Explicit model test: no weights are downloaded by CTest.
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {
void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireSame(const qfn::Session& expected, const qfn::Session& actual) {
  Require(expected.Position() == actual.Position() &&
              std::equal(expected.Tokens().begin(), expected.Tokens().end(),
                         actual.Tokens().begin(), actual.Tokens().end()),
          "rollback changed committed tokens or position");
  Require(expected.Logits().size() == actual.Logits().size() &&
              std::memcmp(expected.Logits().data(), actual.Logits().data(),
                          expected.Logits().size_bytes()) == 0,
          "rollback or continuation changed raw logits");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr
        << "Usage: rollback_test --model FIRST.gguf --mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = 128, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model && model->HasMtp(), error);
    const auto mode = gufo::core::SessionMode::kSpeculative;
    auto reference = model->CreateSession(mode, 128, &error);
    auto tested = model->CreateSession(mode, 128, &error);
    Require(reference && tested, error);
    const auto tokens = model->Tokenize(
        "Write a Python function that returns the sum of the even numbers "
        "in a list. Include a short example and explain why it works.");
    Require(tokens.size() >= 24, "test prompt is too short");
    const auto prompt = std::span(tokens).first(16);
    const auto chain = std::span(tokens).subspan(16, 8);
    Require(tested->MaxVerifyWidth() == chain.size(),
            "test requires the complete eight-row verification range");
    Require(reference->Sync(prompt, &error), error);
    auto base = reference->SaveSnapshot(&error);
    Require(base != nullptr, error);
    std::vector<float> rows;
    unsigned cases = 0;
    for (std::size_t width = 2; width <= tested->MaxVerifyWidth(); ++width) {
      for (std::size_t keep = 1; keep <= width; ++keep) {
        // Repeated shapes cover initial execution, graph capture, and replay.
        for (unsigned repeat = 0; repeat < 3; ++repeat) {
          Require(reference->RestoreSnapshot(*base, &error) &&
                      tested->RestoreSnapshot(*base, &error),
                  error);
          Require(reference->TeacherForce(chain.first(keep), &rows, &error),
                  error);
          Require(tested->TeacherForce(chain.first(width), &rows, &error, false,
                                       keep),
                  error);
          RequireSame(*reference, *tested);
          for (std::size_t i = 0; i < 2; ++i) {
            const auto next = prompt.subspan(i, 1);
            Require(reference->TeacherForce(next, &rows, &error) &&
                        tested->TeacherForce(next, &rows, &error),
                    error);
            RequireSame(*reference, *tested);
          }
          ++cases;
        }
      }
    }
    Require(tested->RestoreSnapshot(*base, &error), error);
    Require(!tested->TeacherForce(chain.first(2), &rows, &error, false, 3) &&
                tested->IsValid() && tested->Position() == prompt.size(),
            "invalid kept prefix modified the session");
    tested->Reset();
    Require(tested->Sync(prompt, &error), error);
    Require(reference->RestoreSnapshot(*base, &error), error);
    RequireSame(*reference, *tested);
    std::cout << "PASS: " << cases
              << " rollback prefixes and trunk continuations exact\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
