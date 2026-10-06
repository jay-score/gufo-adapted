// Session snapshot round trips: a restored session must continue exactly
// like the session it was captured from, in memory and through the
// persistent byte form, at a prompt boundary and mid-decode.
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

struct Decoded {
  std::vector<std::int32_t> tokens;
  qfn::Session::SpeculativeStats stats;
};

Decoded Decode(qfn::Session& session, std::size_t count,
               sampling::SamplerState& sampler, std::size_t budget) {
  std::string error;
  const auto before = session.Statistics();
  Decoded out;
  while (out.tokens.size() < count) {
    qfn::Session::DecodeResult step;
    Require(session.DecodeStep(std::min(budget, count - out.tokens.size()),
                               sampler, &step, &error, false),
            error);
    Require(!step.tokens.empty(), "empty decode step");
    out.tokens.insert(out.tokens.end(), step.tokens.begin(), step.tokens.end());
  }
  const auto after = session.Statistics();
  out.stats = {after.cycles - before.cycles, after.drafted - before.drafted,
               after.accepted - before.accepted};
  return out;
}

/// Decodes `count` tokens past EOS with a fresh seeded sampler.
Decoded Decode(qfn::Session& session, std::size_t count,
               const sampling::SamplingConfig& config, std::size_t budget) {
  const auto history = session.Tokens();
  const std::vector<sampling::TokenId> initial(history.begin(), history.end());
  sampling::SamplerState sampler(config, initial);
  return Decode(session, count, sampler, budget);
}

void RequireSame(const Decoded& expected, const Decoded& actual,
                 const std::string& what) {
  Require(expected.tokens == actual.tokens, what + ": tokens differ");
  Require(expected.stats.drafted == actual.stats.drafted &&
              expected.stats.accepted == actual.stats.accepted,
          what + ": draft acceptance differs");
}

double Millis(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

std::vector<std::uint8_t> Payload(const qfn::SessionSnapshot& snapshot) {
  std::vector<std::uint8_t> bytes(snapshot.SizeBytes());
  Require(snapshot.CopyTo(bytes), "snapshot export failed");
  std::size_t offset = 0;
  snapshot.StreamTo([&](auto chunk) {
    Require(
        chunk.size() <= bytes.size() - offset &&
            std::memcmp(bytes.data() + offset, chunk.data(), chunk.size()) == 0,
        "streamed snapshot differs from contiguous export");
    offset += chunk.size();
  });
  Require(offset == bytes.size(), "streamed snapshot is truncated");
  return bytes;
}

void CheckIncrementalSnapshots(const std::shared_ptr<qfn::Model>& model,
                               std::span<const std::int32_t> prompt,
                               gufo::core::SessionMode mode) {
  std::string error;
  auto source = model->CreateSession(mode, 8192, &error);
  auto reference = model->CreateSession(mode, 8192, &error);
  Require(source && reference, error);
  Require(source->Sync(prompt.first(2047), &error), error);
  auto prefix = source->SaveSnapshot(&error);
  Require(prefix && prefix->SharedDeviceBytes() == 0,
          "initial snapshot unexpectedly shares rows");
  const auto prefix_bytes = Payload(*prefix);
  Require(reference->RestoreSnapshot(prefix_bytes, &error), error);
  Require(source->Sync(prompt, &error) && reference->Sync(prompt, &error),
          error);
  auto start = std::chrono::steady_clock::now();
  auto grown = source->SaveSnapshot(&error);
  const auto save_ms = Millis(start);
  auto complete = reference->SaveSnapshot(&error);
  Require(grown && complete && grown->SharedDeviceBytes() != 0,
          "growing snapshot did not share its captured prefix");
  Require(complete->SharedDeviceBytes() == 0,
          "raw restore retained unproven snapshot provenance");
  Require(grown->CopiedDeviceBytes() + grown->SharedDeviceBytes() ==
              complete->CopiedDeviceBytes(),
          "incremental capture byte accounting differs from full capture");
  const auto grown_bytes = Payload(*grown);
  Require(grown_bytes == Payload(*complete),
          "incremental snapshot differs from independent full capture");
  auto same = source->SaveSnapshot(&error);
  Require(same && same->CopiedDeviceBytes() < grown->CopiedDeviceBytes(),
          "unchanged checkpoint recopied its KV history");
  Require(Payload(*same) == grown_bytes,
          "sharing an unchanged prefix changed snapshot bytes");

  // Rewind, overwrite a suffix, and compare against an independent raw restore.
  std::vector<std::int32_t> branch(prompt.begin(), prompt.end());
  std::rotate(branch.begin() + 2047, branch.begin() + 2079, branch.end());
  Require(source->RestoreSnapshot(*prefix, &error) &&
              reference->RestoreSnapshot(prefix_bytes, &error),
          error);
  Require(source->Sync(branch, &error) && reference->Sync(branch, &error),
          error);
  auto branched = source->SaveSnapshot(&error);
  complete = reference->SaveSnapshot(&error);
  Require(branched && complete && branched->SharedDeviceBytes() != 0, error);
  Require(Payload(*branched) == Payload(*complete),
          "branch reused overwritten snapshot rows");
  Require(Payload(*grown) == grown_bytes,
          "branch changed an independent older checkpoint");

  // Resetting and recomputing identical tokens must drop all provenance.
  auto reader = std::async(std::launch::async, [&] { return Payload(*grown); });
  source->Reset();
  Require(source->Sync(prompt, &error), error);
  auto reset = source->SaveSnapshot(&error);
  Require(reset && reset->SharedDeviceBytes() == 0,
          "reset reused a token-equal but unproven cache prefix");
  Require(reader.get() == grown_bytes,
          "concurrent export depended on the source session");

  // Failed capture drains its queued copies and does not replace the base.
  unsigned checks = 0;
  source->SetCancellationCheck([&] { return ++checks > 3; });
  Require(!source->SaveSnapshot(&error), "cancelled capture succeeded");
  source->SetCancellationCheck({});
  auto retry = source->SaveSnapshot(&error);
  Require(retry && retry->SharedDeviceBytes() != 0, error);
  Require(Payload(*retry) == Payload(*reset),
          "cancelled capture damaged the last completed checkpoint");
  retry.reset();
  auto expired = source->SaveSnapshot(&error);
  Require(expired && expired->SharedDeviceBytes() == 0,
          "session retained an evicted checkpoint");

  // Both the direct chunk restore and contiguous disk bytes survive
  // destruction.
  source.reset();
  Require(reference->RestoreSnapshot(*grown, &error), error);
  auto survivor = reference->SaveSnapshot(&error);
  Require(survivor && Payload(*survivor) == grown_bytes,
          "source destruction invalidated shared checkpoint blocks");
  Require(reference->RestoreSnapshot(grown_bytes, &error), error);
  complete = reference->SaveSnapshot(&error);
  Require(complete && Payload(*complete) == grown_bytes,
          "version-15 persistent snapshot round trip changed bytes");
  std::cout << "incremental mode=" << static_cast<int>(mode)
            << " copied=" << grown->CopiedDeviceBytes()
            << " shared=" << grown->SharedDeviceBytes()
            << " save_ms=" << save_ms << " exact=1\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr
        << "Usage: snapshot_test --model FIRST.gguf --mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    constexpr std::uint32_t kContext = 8192;
    auto model = qfn::Model::Load(argv[2],
                                  {.max_context = kContext,
                                   .mtp_model_path = argv[4],
                                   .max_draft_tokens = 7},
                                  &error);
    Require(model != nullptr, error);
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    Require(!pattern.empty(), "empty prompt pattern");
    // Partial prefill batch and three unpooled raw indexer rows at the end
    // of the ring. Continuing (and speculative rollback) crosses its wrap.
    std::vector<std::int32_t> prompt(4095);
    for (std::size_t i = 0; i < prompt.size(); ++i)
      prompt[i] = pattern[i % pattern.size()];
    CheckIncrementalSnapshots(model, prompt,
                              gufo::core::SessionMode::kAutoregressive);
    CheckIncrementalSnapshots(model, prompt,
                              gufo::core::SessionMode::kSpeculative);
    const sampling::SamplingConfig config{
        .temperature = 0.8F, .top_k = 40, .top_p = 0.9F, .seed = 7};
    constexpr std::size_t kTokens = 48;

    auto origin = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(origin != nullptr, error);
    Require(origin->Sync(prompt, &error), error);
    const std::vector<float> prompt_logits(origin->Logits().begin(),
                                           origin->Logits().end());

    auto start = std::chrono::steady_clock::now();
    auto at_prompt = origin->SaveSnapshot(&error);
    Require(at_prompt != nullptr, error);
    const double save_ms = Millis(start);
    Require(at_prompt->SizeBytes() == origin->SnapshotBytes(),
            "snapshot size differs from the estimate");
    std::cout << "snapshot tokens=" << prompt.size()
              << " bytes=" << at_prompt->SizeBytes() << " save_ms=" << save_ms
              << "\n";

    const Decoded expected = Decode(*origin, kTokens, config, 8);
    Require(expected.stats.drafted > 0, "MTP did not draft");

    // Restore into a fresh session.
    auto restored = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(restored != nullptr, error);
    start = std::chrono::steady_clock::now();
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    std::cout << "restore_ms=" << Millis(start) << "\n";
    Require(std::equal(prompt.begin(), prompt.end(), restored->Tokens().begin(),
                       restored->Tokens().end()),
            "restored tokens differ");
    Require(restored->Position() == prompt.size(), "restored position");
    Require(std::memcmp(prompt_logits.data(), restored->Logits().data(),
                        prompt_logits.size() * sizeof(float)) == 0,
            "restored logits differ");
    RequireSame(expected, Decode(*restored, kTokens, config, 8),
                "fresh session restore");

    // The persistent byte form restores the same way.
    std::vector<std::uint8_t> bytes(at_prompt->SizeBytes());
    Require(at_prompt->CopyTo(bytes), "snapshot copy");
    auto persisted = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    Require(persisted != nullptr, error);
    Require(persisted->RestoreSnapshot(bytes, &error), error);
    RequireSame(expected, Decode(*persisted, kTokens, config, 8),
                "persistent restore");

    // Restoring over a session that already decoded discards its state.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    RequireSame(expected, Decode(*origin, kTokens, config, 8),
                "restore over a used session");

    // Single-token decoding from a restored session also matches.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    const Decoded single = Decode(*origin, 16, config, 1);
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    Require(single.tokens == Decode(*restored, 16, config, 1).tokens,
            "single-token decode differs after restore");

    // A mid-decode snapshot carries the draft block's caught-up state and
    // the partial batch of kept trunk rows.
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    const Decoded first_half = Decode(*origin, 16, config, 8);
    auto mid = origin->SaveSnapshot(&error);
    Require(mid != nullptr, error);
    const Decoded second_half = Decode(*origin, kTokens, config, 8);
    Require(restored->RestoreSnapshot(*mid, &error), error);
    Require(restored->Position() == prompt.size() + first_half.tokens.size(),
            "mid-decode position");
    // Controller state is part of a stochastic continuation's replay.
    RequireSame(second_half, Decode(*restored, kTokens, config, 8),
                "mid-decode restore");

    // Capture a rejected proposal before its residual has been evaluated.
    // The context snapshot and the request-owned sampler must replay together.
    auto pending = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        kContext, &error);
    const auto short_prompt = std::span(prompt).first(16);
    Require(pending && pending->Sync(short_prompt, &error), error);
    const std::vector<sampling::TokenId> short_history(short_prompt.begin(),
                                                       short_prompt.end());
    sampling::SamplerState sampler({.temperature = 2.0F, .seed = 73},
                                   short_history);
    bool rejected = false;
    for (unsigned attempt = 0; attempt < 32 && !rejected; ++attempt) {
      qfn::Session::DecodeResult step;
      Require(pending->DecodeStep(2, sampler, &step, &error, false), error);
      rejected = step.tokens.size() == 1;
    }
    Require(rejected, "snapshot test did not exercise a deferred residual");
    auto residual_snapshot = pending->SaveSnapshot(&error);
    Require(residual_snapshot != nullptr, error);
    auto replay_sampler = sampler;
    const auto residual_expected = Decode(*pending, 16, sampler, 8);
    Require(restored->RestoreSnapshot(*residual_snapshot, &error), error);
    RequireSame(residual_expected, Decode(*restored, 16, replay_sampler, 8),
                "deferred residual restore");
    Require(sampler.rng_state() == replay_sampler.rng_state(),
            "deferred residual restore changed RNG");

    // Extending the restored context keeps the prefix.
    Require(restored->RestoreSnapshot(*at_prompt, &error), error);
    std::vector<std::int32_t> extended = prompt;
    extended.insert(extended.end(), expected.tokens.begin(),
                    expected.tokens.end());
    Require(restored->Sync(extended, &error), error);
    Require(origin->RestoreSnapshot(*at_prompt, &error), error);
    Require(origin->Sync(extended, &error), error);
    Require(std::memcmp(origin->Logits().data(), restored->Logits().data(),
                        prompt_logits.size() * sizeof(float)) == 0,
            "extension after restore differs");

    // Cached state from older prefill arithmetic must be rebuilt.
    auto incompatible = bytes;
    const std::uint32_t old_version = qfn::Session::kSnapshotPayloadVersion - 1;
    std::memcpy(incompatible.data() + 8, &old_version, sizeof(old_version));
    Require(!restored->RestoreSnapshot(incompatible, &error),
            "incompatible snapshot version accepted");
    Require(restored->Position() == extended.size(),
            "incompatible snapshot disturbed the session");

    // Rejections: truncated payload, and a context too small to hold it.
    Require(!restored->RestoreSnapshot(
                std::span<const std::uint8_t>(bytes).first(bytes.size() / 2),
                &error),
            "truncated payload accepted");
    Require(restored->Position() == extended.size(),
            "rejected payload disturbed the session");
    auto small = model->CreateSession(
        model->HasMtp() ? gufo::core::SessionMode::kSpeculative
                        : gufo::core::SessionMode::kAutoregressive,
        2048, &error);
    Require(small != nullptr, error);
    Require(!small->RestoreSnapshot(*at_prompt, &error),
            "oversized snapshot accepted");
    std::cout << "snapshot round trips exact=1\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << "\n";
    return 1;
  }
}
