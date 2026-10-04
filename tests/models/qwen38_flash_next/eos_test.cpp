// Explicit real-model end-of-turn checks. Keep EOS enabled: tg128 and the
// rollback probe deliberately continue past it and cannot cover this contract.
#include <algorithm>
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <latch>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/inference_backend.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace sampling = gufo::sampling;
namespace server = gufo::server;

namespace {

void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

void RequireExact(std::span<const float> expected,
                  std::span<const float> actual) {
  Require(expected.size() == actual.size() &&
              std::memcmp(expected.data(), actual.data(),
                          expected.size_bytes()) == 0,
          "full frontier logits differ from committed-token replay");
}

constexpr std::string_view kPrompt =
    "<|im_start|>user\nReply with exactly this sentence and nothing else: "
    "The quick brown fox jumps over the lazy dog.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n";

std::unique_ptr<qfn::Session> NewSession(
    const std::shared_ptr<qfn::Model>& model, bool speculative) {
  std::string error;
  auto session = model->CreateSession(
      speculative ? gufo::core::SessionMode::kSpeculative
                  : gufo::core::SessionMode::kAutoregressive,
      2048, &error);
  Require(session != nullptr, error);
  return session;
}

std::vector<std::int32_t> Reference(const std::shared_ptr<qfn::Model>& model,
                                    std::span<const std::int32_t> prompt) {
  auto session = NewSession(model, false);
  std::string error;
  Require(session->Sync(prompt, &error), error);
  sampling::SamplerState sampler({.seed = 47});
  std::vector<std::int32_t> tokens;
  for (unsigned i = 0; i < 128; ++i) {
    qfn::Session::DecodeResult step;
    Require(session->DecodeStep(1, sampler, &step, &error), error);
    tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
    if (step.stop) {
      Require(tokens.size() >= 7, "fixture is too short for EOS offsets 0-7");
      return tokens;
    }
  }
  throw std::runtime_error("reference fixture did not reach EOS");
}

void CheckRetainedState(qfn::Session& session, qfn::Session& reference,
                        sampling::SamplerState& sampler,
                        const qfn::Session::DecodeResult& step,
                        std::vector<std::int32_t>& history,
                        const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  for (auto token : step.tokens) {
    Require(!model->IsStopToken(token), "EOS leaked into committed tokens");
    Require(reference.Evaluate(token, &error), error);
    history.push_back(token);
  }
  Require(session.IsValid() && session.Position() == history.size() &&
              std::ranges::equal(session.Tokens(), history),
          "executed position is outside committed history");
  RequireExact(reference.Logits(), session.Logits());
  const auto retained = sampler.history();
  Require(retained.size() <= history.size() &&
              std::equal(retained.begin(), retained.end(),
                         history.end() - retained.size()),
          "sampler accepted an uncommitted token");
  if (step.stop)
    Require(!sampler.SaveDrawState().pending,
            "EOS was deferred as a continuation token");
}

void CheckSnapshot(qfn::Session& session, sampling::SamplerState sampler,
                   const std::shared_ptr<qfn::Model>& model) {
  std::string error;
  const auto snapshot = session.SaveSnapshot(&error);
  Require(snapshot != nullptr, error);
  auto restored = NewSession(model, true);
  Require(restored->RestoreSnapshot(*snapshot, &error), error);
  Require(restored->Position() == session.Position() &&
              std::ranges::equal(restored->Tokens(), session.Tokens()),
          "EOS snapshot changed committed history");
  RequireExact(session.Logits(), restored->Logits());
  auto replay_sampler = sampler;
  qfn::Session::DecodeResult expected, actual;
  // Continuing past EOS is intentional here: it tests the restored MTP head
  // and rollback state, rather than passing on two immediate empty stops.
  Require(session.DecodeStep(3, sampler, &expected, &error, false), error);
  Require(restored->DecodeStep(3, replay_sampler, &actual, &error, false),
          error);
  Require(expected.tokens == actual.tokens &&
              sampler.rng_state() == replay_sampler.rng_state() &&
              sampler.SaveDrawState().pending ==
                  replay_sampler.SaveDrawState().pending,
          "EOS snapshot changed continuation or sampling draws");
  RequireExact(session.Logits(), restored->Logits());
}

void CheckEngine(const std::shared_ptr<qfn::Model>& model) {
  const auto prompt = model->Tokenize(kPrompt);
  const auto completion = Reference(model, prompt);
  const std::array<std::size_t, 9> distances{
      0, 1, 2, 3, 4, 5, 6, 7, completion.size()};
  const std::array<sampling::SamplingConfig, 3> configs{{
      {.seed = 47},
      {.seed = 47, .repeat_penalty = 1.0001F},
      {.temperature = 0.7F, .top_k = 1, .seed = 47},
  }};
  std::set<std::size_t> stop_slots;
  for (const auto width : {1U, 2U}) {
    for (std::size_t config = 0; config < configs.size(); ++config) {
      for (const auto distance : distances) {
        std::vector<std::unique_ptr<qfn::Session>> sessions, references;
        std::vector<sampling::SamplerState> samplers;
        std::vector<std::vector<std::int32_t>> histories;
        std::array<bool, 2> stopped{};
        std::string error;
        for (unsigned row = 0; row < width; ++row) {
          const auto remaining = row == 0        ? distance
                                 : distance <= 7 ? 7 - distance
                                                 : 0;
          histories.emplace_back(prompt);
          histories.back().insert(histories.back().end(), completion.begin(),
                                  completion.end() - remaining);
          sessions.push_back(NewSession(model, true));
          references.push_back(NewSession(model, false));
          Require(sessions.back()->Sync(histories.back(), &error) &&
                      references.back()->Sync(histories.back(), &error),
                  error);
          const std::vector<sampling::TokenId> initial(histories.back().begin(),
                                                       histories.back().end());
          samplers.emplace_back(configs[config], initial);
        }
        for (unsigned cycle = 0; cycle < 32; ++cycle) {
          std::array<qfn::Session::DecodeResult, 2> decoded;
          std::vector<qfn::Session::DecodeRequest> requests;
          for (unsigned row = 0; row < width; ++row)
            if (!stopped[row])
              requests.push_back(
                  {sessions[row].get(), 8, &samplers[row], &decoded[row]});
          Require(qfn::Session::DecodeBatch(requests, &error), error);
          for (unsigned row = 0; row < width; ++row) {
            if (stopped[row])
              continue;
            const auto& step = decoded[row];
            Require(!step.tokens.empty() || step.stop,
                    "decode made no progress");
            CheckRetainedState(*sessions[row], *references[row], samplers[row],
                               step, histories[row], model);
            if (step.stop) {
              stop_slots.insert(step.tokens.size());
              stopped[row] = true;
            }
          }
          if (std::all_of(stopped.begin(), stopped.begin() + width,
                          [](bool stop) { return stop; }))
            break;
        }
        for (unsigned row = 0; row < width; ++row) {
          Require(stopped[row], "EOS offset fixture did not stop");
          if (distance == 0 || distance == 1 || distance == 7)
            CheckSnapshot(*sessions[row], samplers[row], model);
        }
        std::cout << "eos width=" << width << " config=" << config
                  << " distance=" << distance << " state_exact=1\n"
                  << std::flush;
      }
    }
  }
  Require(stop_slots.contains(0) && stop_slots.size() > 1,
          "EOS matrix missed the anchor or verification stop path");
  std::cout << "EOS committed-prefix slots:";
  for (const auto slot : stop_slots)
    std::cout << ' ' << slot;
  std::cout << '\n';
  // A less constrained prompt exercises real sampled residual corrections.
  // The exact-sentence fixture above can accept every proposal on some quants.
  const auto varied = model->Tokenize(
      "<|im_start|>user\nIn one sentence, explain why a small "
      "sailboat can sail upwind.<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  auto session = NewSession(model, true);
  auto reference = NewSession(model, false);
  std::string error;
  Require(session->Sync(varied, &error) && reference->Sync(varied, &error),
          error);
  std::vector<std::int32_t> history(varied);
  const std::vector<sampling::TokenId> initial(varied.begin(), varied.end());
  sampling::SamplerState sampler(
      {.temperature = 0.8F, .top_k = 40, .top_p = 0.9F, .seed = 47}, initial);
  std::size_t rejected_cycles = 0;
  bool stopped = false;
  for (unsigned cycle = 0; cycle < 64; ++cycle) {
    const auto before = session->Statistics();
    qfn::Session::DecodeResult step;
    Require(session->DecodeStep(8, sampler, &step, &error), error);
    CheckRetainedState(*session, *reference, sampler, step, history, model);
    const auto after = session->Statistics();
    if (!step.stop &&
        after.drafted - before.drafted > after.accepted - before.accepted) {
      ++rejected_cycles;
      Require(sampler.SaveDrawState().pending.has_value(),
              "sampled rejection lost its residual draw");
    }
    if (step.stop) {
      stopped = true;
      break;
    }
  }
  Require(stopped && rejected_cycles > 0,
          "sampled fixture missed EOS after residual correction");
  CheckSnapshot(*session, sampler, model);
  std::cout << "sampled rejection_cycles=" << rejected_cycles
            << " EOS=1 frontier_and_continuation_exact=1\n"
            << std::flush;
}

void CheckServing(const std::shared_ptr<qfn::Model>& model) {
  using Backend = server::InferenceBackend;
  using Finish = Backend::FinishReason;
  const sampling::SamplingConfig config{.seed = 47};
  for (const auto width : {1U, 2U}) {
    for (const bool mtp : {false, true}) {
      Backend backend;
      std::string error;
      server::TextSpeculativeConfig options;
      if (mtp)
        options.backend = server::TextSpeculativeBackend::kMtp;
      Require(backend.load(model, &error, 2048, width, {}, {}, options), error);
      auto full = backend.complete(kPrompt, 128, config);
      Require(full.finish_reason == Finish::kStop && full.tokens.size() >= 7 &&
                  full.completion_tokens == full.tokens.size(),
              "serving EOS or completion accounting failed");
      for (const auto budget : {1U, 2U, 3U, 8U, 128U}) {
        const auto result = backend.complete(kPrompt, budget, config);
        Require(result.tokens.size() ==
                        std::min<std::size_t>(budget, full.tokens.size()) &&
                    std::equal(result.tokens.begin(), result.tokens.end(),
                               full.tokens.begin()) &&
                    result.finish_reason == (budget <= full.tokens.size()
                                                 ? Finish::kLength
                                                 : Finish::kStop),
                "output budget changed completion or terminal reason");
      }
      auto continued =
          backend.complete(std::string(kPrompt) + full.text, 8, config);
      Require(continued.tokens.empty() &&
                  continued.finish_reason == Finish::kStop &&
                  continued.cache_hit,
              "warm EOS continuation retained hidden tokens or lost its "
              "checkpoint");
      for (const auto& stop : {full.text.substr(0, 3), std::string("fox")}) {
        const auto result =
            backend.complete(kPrompt, 128, config, {}, {}, "anonymous", {stop});
        const auto replay =
            backend.complete(kPrompt, 128, config, {}, {}, "anonymous", {stop});
        Require(result.finish_reason == Finish::kStopSequence &&
                    result.text == full.text.substr(0, full.text.find(stop)) &&
                    result.text == replay.text &&
                    result.tokens == replay.tokens,
                "stop sequence corrupted retained state or exposed its tail");
      }
      const auto cancelled = backend.complete(
          kPrompt, 128, config, {}, [](std::string_view) { return false; });
      Require(cancelled.cancelled, "callback cancellation was ignored");
      const auto after_cancel = backend.complete(kPrompt, 128, config);
      Require(
          after_cancel.tokens == full.tokens && after_cancel.text == full.text,
          "cancelled speculative state corrupted the next request");
      if (width == 2) {
        std::latch ready(2);
        auto run = [&] {
          ready.arrive_and_wait();
          return backend.complete(kPrompt, 128, config);
        };
        auto first = std::async(std::launch::async, run);
        auto second = std::async(std::launch::async, run);
        const auto a = first.get(), b = second.get();
        Require(a.tokens == full.tokens && b.tokens == full.tokens &&
                    a.finish_reason == Finish::kStop &&
                    b.finish_reason == Finish::kStop &&
                    std::max(a.physical_execution_width,
                             b.physical_execution_width) == 2,
                "concurrent EOS requests failed or did not exercise batching");
      }
      std::cout << "serving width=" << width << " mtp=" << mtp
                << " EOS/budgets/stops/cancellation/cache=pass decode_ms="
                << full.decode_ms << " completion=" << full.completion_tokens
                << '\n'
                << std::flush;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5 || std::string_view(argv[1]) != "--model" ||
      std::string_view(argv[3]) != "--mtp-model") {
    std::cerr << "Usage: eos_test --model FIRST.gguf --mtp-model MTP.gguf\n";
    return 77;
  }
  try {
    std::string error;
    auto model = qfn::Model::Load(
        argv[2],
        {.max_context = 2048, .mtp_model_path = argv[4], .max_draft_tokens = 7},
        &error);
    Require(model && model->HasMtp(), error);
    CheckEngine(model);
    CheckServing(model);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
