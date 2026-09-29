// Appended images must preserve the exact text/image prefix, including restored
// and speculative state. --baseline records the original cold behavior.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace vision = gufo::models::qwen::vision;
namespace tok = gufo::tokenization;

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::size_t Common(const vision::Prompt& a, const vision::Prompt& b) {
  return std::mismatch(a.tokens.begin(), a.tokens.end(), b.tokens.begin(),
                       b.tokens.end())
             .first -
         a.tokens.begin();
}

std::vector<float> Score(qfn::Session& session,
                         const std::vector<std::int32_t>& teacher,
                         const std::string& label) {
  std::vector<float> all;
  double nll = 0;
  std::string error;
  for (const auto token : teacher) {
    const auto logits = session.Logits();
    all.insert(all.end(), logits.begin(), logits.end());
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (const auto value : logits) {
      Require(std::isfinite(value), "nonfinite logit");
      sum += std::exp(static_cast<double>(value - maximum));
    }
    nll += std::log(sum) + maximum - logits[token];
    Require(session.Evaluate(token, &error), error);
  }
  std::cout << label << " sha256="
            << gufo::crypto::Sha256Hex(
                   {reinterpret_cast<const std::uint8_t*>(all.data()),
                    all.size() * sizeof(float)})
            << " ppl=" << std::exp(nll / teacher.size()) << '\n';
  return all;
}

void Equal(const std::vector<float>& a, const std::vector<float>& b) {
  Require(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "cached continuation changed full logits");
}

std::vector<std::int32_t> Decode(qfn::Session& session, bool sampled) {
  gufo::sampling::SamplingConfig config;
  config.temperature = sampled ? 0.8F : 0;
  config.top_k = 20;
  config.top_p = 0.95F;
  config.seed = 47;
  const auto history = session.Tokens();
  const std::vector<gufo::sampling::TokenId> initial(history.begin(),
                                                     history.end());
  gufo::sampling::SamplerState sampler(config, initial);
  std::vector<std::int32_t> result;
  std::string error;
  while (result.size() < 16) {
    qfn::Session::DecodeResult step;
    Require(session.DecodeStep(std::min<std::size_t>(8, 16 - result.size()),
                               sampler, &step, &error, false),
            error);
    Require(!step.tokens.empty(), "empty decode");
    result.insert(result.end(), step.tokens.begin(), step.tokens.end());
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "Usage: image_prefix_test MODEL MTP MMPROJ IMAGE_A IMAGE_B "
                 "[--baseline]\n";
    return 77;
  }
  try {
    const bool baseline =
        argc == 7 && std::string_view(argv[6]) == "--baseline";
    Require(argc == 6 || baseline, "unknown argument");
    constexpr std::uint32_t context = 8192;
    std::string error;
    auto model = qfn::Model::Load(argv[1],
                                  {.max_context = context,
                                   .mtp_model_path = argv[2],
                                   .max_draft_tokens = 7,
                                   .vision_model_path = argv[3]},
                                  &error);
    Require(model != nullptr && model->VisionEncoder() != nullptr, error);
    std::string text;
    for (int i = 0; i < 128; ++i)
      text += "A screenshot extends this unchanged conversation. ";
    std::vector<tok::ChatMessage> messages;
    messages.emplace_back(tok::ChatRole::kUser, text);
    tok::ChatTemplateOptions options;
    options.enable_thinking = false;
    const auto prepare = [&] {
      return std::make_shared<vision::Prompt>(
          vision::Prepare(model->tokenizer(), messages, {}, options,
                          model->VisionEncoder()->identity(), context));
    };
    auto p0 = prepare();
    messages.emplace_back(tok::ChatRole::kAssistant, "Ready.");
    messages.emplace_back(tok::ChatRole::kUser, "Describe this screenshot.");
    messages.back().images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                gufo::core::ReadImageFile(argv[4]))});
    auto p1 = prepare();
    messages.emplace_back(tok::ChatRole::kAssistant,
                          "The first screenshot is visible.");
    messages.emplace_back(tok::ChatRole::kUser,
                          "Describe the next screenshot.");
    messages.back().images.push_back(
        {0, std::make_shared<const std::vector<std::uint8_t>>(
                gufo::core::ReadImageFile(argv[5]))});
    auto p2 = prepare();
    const std::vector<std::shared_ptr<vision::Prompt>> prompts{p0, p1, p2};
    const std::vector<std::size_t> ends{Common(*p0, *p2), Common(*p1, *p2),
                                        p2->tokens.size()};
    const std::vector<std::int32_t> tokens(p2->tokens.begin(),
                                           p2->tokens.end());
    const auto teacher =
        model->Tokenize("The screenshot shows a colored pattern.");
    for (const auto mode : {gufo::core::SessionMode::kAutoregressive,
                            gufo::core::SessionMode::kSpeculative}) {
      const bool mtp = mode == gufo::core::SessionMode::kSpeculative;
      auto warm = model->CreateSession(mode, context, &error);
      Require(warm != nullptr, error);
      std::unique_ptr<qfn::SessionSnapshot> previous;
      for (std::size_t stage = 0; stage < prompts.size(); ++stage) {
        const auto history = std::span(tokens).first(ends[stage]);
        warm->ConfigureVision(prompts[stage]);
        if (!baseline && stage > 0)
          Require(warm->Position() == ends[stage - 1],
                  "append reset the live prefix");
        Require(warm->Sync(history, &error), error);
        auto snapshot = warm->SaveSnapshot(&error);
        Require(snapshot != nullptr, error);
        const auto label =
            "mtp=" + std::to_string(mtp) + " stage=" + std::to_string(stage);
        const auto actual = Score(*warm, teacher, label + " warm");
        auto cold = model->CreateSession(mode, context, &error);
        Require(cold != nullptr, error);
        cold->ConfigureVision(prompts[stage]);
        Require(cold->Sync(history, &error), error);
        Equal(actual, Score(*cold, teacher, label + " cold"));
        Require(warm->RestoreSnapshot(*snapshot, &error), error);
        if (!baseline && previous) {
          cold->ConfigureVision(prompts[stage]);
          Require(cold->RestoreSnapshot(*previous, &error), error);
          Require(cold->Position() == ends[stage - 1],
                  "snapshot prefix position");
          Require(cold->Sync(history, &error), error);
          Equal(actual, Score(*cold, teacher, label + " restored"));
        }
        if (stage == 2) {
          for (const bool sampled : {false, true}) {
            cold->Reset();
            cold->ConfigureVision(prompts[stage]);
            Require(cold->Sync(history, &error), error);
            Require(warm->RestoreSnapshot(*snapshot, &error), error);
            Require(Decode(*warm, sampled) == Decode(*cold, sampled),
                    "cached greedy/seeded decode differs");
          }
          if (!baseline) {
            // Save midway through an image, then attach another image. The
            // entire first image identity must constrain even a partial state.
            const auto partial = p1->images.front().grid.offset + 13;
            cold->Reset();
            cold->ConfigureVision(p1);
            Require(cold->Sync(std::span(tokens).first(partial), &error),
                    error);
            auto partial_snapshot = cold->SaveSnapshot(&error);
            Require(partial_snapshot != nullptr, error);
            warm->ConfigureVision(p2);
            Require(warm->RestoreSnapshot(*partial_snapshot, &error), error);
            Require(warm->Sync(history, &error), error);
            Equal(actual, Score(*warm, teacher, label + " partial-image"));

            // A cancelled append can safely restart from the earlier snapshot.
            Require(warm->RestoreSnapshot(*previous, &error), error);
            warm->SetCancellationCheck([] { return true; });
            Require(!warm->Sync(history, &error), "cancellation was ignored");
            warm->SetCancellationCheck({});
            warm->ConfigureVision(p2);
            Require(warm->RestoreSnapshot(*previous, &error), error);
            Require(warm->Sync(history, &error), error);
            Equal(actual, Score(*warm, teacher, label + " cancelled-resumed"));
          }
          // Changing an earlier image at the same token positions must reset.
          const auto original_image = messages[2].images[0];
          messages[2].images[0] = messages.back().images[0];
          auto changed = prepare();
          warm->ConfigureVision(changed);
          Require(warm->Position() == 0,
                  "changed earlier image retained stale state");
          Require(!warm->RestoreSnapshot(*snapshot, &error),
                  "changed image accepted an incompatible snapshot");
          messages[2].images[0] = original_image;
          // Text restore must discard images affecting the restored tokens.
          cold->Reset();
          cold->ConfigureVision(nullptr);
          const std::array<std::int32_t, 2> plain{42, 43};
          Require(cold->Sync(plain, &error), error);
          const auto plain_snapshot = cold->SaveSnapshot(&error);
          Require(plain_snapshot != nullptr, error);
          auto at_start = std::make_shared<vision::Prompt>(*p1);
          at_start->images.front().grid.offset = 0;
          at_start->rope.images.front().offset = 0;
          warm->ConfigureVision(at_start);
          Require(warm->RestoreSnapshot(*plain_snapshot, &error), error);
          Equal(Score(*cold, teacher, label + " plain"),
                Score(*warm, teacher, label + " cleared-image"));
        } else {
          Require(warm->RestoreSnapshot(*snapshot, &error), error);
        }
        previous = std::move(snapshot);
      }
    }
    std::cout << "image prefix checks passed baseline=" << baseline << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
