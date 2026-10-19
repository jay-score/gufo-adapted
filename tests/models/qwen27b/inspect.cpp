#include <algorithm>
#include <array>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"
#include "src/models/qwen/gemm_route.hpp"
#include "src/models/qwen/state.hpp"

// A CPU-only probe of the production loader and route resolver. This does not
// execute kernels; the variant runner requires separate GPU checks.
int main(int argc, const char* const* argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("usage: qwen27b_inspect MODEL.gguf");
    }
    std::string error;
    const auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
    if (!reader) {
      throw std::runtime_error(error);
    }
    const auto weights =
        gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
    if (!weights) {
      throw std::runtime_error(error);
    }
    if (weights->config.hidden_size != 5120 ||
        weights->config.vocab_size != 248320 || weights->layers.size() != 64) {
      throw std::runtime_error("variant probe requires Qwen3.8 27B");
    }
    using gufo::json::Value;
    using namespace gufo::models::qwen;
    Value result = Value::object();
    result["schema"] = "gufo-qwen27b-inspection/1";
    result["tensors"] = Value::array();
    std::set<const void*> bound;
    const auto add = [&](const gufo::models::QwenTensorRef& ref,
                         const char* role) {
      if (ref.empty()) {
        return;
      }
      const auto found = std::ranges::find_if(
          reader->GetTensors(),
          [&](const auto& tensor) { return tensor.data == ref.data; });
      if (found == reader->GetTensors().end()) {
        throw std::runtime_error("loader returned an unindexed tensor");
      }
      bound.insert(ref.data);
      Value row = Value::object();
      row["name"] = std::string(found->name);
      row["role"] = role;
      row["stored_type"] = std::string(gufo::core::ToString(ref.type));
      row["runtime_type"] = row["stored_type"];
      row["encoded_bytes"] = ref.EncodedSizeBytes();
      row["conversion"] = "none";
      row["shape"] = Value::array();
      for (const auto dimension : found->dimensions) {
        row["shape"].push_back(Value(dimension));
      }
      if (std::string_view(role) == "projection") {
        const auto k = found->dimensions.front();
        const auto m = ref.num_elements / k;
        row["routes"] = Value::object();
        const std::array modes{QwenGemmMode::kHipDecode,
                               QwenGemmMode::kHipPrefill,
                               QwenGemmMode::kHipMtp};
        const std::array names{"decode", "prefill", "verification"};
        for (std::size_t i = 0; i < modes.size(); ++i) {
          const auto route = ResolveQwenGemmRoute(
              {ref.type, i == 0 ? 1U : 8U, m, k, modes[i]});
          if (!route.accepted()) {
            throw std::runtime_error("rejected route for " +
                                     std::string(found->name));
          }
          row["routes"][names[i]] = std::string(QwenGemmRouteName(route.route));
        }
      }
      result["tensors"].push_back(std::move(row));
    };
    add(weights->token_embd, "embedding");
    add(weights->output_norm, "normalization");
    add(weights->output, "projection");
    using Layer = gufo::models::QwenLayerWeights;
    const std::array norms{&Layer::attn_norm, &Layer::ffn_norm,
                           &Layer::attn_q_norm, &Layer::attn_k_norm};
    const std::array parameters{&Layer::ssm_a, &Layer::ssm_conv1d,
                                &Layer::ssm_dt, &Layer::ssm_norm};
    const std::array projections{
        &Layer::attn_q,      &Layer::attn_k,   &Layer::attn_v,
        &Layer::attn_output, &Layer::attn_qkv, &Layer::attn_gate,
        &Layer::ssm_alpha,   &Layer::ssm_beta, &Layer::ssm_out,
        &Layer::ffn_gate,    &Layer::ffn_up,   &Layer::ffn_down};
    for (const auto& layer : weights->layers) {
      for (const auto member : norms)
        add(layer.*member, "normalization");
      for (const auto member : parameters)
        add(layer.*member, "ssm_parameter");
      for (const auto member : projections)
        add(layer.*member, "projection");
    }
    result["unclaimed_tensors"] = Value::array();
    for (const auto& tensor : reader->GetTensors()) {
      if (!bound.contains(tensor.data)) {
        result["unclaimed_tensors"].push_back(Value(std::string(tensor.name)));
      }
    }
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
