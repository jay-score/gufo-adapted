#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <thread>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/platform/tuning.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Conversion {
  void* source;
  void* destination;
  std::size_t count;
};

/// Dense formats the HIP GEMM tier cannot run natively (Swift IQ4_XS dense,
/// Q5_K/Q6_K attention). Re-encoded to Q8_0 at upload; Q8_0's per-32 step is
/// finer than the coarser source blocks, so the added rounding is small next
/// to the source quantization it sits on (same argument as the original
/// Q6_K head conversion).
[[nodiscard]] bool NeedsQ8Conversion(core::GgmlType type) noexcept {
  using core::GgmlType;
  return type == GgmlType::kQ4_K || type == GgmlType::kQ5_K ||
         type == GgmlType::kQ6_K || type == GgmlType::kQ5_1 ||
         type == GgmlType::kIQ3_S || type == GgmlType::kIQ4_XS ||
         type == GgmlType::kIQ4_NL;
}

struct BlockQ5_1 {
  std::uint16_t d;
  std::uint16_t m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(BlockQ5_1) == 24);

void DequantizeQ5_1Row(const std::uint8_t* src, float* dst, std::size_t k) {
  for (std::size_t b = 0; b < k / 32; ++b) {
    BlockQ5_1 blk;
    std::memcpy(&blk, src + b * sizeof(BlockQ5_1), sizeof(BlockQ5_1));
    const float d = gufo::quant::Fp16ToFloat(blk.d);
    const float m = gufo::quant::Fp16ToFloat(blk.m);
    float* y = dst + b * 32;
    for (int j = 0; j < 16; ++j) {
      const int xh0 = ((blk.qh >> j) & 1U) << 4;
      const int xh1 = ((blk.qh >> (j + 16)) & 1U) << 4;
      y[j] = static_cast<float>((blk.qs[j] & 0x0F) | xh0) * d + m;
      y[j + 16] = static_cast<float>((blk.qs[j] >> 4) | xh1) * d + m;
    }
  }
}

/// Dequantizes one row of `t` (expert 0, dense has no experts) into `out`.
void DequantizeDenseRow(const TensorRef& t, std::size_t row, float* out) {
  const std::uint8_t* src = t.Expert(0) + row * t.RowBytes();
  const std::size_t k = t.cols;
  switch (t.type) {
    case core::GgmlType::kQ4_K:
      gufo::quant::DequantizeQ4_K(src, out, k);
      break;
    case core::GgmlType::kQ5_K:
      gufo::quant::DequantizeQ5_K(src, out, k);
      break;
    case core::GgmlType::kQ6_K:
      gufo::quant::DequantizeQ6_K(src, out, k);
      break;
    case core::GgmlType::kQ5_1:
      DequantizeQ5_1Row(src, out, k);
      break;
    case core::GgmlType::kIQ3_S:
      gufo::quant::DequantizeIQ3_S(src, out, k);
      break;
    case core::GgmlType::kIQ4_XS:
      gufo::quant::DequantizeIQ4_XS(src, out, k);
      break;
    case core::GgmlType::kIQ4_NL:
      gufo::quant::DequantizeIQ4_NL(src, out, k);
      break;
    default:
      std::memset(out, 0, k * sizeof(float));
      break;
  }
}

/// Re-encodes `rows` rows of `t` (starting at `r0`) as Q8_0 (fp16 scale +
/// 32 int8 per block) into `dst`. Uses the core dequantizers so the
/// conversion matches the CPU oracle's reading of the source format.
void ConvertRowsToQ8_0(const TensorRef& t, std::uint8_t* dst, std::size_t r0,
                       std::size_t rows) {
  const std::size_t cols = t.cols;
  const std::size_t dst_row = cols / 32 * 34;
  std::vector<float> values(cols);
  for (std::size_t r = 0; r < rows; ++r) {
    DequantizeDenseRow(t, r0 + r, values.data());
    std::uint8_t* out = dst + r * dst_row;
    for (std::size_t b = 0; b < cols / 32; ++b) {
      const float* x = values.data() + b * 32;
      float amax = 0.0F;
      for (int i = 0; i < 32; ++i)
        amax = std::max(amax, std::fabs(x[i]));
      const auto scale = static_cast<_Float16>(amax / 127.0F);
      const float d = static_cast<float>(scale);
      const float inverse = d != 0.0F ? 1.0F / d : 0.0F;
      std::memcpy(out + b * 34, &scale, sizeof(scale));
      auto* q = reinterpret_cast<std::int8_t*>(out + b * 34 + 2);
      for (int i = 0; i < 32; ++i) {
        const float v = std::nearbyint(x[i] * inverse);
        q[i] = static_cast<std::int8_t>(std::clamp(v, -127.0F, 127.0F));
      }
    }
  }
}

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<Conversion>& conversions;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  /// Layer() leaves the routed experts for a later Experts() pass.
  bool defer_experts{false};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor Copy(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t size = t.SizeBytes();
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset, size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = static_cast<std::uint32_t>(t.experts);
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && t.experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  /// A K-quant or I-quant dense matrix (Q6_K output head of unsloth
  /// UD-IQ4_XS, Swift IQ4_XS dense projections) re-encoded as Q8_0 on the
  /// host, so it runs on the tuned Q8_0 dense tier instead of a path the HIP
  /// kernels do not have.
  DeviceTensor CopyConvertedAsQ8_0(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (!NeedsQ8Conversion(t.type) || t.cols % 32 != 0 || t.experts != 1 ||
        t.RowBytes() == 0) {
      Fail("tensor " + std::string(t.name) + " has an unsupported shape");
      return d;
    }
    const std::size_t dst_row = t.cols / 32 * 34;
    const std::size_t size = dst_row * t.rows;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name));
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    constexpr std::size_t kChunkRows = 16384;
    std::vector<std::uint8_t> host(kChunkRows * dst_row);
    const std::size_t workers =
        std::max<std::size_t>(1, std::thread::hardware_concurrency());
    for (std::size_t r0 = 0; r0 < t.rows; r0 += kChunkRows) {
      const std::size_t rows = std::min<std::size_t>(kChunkRows, t.rows - r0);
      const std::size_t per = (rows + workers - 1) / workers;
      std::vector<std::jthread> pool;
      for (std::size_t w = 0; w < workers && w * per < rows; ++w) {
        const std::size_t begin = w * per;
        const std::size_t count = std::min(per, rows - begin);
        pool.emplace_back([&, begin, count] {
          ConvertRowsToQ8_0(t, host.data() + begin * dst_row, r0 + begin,
                            count);
        });
      }
      pool.clear();
      if (hipMemcpy(static_cast<std::uint8_t*>(ptr) + r0 * dst_row, host.data(),
                    rows * dst_row, hipMemcpyHostToDevice) != hipSuccess) {
        Fail("upload failed for " + std::string(t.name));
        return d;
      }
    }
    (void)hipMemset(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin);
    d.data = ptr;
    d.type = core::GgmlType::kQ8_0;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    return d;
  }

  /// Dense projections: native Q8_0/BF16/F16/F32 upload, otherwise re-encode
  /// to Q8_0. Routed experts keep their own Copy path (native MMQ kernels).
  DeviceTensor CopyDense(const TensorRef& t) {
    if (t.empty() || !ok) {
      return DeviceTensor{};
    }
    return NeedsQ8Conversion(t.type) ? CopyConvertedAsQ8_0(t) : Copy(t);
  }

  /// Small F32-only convolution weights; Swift stores ple_conv1d as F16.
  /// Converts half precision to F32 on the host, leaving F32 untouched.
  DeviceTensor CopyAsF32(const TensorRef& t) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    if (t.type == core::GgmlType::kF32) {
      return Copy(t);
    }
    if ((t.type != core::GgmlType::kF16 &&
         t.type != core::GgmlType::kBF16) ||
        t.experts != 1) {
      Fail("tensor " + std::string(t.name) + " has an unsupported format");
      return d;
    }
    const std::size_t count = t.cols * t.rows;
    const std::size_t size = count * sizeof(float);
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name));
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::vector<float> host(count);
    const auto* src = static_cast<const std::uint16_t*>(t.data);
    if (t.type == core::GgmlType::kF16) {
      for (std::size_t i = 0; i < count; ++i) {
        host[i] = gufo::quant::Fp16ToFloat(src[i]);
      }
    } else {
      for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t bits = static_cast<std::uint32_t>(src[i]) << 16;
        std::memcpy(&host[i], &bits, sizeof(bits));
      }
    }
    if (hipMemcpy(ptr, host.data(), size, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin) !=
            hipSuccess) {
      Fail("upload failed for " + std::string(t.name));
      return d;
    }
    d.data = ptr;
    d.type = core::GgmlType::kF32;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = 1;
    return d;
  }

  // GGUF packs [fc_embedding | fc_hidden] across each row. Split on a
  // quantization-block boundary without dequantizing or changing any weight.
  void SplitMtpProjection(const TensorRef& t, DeviceTensor& embedding,
                          DeviceTensor& hidden) {
    if (t.empty() || !ok)
      return;
    // Converted dense (Swift-style IQ/K) uploads as Q8_0 first; the split
    // below then works on the Q8_0 row layout.
    const bool converted = NeedsQ8Conversion(t.type);
    const auto combined = converted ? CopyConvertedAsQ8_0(t) : Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("MTP projection upload failed");
      return;
    }
    const std::size_t full_row =
        converted ? combined.cols / 32 * 34 : t.SizeBytes() / t.rows;
    const std::size_t row_bytes = full_row / 2;
    const std::size_t part_bytes = row_bytes * t.rows;
    const std::size_t combined_size =
        converted ? full_row * t.rows : t.SizeBytes();
    for (std::uint32_t part = 0; part < 2; ++part) {
      auto& dst = part == 0 ? embedding : hidden;
      dst = combined;
      dst.cols /= 2;
      if (hipMalloc(&dst.data, part_bytes + kTailMargin) != hipSuccess) {
        Fail("MTP split projection allocation failed");
        return;
      }
      allocations.push_back(dst.data);
      bytes += part_bytes + kTailMargin;
      const auto* src =
          static_cast<const std::uint8_t*>(combined.data) + part * row_bytes;
      if (hipMemcpy2D(dst.data, row_bytes, src, 2 * row_bytes, row_bytes,
                      t.rows, hipMemcpyDeviceToDevice) != hipSuccess ||
          hipMemset(static_cast<std::uint8_t*>(dst.data) + part_bytes, 0,
                    kTailMargin) != hipSuccess) {
        Fail("MTP split projection copy failed");
        return;
      }
    }
    std::erase(allocations, combined.data);
    (void)hipFree(combined.data);
    bytes -= combined_size + kTailMargin;
  }

  /// Uploads matrices of one type stacked along rows; every input shares
  /// `cols`. An F32 stack (router logits, GDN alpha/beta: the only
  /// unquantized projections) is narrowed to F16, which the wide-batch GEMM
  /// tier runs at speed. A Q8_0 stack merges projections of one input into
  /// a single decode GEMV.
  DeviceTensor Stack(std::initializer_list<const TensorRef*> parts) {
    DeviceTensor d;
    if (!ok) {
      return d;
    }
    std::size_t rows = 0;
    std::size_t size = 0;
    const core::GgmlType type = (*parts.begin())->type;
    for (const TensorRef* t : parts) {
      if (t->empty() || t->type != type || t->cols != (*parts.begin())->cols ||
          (type != core::GgmlType::kF32 && type != core::GgmlType::kQ8_0)) {
        Fail("stacked upload needs F32 or Q8_0 tensors of one shape");
        return d;
      }
      rows += t->rows;
      size += t->SizeBytes();
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    std::size_t offset = 0;
    for (const TensorRef* t : parts) {
      if (!stager.Copy(shard_base + t->shard, t->file_offset, t->SizeBytes(),
                       static_cast<std::uint8_t*>(ptr) + offset, error)) {
        Fail("upload failed for " + std::string(t->name));
        return d;
      }
      offset += t->SizeBytes();
    }
    if (type == core::GgmlType::kQ8_0) {
      (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0,
                           kTailMargin, nullptr);
      d.data = ptr;
      d.type = type;
      d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
      d.rows = static_cast<std::uint32_t>(rows);
      max_q8_cols = std::max<std::size_t>(max_q8_cols, d.cols);
      return d;
    }
    const std::size_t count = size / sizeof(float);
    void* half = nullptr;
    if (hipMalloc(&half, count * sizeof(std::uint16_t) + kTailMargin) !=
        hipSuccess) {
      Fail("hipMalloc failed for stacked tensor");
      return d;
    }
    allocations.push_back(half);
    bytes += count * sizeof(std::uint16_t) + kTailMargin;
    // Keep the small F32 stacks until the disk pipeline drains. Converting
    // each router immediately would serialize every layer's uploads.
    conversions.push_back({ptr, half, count});
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(half) + count * 2, 0,
                         kTailMargin, nullptr);
    d.data = half;
    d.type = core::GgmlType::kF16;
    d.cols = static_cast<std::uint32_t>((*parts.begin())->cols);
    d.rows = static_cast<std::uint32_t>(rows);
    max_half_cols = std::max<std::size_t>(max_half_cols, d.cols);
    return d;
  }

  DeviceMixer Mixer(const HcMixer& m) {
    return {Copy(m.norm), CopyDense(m.down), CopyDense(m.up),
            CopyDense(m.inject)};
  }

  DeviceLayer Layer(const LayerWeights& l) {
    DeviceLayer d;
    d.linear = l.linear;
    d.hc_attn = Mixer(l.hc_attn);
    d.hc_ffn = Mixer(l.hc_ffn);
    // Projections of one input are stacked into one Q8_0 GEMV where the
    // quantization allows; otherwise they stay separate.
    const auto stackable = [](std::initializer_list<const TensorRef*> parts) {
      for (const TensorRef* t : parts) {
        if (t->empty() || t->type != core::GgmlType::kQ8_0 ||
            t->cols != (*parts.begin())->cols) {
          return false;
        }
      }
      return true;
    };
    if (l.linear && stackable({&l.ssm_qkv, &l.ssm_gate})) {
      d.ssm_in = Stack({&l.ssm_qkv, &l.ssm_gate});
    } else {
      d.ssm_qkv = CopyDense(l.ssm_qkv);
      d.ssm_gate = CopyDense(l.ssm_gate);
    }
    d.ssm_conv1d = CopyAsF32(l.ssm_conv1d);
    if (l.linear) {
      d.ssm_alpha_beta = Stack({&l.ssm_alpha, &l.ssm_beta});
    }
    d.ssm_dt = Copy(l.ssm_dt);
    d.ssm_a = Copy(l.ssm_a);
    d.ssm_norm = Copy(l.ssm_norm);
    d.ssm_out = CopyDense(l.ssm_out);
    if (!l.linear && stackable({&l.attn_q, &l.attn_k, &l.attn_v})) {
      d.attn_qkv = Stack({&l.attn_q, &l.attn_k, &l.attn_v});
    } else {
      d.attn_q = CopyDense(l.attn_q);
      d.attn_k = CopyDense(l.attn_k);
      d.attn_v = CopyDense(l.attn_v);
    }
    d.attn_out = CopyDense(l.attn_out);
    d.attn_q_norm = Copy(l.attn_q_norm);
    d.attn_k_norm = Copy(l.attn_k_norm);
    d.indexer_q = CopyDense(l.indexer_q);
    d.indexer_k = CopyDense(l.indexer_k);
    d.indexer_q_norm = Copy(l.indexer_q_norm);
    d.indexer_k_norm = Copy(l.indexer_k_norm);
    d.ple_key = CopyDense(l.ple_key);
    d.ple_value = CopyDense(l.ple_value);
    d.ple_norm_key = Copy(l.ple_norm_key);
    d.ple_norm_query = Copy(l.ple_norm_query);
    d.ple_norm_conv = Copy(l.ple_norm_conv);
    d.ple_conv1d = CopyAsF32(l.ple_conv1d);
    d.router = Stack({&l.router, &l.shexp_gate_inp});
    if (!defer_experts) {
      Experts(l, d);
    }
    d.shexp_gate = CopyDense(l.shexp_gate);
    d.shexp_up = CopyDense(l.shexp_up);
    d.shexp_down = CopyDense(l.shexp_down);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    SplitMtpProjection(l.nextn_eh_proj, d.nextn_fc_embedding,
                       d.nextn_fc_hidden);
    d.nextn_head = Mixer(l.nextn_head);
    return d;
  }

  void Experts(const LayerWeights& l, DeviceLayer& d) {
    d.ffn_gate_exps = Copy(l.ffn_gate_exps);
    d.ffn_up_exps = Copy(l.ffn_up_exps);
    d.ffn_down_exps = Copy(l.ffn_down_exps);
  }
};

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg) {
  // The CPU reference also reads Q6_K, but the routed kernels do not. Reject
  // it for experts before allocating device weights; dense projections and
  // the embedding/output head are re-encoded as Q8_0 instead.
  const auto supported = [&](const TensorRef& t) {
    if (t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr && !layer_supported(mtp->block))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  std::vector<Conversion> conversions;
  Uploader up{*stager,           conversions,     m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg};
  const auto head = [&](const TensorRef& t) {
    return NeedsQ8Conversion(t.type) ? up.CopyConvertedAsQ8_0(t) : up.Copy(t);
  };
  // Tuning::hot_first_upload: allocation order decides placement. Once most
  // of the GPU memory is taken, later allocations get slower memory (up to
  // ~20% lower GEMV bandwidth at the tail on gfx1151 under Windows).
  // Everything read in full on every token goes first: the output head,
  // then every dense projection (target and MTP). The routed experts follow
  // (each byte is read by a few percent of tokens), and the token embedding,
  // a row gather, comes last.
  const bool hot_first = platform::PlatformTuning().hot_first_upload;
  const bool tied = w.output.data == w.token_embd.data;
  if (tied || !hot_first) {
    m->token_embd_ = head(w.token_embd);
  }
  m->output_ = tied ? m->token_embd_ : head(w.output);
  m->hc_head_ = up.Mixer(w.hc_head);
  up.defer_experts = hot_first;
  m->layers_.reserve(w.layers.size());
  for (const auto& l : w.layers) {
    m->layers_.push_back(up.Layer(l));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp != nullptr) {
    // The sidecar has its own shard index; reuse the target's readers and
    // staging pool.
    up.shard_base = shard_count;
    m->mtp_ = up.Layer(mtp->block);
    m->has_mtp_ = true;
    up.shard_base = 0;
  }
  if (hot_first) {
    for (std::size_t i = 0; i < w.layers.size() && up.ok; ++i) {
      up.Experts(w.layers[i], m->layers_[i]);
    }
    if (mtp != nullptr && up.ok) {
      up.shard_base = shard_count;
      up.Experts(mtp->block, m->mtp_);
      up.shard_base = 0;
    }
    if (!tied) {
      m->token_embd_ = head(w.token_embd);
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  for (const auto& c : conversions) {
    NarrowActivations(static_cast<const float*>(c.source), c.destination, false,
                      c.count, nullptr);
  }
  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess) {
    if (error_msg != nullptr) {
      *error_msg =
          "weight conversion failed: " + std::string(hipGetErrorString(status));
    }
    return nullptr;
  }
  for (const auto& c : conversions) {
    std::erase(m->allocations_, c.source);
    (void)hipFree(c.source);
    m->bytes_ -= c.count * sizeof(float) + kTailMargin;
  }
  return m;
}

}  // namespace gufo::models::qwen38_flash_next::rocm
