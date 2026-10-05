#include "src/cli/serve/openai_chat.hpp"

#include <unicode/regex.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/cli/serve/sampling_request.hpp"
#include "src/cli/serve/stop_sequences.hpp"
#include "src/core/image.hpp"
#include "src/core/json.hpp"
#include "src/core/utf8.hpp"

namespace gufo::server {
namespace {

struct ParsedChatRequest {
  ChatRequest chat;
  std::string model;
  std::size_t max_tokens{0};
  sampling::SamplingConfig sampling;
  bool stream{false};
  bool include_usage{false};
};

struct ParsedToolCall {
  std::string id;
  std::string name;
  std::vector<tokenization::ChatMessage::ToolArgument> arguments;
};

struct ParsedGeneration {
  std::string text;
  std::string reasoning_content;
  std::vector<ParsedToolCall> tool_calls;
};

constexpr std::array<std::string_view, 7> kToolMarkers{
    "<tool_call>",          "<｜DSML｜tool_calls｜>", "<｜DSML｜tool_calls>",
    "<DSML｜tool_calls｜>", "<DSML｜tool_calls>",     "<tool_calls｜>",
    "<tool_calls>",
};

long long Now() {
  return static_cast<long long>(std::time(nullptr));
}

std::string RandomId(std::string_view prefix) {
  static constexpr std::string_view kCharacters =
      "abcdefghijklmnopqrstuvwxyz0123456789";
  static thread_local std::mt19937 generator(std::random_device{}());
  std::uniform_int_distribution<std::size_t> distribution{
      0, kCharacters.size() - 1};

  std::string result(prefix);
  result.reserve(prefix.size() + 20);
  for (int index = 0; index < 20; ++index) {
    result.push_back(kCharacters[distribution(generator)]);
  }
  return result;
}

HttpResponse Error(int status, const char* reason, std::string message,
                   const char* code) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = std::move(message);
  error["type"] = "invalid_request_error";
  error["code"] = code;
  response["error"] = std::move(error);
  return {.status = status, .reason = reason, .body = response.dump()};
}

const char* StatusReason(int status) noexcept {
  switch (status) {
    case 408:
      return "Request Timeout";
    case 429:
      return "Too Many Requests";
    case 503:
      return "Service Unavailable";
    case 502:
      return "Bad Gateway";
    default:
      return "Internal Server Error";
  }
}

HttpResponse GenerationError(const TextGenerationError& exception) {
  json::Value response = json::Value::object();
  json::Value error = json::Value::object();
  error["message"] = exception.what();
  error["type"] = "server_error";
  error["code"] = exception.stable_code();
  response["error"] = std::move(error);
  HttpResponse output{
      .status = exception.http_status(),
      .reason = StatusReason(exception.http_status()),
      .body = response.dump(),
      .headers = {},
      .streaming_body = {},
  };
  if (exception.retryable()) {
    output.headers.emplace_back("Retry-After", "1");
  }
  return output;
}

bool IsKnownRole(std::string_view role) {
  return role == "system" || role == "developer" || role == "user" ||
         role == "assistant" || role == "tool";
}

tokenization::ChatRole ParseRole(std::string_view role) {
  if (role == "system") {
    return tokenization::ChatRole::kSystem;
  }
  if (role == "developer") {
    return tokenization::ChatRole::kDeveloper;
  }
  if (role == "assistant") {
    return tokenization::ChatRole::kAssistant;
  }
  if (role == "tool") {
    return tokenization::ChatRole::kTool;
  }
  return tokenization::ChatRole::kUser;
}

bool ParseContent(const json::Value* content,
                  tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error,
                  bool responses = false) {
  auto* output = &message->content;
  if (content == nullptr || content->is_null()) {
    return true;
  }
  if (content->is_string()) {
    *output = content->get_str();
    return true;
  }
  if (!content->is_array()) {
    *error = "message content must be a string, null, or content-part array";
    return false;
  }

  for (const auto& part : content->items()) {
    if (!part.is_object()) {
      *error = "message content parts must be objects";
      return false;
    }
    const std::string type = part.member_str("type", responses ? "" : "text");
    if (type == (responses ? "input_image" : "image_url")) {
      const auto* image = responses ? &part : part.find("image_url");
      const auto* url = image != nullptr && image->is_object()
                            ? image->find(responses ? "image_url" : "url")
                            : nullptr;
      if (responses) {
        if (const auto* file = part.find("file_id");
            file != nullptr && !file->is_null()) {
          *error = "input_image.file_id is not supported; use image_url";
          return false;
        }
      }
      const bool image_role =
          message->role == tokenization::ChatRole::kUser ||
          (responses && message->role == tokenization::ChatRole::kTool);
      if (!image_role || url == nullptr || !url->is_string() ||
          message->images.size() >= 16) {
        *error = responses ? "input_image requires a user message or function "
                             "result and a string image_url (at most 16 images)"
                           : "image_url requires a user message and a string "
                             "URL (at most 16 "
                             "images)";
        return false;
      }
      // Resolution is model-owned; accept only the automatic policy rather
      // than silently ignoring a requested low/high preprocessing policy.
      const auto* detail = image->find("detail");
      if (detail != nullptr &&
          (!detail->is_string() || detail->get_str() != "auto")) {
        *error = "image detail supports only 'auto'";
        return false;
      }
      try {
        message->images.push_back(
            {output->size(), std::make_shared<const std::vector<std::uint8_t>>(
                                 core::ReadImageUrl(url->get_str(), budget))});
      } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
      }
      continue;
    }
    if (type != "text" && type != "input_text" &&
        !(responses && type == "output_text")) {
      *error = responses ? "content parts must use input_text, output_text, "
                           "text or input_image"
                         : "message content parts must use text or image_url";
      return false;
    }
    const json::Value* text = part.find("text");
    if (text == nullptr || !text->is_string()) {
      *error = "text message content parts require a string 'text'";
      return false;
    }
    output->append(text->get_str());
  }
  return true;
}

// Declared tool names reach the Qwen and DeepSeek renderers unescaped, inside
// "<function=NAME>" and "name=\"NAME\"", so the characters that frame a call
// are excluded. The dots, colons and slashes that agent harnesses give bridged
// tool names are data and are kept. Non-ASCII bytes are excluded as well: a
// name is placed in a prompt the model reads and in operator logs, where
// confusable and invisible characters buy a client nothing.
// Historical names describe past output and are preserved by the renderers;
// they do not declare tools the model is allowed to call now.
constexpr std::string_view kToolNameRule =
    "function names require 1-64 printable ASCII characters other than "
    "spaces, '<', '>', '\"' and '\\'";

bool RenderableToolName(std::string_view name) {
  return !name.empty() && name.size() <= 64 &&
         std::ranges::none_of(name, [](unsigned char c) {
           return c <= 0x20 || c >= 0x7F || c == '<' || c == '>' || c == '"' ||
                  c == '\\';
         });
}

bool ParseArguments(std::string_view arguments,
                    std::vector<tokenization::ChatMessage::ToolArgument>* out,
                    std::string* error) {
  json::Value parsed;
  try {
    parsed = json::parse(arguments);
  } catch (const std::exception& exception) {
    *error =
        std::string("tool arguments are not valid JSON: ") + exception.what();
    return false;
  }
  if (!parsed.is_object()) {
    *error = "tool arguments must encode a JSON object";
    return false;
  }
  for (const auto& [name, value] : parsed.members()) {
    out->push_back({
        .name = name,
        .value = value.is_string() ? value.get_str() : value.dump(),
        .is_string = value.is_string(),
    });
  }
  return true;
}

bool ParseHistoricalFunction(const json::Value& function,
                             tokenization::ChatMessage::ToolCall* call,
                             std::string* error) {
  const auto* name = function.find("name");
  const auto* arguments = function.find("arguments");
  if (!name || !name->is_string() || name->str().empty() || !arguments ||
      !arguments->is_string()) {
    *error =
        "historical function calls require a non-empty string name and "
        "string arguments";
    return false;
  }
  // NUL cannot pass through the DeepSeek tokenizer's C-string interface.
  if (name->str().find('\0') != std::string::npos) {
    *error = "historical function names cannot contain NUL";
    return false;
  }
  call->name = name->str();
  return ParseArguments(arguments->str(), &call->arguments, error);
}

bool ParseMessage(const json::Value& value, tokenization::ChatMessage* message,
                  core::ImageReadBudget& budget, std::string* error) {
  if (!value.is_object()) {
    *error = "each message must be an object";
    return false;
  }
  const std::string role = value.member_str("role");
  if (!IsKnownRole(role)) {
    *error = "message role must be system, developer, user, assistant, or tool";
    return false;
  }
  message->role = ParseRole(role);
  message->name = value.member_str("name");
  message->tool_call_id = value.member_str("tool_call_id");
  if (!ParseContent(value.find("content"), message, budget, error)) {
    return false;
  }
  if (const json::Value* reasoning = value.find("reasoning_content");
      reasoning != nullptr && !reasoning->is_null()) {
    if (message->role != tokenization::ChatRole::kAssistant ||
        !reasoning->is_string()) {
      *error =
          "'reasoning_content' is only valid as a string on assistant "
          "messages";
      return false;
    }
    message->thought = reasoning->get_str();
  }

  // SDK message objects serialize an absent field as null.
  const json::Value* tool_calls = value.find("tool_calls");
  if (tool_calls == nullptr || tool_calls->is_null()) {
    return true;
  }
  if (message->role != tokenization::ChatRole::kAssistant ||
      !tool_calls->is_array()) {
    *error = "'tool_calls' is only valid as an array on assistant messages";
    return false;
  }
  for (const auto& item : tool_calls->items()) {
    if (!item.is_object() ||
        item.member_str("type", "function") != "function") {
      *error = "only function tool calls are supported";
      return false;
    }
    const json::Value* function = item.find("function");
    if (function == nullptr || !function->is_object()) {
      *error = "assistant tool calls require a function object";
      return false;
    }
    tokenization::ChatMessage::ToolCall call;
    call.id = item.member_str("id");
    if (!ParseHistoricalFunction(*function, &call, error)) {
      return false;
    }
    message->tool_calls.push_back(std::move(call));
  }
  return true;
}

bool ParseTools(const json::Value* tools,
                std::vector<tokenization::ChatTool>* output,
                std::string* error) {
  if (tools == nullptr || tools->is_null()) {
    return true;
  }
  if (!tools->is_array()) {
    *error = "'tools' must be an array";
    return false;
  }
  for (const auto& item : tools->items()) {
    if (!item.is_object()) {
      *error = "'tools' entries must be objects";
      return false;
    }
    if (item.member_str("type") != "function") {
      *error = "only function tools are supported";
      return false;
    }

    const json::Value* function = item.find("function");
    if (function != nullptr && !function->is_object()) {
      *error = "'function' must be an object";
      return false;
    }
    const json::Value* src = function != nullptr ? function : &item;
    // OpenAI uses "parameters"; some agent clients send parametersJsonSchema.
    const json::Value* params_src = src->find("parameters");
    if (params_src == nullptr || params_src->is_null()) {
      params_src = src->find("parametersJsonSchema");
    }

    tokenization::ChatTool tool;
    tool.name = src->member_str("name");
    tool.description = src->member_str("description");
    if (!RenderableToolName(tool.name)) {
      *error = std::string(kToolNameRule);
      return false;
    }
    if (params_src != nullptr && !params_src->is_null() &&
        !params_src->is_object()) {
      *error = "function tools require an object parameters schema";
      return false;
    }
    // Preserve nested definitions and their field order. For flat tools, move
    // the complete function body (including strict) under "function".
    json::Value function_obj = json::Value::object();
    for (const auto& [key, value] : src->members()) {
      if (key == "parametersJsonSchema" ||
          (function == nullptr && key == "type")) {
        continue;
      }
      function_obj.append_member(key, value);
    }
    function_obj["parameters"] =
        params_src != nullptr && params_src->is_object()
            ? *params_src
            : json::Value::object();
    tool.parameters_json = function_obj.find("parameters")->dump();
    json::Value definition = function != nullptr ? item : json::Value::object();
    definition["type"] = "function";
    definition["function"] = std::move(function_obj);
    tool.definition_json = definition.dump();
    output->push_back(std::move(tool));
  }
  return true;
}

bool ParseToolChoice(const json::Value* value, ChatRequest* request,
                     std::string* error) {
  if (value == nullptr || value->is_null()) {
    return true;
  }
  if (value->is_string()) {
    const std::string choice = value->get_str();
    if (choice == "auto") {
      request->tool_choice = ChatRequest::ToolChoice::kAuto;
      return true;
    }
    if (choice == "none") {
      request->tool_choice = ChatRequest::ToolChoice::kNone;
      return true;
    }
    if (choice == "required") {
      request->tool_choice = ChatRequest::ToolChoice::kRequired;
      return true;
    }
  }
  *error = "'tool_choice' must be auto, none, or required";
  return false;
}

std::optional<ReasoningEffort> ParseReasoningEffortName(
    std::string_view value) {
  if (value == "minimal") {
    return ReasoningEffort::kMinimal;
  }
  if (value == "low") {
    return ReasoningEffort::kLow;
  }
  if (value == "medium") {
    return ReasoningEffort::kMedium;
  }
  if (value == "high") {
    return ReasoningEffort::kHigh;
  }
  if (value == "xhigh") {
    return ReasoningEffort::kXHigh;
  }
  if (value == "max") {
    return ReasoningEffort::kMax;
  }
  return std::nullopt;
}

bool AssignReasoningEnabled(ReasoningOptions* options, bool enabled,
                            std::string* error) {
  if (options->enabled.has_value() && *options->enabled != enabled) {
    *error = "reasoning controls disagree about whether thinking is enabled";
    return false;
  }
  options->enabled = enabled;
  return true;
}

bool AssignReasoningEffort(ReasoningOptions* options, std::string_view value,
                           std::string* error, bool enable_thinking = true) {
  if (value == "off" || value == "none") {
    return AssignReasoningEnabled(options, false, error);
  }
  const auto effort = ParseReasoningEffortName(value);
  if (!effort.has_value()) {
    *error =
        "reasoning_effort must be off, minimal, low, medium, high, xhigh, or "
        "max";
    return false;
  }
  if (options->effort.has_value() && options->effort != effort) {
    *error = "top-level and chat_template_kwargs reasoning_effort disagree";
    return false;
  }
  options->effort = effort;
  return !enable_thinking || AssignReasoningEnabled(options, true, error);
}

bool ParseReasoningOptions(const json::Value& body, ReasoningOptions* options,
                           std::string* error) {
  if (const json::Value* thinking = body.find("thinking")) {
    if (!thinking->is_object()) {
      *error = "'thinking' must be an object";
      return false;
    }
    const json::Value* type = thinking->find("type");
    if (type == nullptr || !type->is_string()) {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    const std::string value = type->get_str();
    if (value != "enabled" && value != "disabled") {
      *error = "'thinking.type' must be enabled or disabled";
      return false;
    }
    if (!AssignReasoningEnabled(options, value == "enabled", error)) {
      return false;
    }
  }

  if (const json::Value* effort = body.find("reasoning_effort");
      effort != nullptr && !effort->is_null()) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error)) {
      if (error->empty()) {
        *error = "'reasoning_effort' must be a string";
      }
      return false;
    }
  }

  const json::Value* kwargs = body.find("chat_template_kwargs");
  if (kwargs == nullptr) {
    return true;
  }
  if (!kwargs->is_object()) {
    *error = "'chat_template_kwargs' must be an object";
    return false;
  }
  if (const json::Value* enabled = kwargs->find("enable_thinking")) {
    if (!enabled->is_bool() ||
        !AssignReasoningEnabled(options, enabled->as_bool(), error)) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.enable_thinking' must be a boolean";
      }
      return false;
    }
  }
  if (const json::Value* mode = kwargs->find("thinking_mode")) {
    if (!mode->is_string()) {
      *error = "'chat_template_kwargs.thinking_mode' must be a string";
      return false;
    }
    const std::string value = mode->get_str();
    if (value != "auto") {
      const bool enabled = value == "thinking" || value == "on";
      if ((!enabled && value != "chat" && value != "off" && value != "none") ||
          !AssignReasoningEnabled(options, enabled, error)) {
        if (error->empty()) {
          *error =
              "'chat_template_kwargs.thinking_mode' must be auto, thinking, "
              "or chat";
        }
        return false;
      }
    }
  }
  if (const json::Value* effort = kwargs->find("reasoning_effort")) {
    if (!effort->is_string() ||
        !AssignReasoningEffort(options, effort->get_str(), error,
                               options->enabled.value_or(true))) {
      if (error->empty()) {
        *error = "'chat_template_kwargs.reasoning_effort' must be a string";
      }
      return false;
    }
  }
  if (const json::Value* preserve = kwargs->find("preserve_thinking")) {
    if (!preserve->is_bool()) {
      *error = "'chat_template_kwargs.preserve_thinking' must be a boolean";
      return false;
    }
    options->preserve_thinking = preserve->as_bool();
  }
  return true;
}

std::optional<HttpResponse> ParseRequest(const HttpRequest& request,
                                         TextGenerationBackend& backend,
                                         ParsedChatRequest* output) {
  json::Value body;
  try {
    body = json::parse(request.body);
  } catch (const std::exception& exception) {
    return Error(400, "Bad Request", exception.what(), "parse_error");
  }
  if (!body.is_object()) {
    return Error(400, "Bad Request", "request body must be a JSON object",
                 "invalid_body");
  }

  output->model = body.member_str("model");
  if (output->model.empty()) {
    return Error(400, "Bad Request", "'model' is required", "missing_model");
  }
  if (output->model != backend.model_id()) {
    return Error(404, "Not Found",
                 "model '" + output->model + "' is not served by this process",
                 "model_not_found");
  }

  output->chat.client_id = request.client_id;
  if (const auto* cache_prompt = body.find("cache_prompt")) {
    if (!cache_prompt->is_bool()) {
      return Error(400, "Bad Request", "'cache_prompt' must be a boolean",
                   "invalid_cache_prompt");
    }
    output->chat.cache_prompt = cache_prompt->as_bool();
  }
  if (const auto error =
          ParseStopSequences(body.find("stop"), StopSequenceFormat::kOpenAi,
                             &output->chat.stop_sequences)) {
    return Error(400, "Bad Request", *error, "invalid_stop");
  }

  const json::Value* messages = body.find("messages");
  if (messages == nullptr || !messages->is_array() || messages->empty()) {
    return Error(400, "Bad Request", "'messages' must be a non-empty array",
                 "missing_messages");
  }
  core::ImageReadBudget image_budget;
  for (const auto& item : messages->items()) {
    tokenization::ChatMessage message;
    std::string parse_error;
    if (!ParseMessage(item, &message, image_budget, &parse_error)) {
      return Error(400, "Bad Request", std::move(parse_error),
                   "invalid_messages");
    }
    output->chat.messages.push_back(std::move(message));
  }

  std::string parse_error;
  if (!ParseTools(body.find("tools"), &output->chat.tools, &parse_error) ||
      !ParseToolChoice(body.find("tool_choice"), &output->chat, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error), "invalid_tools");
  }
  if (output->chat.tool_choice == ChatRequest::ToolChoice::kRequired &&
      output->chat.tools.empty()) {
    return Error(400, "Bad Request",
                 "'tool_choice' cannot be required without tools",
                 "invalid_tool_choice");
  }

  if (!ParseReasoningOptions(body, &output->chat.reasoning, &parse_error)) {
    return Error(400, "Bad Request", std::move(parse_error),
                 "invalid_reasoning");
  }
  if (const auto* kwargs = body.find("chat_template_kwargs")) {
    if (const auto* vision_id = kwargs->find("add_vision_id")) {
      if (!vision_id->is_bool())
        return Error(400, "Bad Request",
                     "'chat_template_kwargs.add_vision_id' must be a boolean",
                     "invalid_template_options");
      output->chat.add_vision_id = vision_id->as_bool();
    }
  }
  const ReasoningOptions defaults = backend.reasoning_defaults();
  if (!output->chat.reasoning.enabled.has_value()) {
    output->chat.reasoning.enabled = defaults.enabled;
  }
  if (!output->chat.reasoning.effort.has_value()) {
    output->chat.reasoning.effort = defaults.effort;
  }
  if (!output->chat.reasoning.preserve_thinking.has_value()) {
    output->chat.reasoning.preserve_thinking = defaults.preserve_thinking;
  }

  if (const json::Value* stream = body.find("stream");
      stream != nullptr && !stream->is_null()) {
    if (!stream->is_bool()) {
      return Error(400, "Bad Request", "'stream' must be a boolean",
                   "invalid_stream");
    }
    output->stream = stream->as_bool();
  }
  if (const json::Value* options = body.find("stream_options");
      options != nullptr && !options->is_null()) {
    if (!options->is_object()) {
      return Error(400, "Bad Request", "'stream_options' must be an object",
                   "invalid_stream_options");
    }
    if (const json::Value* include_usage = options->find("include_usage")) {
      if (!include_usage->is_bool()) {
        return Error(400, "Bad Request",
                     "'stream_options.include_usage' must be a boolean",
                     "invalid_stream_options");
      }
      output->include_usage = include_usage->as_bool();
    }
  }

  const json::Value* max_tokens = body.find("max_completion_tokens");
  if (max_tokens == nullptr || max_tokens->is_null()) {
    max_tokens = body.find("max_tokens");
  }
  if (max_tokens != nullptr && !max_tokens->is_null()) {
    const double value =
        max_tokens->is_number() ? max_tokens->as_double() : 0.0;
    if (!max_tokens->is_number() || !std::isfinite(value) ||
        std::floor(value) != value || value < 1.0 ||
        value >
            static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
      return Error(400, "Bad Request",
                   "'max_tokens' must be a positive integer",
                   "invalid_max_tokens");
    }
    output->max_tokens = max_tokens->as_size();
  }

  sampling::SamplingConfig parsed_sampling;
  if (const auto sampling_error =
          ParseSamplingConfig(body, output->sampling, &parsed_sampling)) {
    return Error(400, "Bad Request", sampling_error->message,
                 sampling_error->code.c_str());
  }
  if (parsed_sampling.temperature > 2.0F) {
    return Error(400, "Bad Request", "'temperature' must be between 0 and 2",
                 "invalid_temperature");
  }
  output->sampling = parsed_sampling;

  if (const json::Value* choices = body.find("n");
      choices != nullptr && !choices->is_null() &&
      (!choices->is_number() || choices->as_double() != 1.0)) {
    return Error(400, "Bad Request", "only n=1 is supported", "unsupported_n");
  }
  for (const std::string_view unsupported :
       {"logprobs", "top_logprobs", "response_format", "modalities", "audio"}) {
    if (const auto* value = body.find(std::string(unsupported));
        value != nullptr && !value->is_null()) {
      if ((unsupported == "logprobs" && value->is_bool() &&
           !value->as_bool()) ||
          (unsupported == "top_logprobs" && value->is_number() &&
           value->as_double() == 0.0) ||
          (unsupported == "response_format" && value->is_object() &&
           value->size() == 1 && value->member_str("type") == "text") ||
          (unsupported == "modalities" && value->is_array() &&
           value->size() == 1 && value->items().front().is_string() &&
           value->items().front().str() == "text"))
        continue;
      return Error(
          400, "Bad Request",
          "request field '" + std::string(unsupported) + "' is not implemented",
          "unsupported_field");
    }
  }
  return std::nullopt;
}

std::string_view Trim(std::string_view value) {
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.front())) != 0) {
    value.remove_prefix(1);
  }
  while (!value.empty() &&
         std::isspace(static_cast<unsigned char>(value.back())) != 0) {
    value.remove_suffix(1);
  }
  return value;
}

// Qwen writes a parameter as "<parameter=name>\nVALUE\n</parameter>": one
// newline on each side is framing, everything else (a file's final newline,
// indentation, blank lines) belongs to the value.
std::string_view StripFramingNewlines(std::string_view value) {
  if (value.starts_with("\r\n"))
    value.remove_prefix(2);
  else if (value.starts_with('\n'))
    value.remove_prefix(1);
  if (value.ends_with("\r\n"))
    value.remove_suffix(2);
  else if (value.ends_with('\n'))
    value.remove_suffix(1);
  return value;
}

// Qwen's XML-like calls look like Python keyword arguments, and the models
// sometimes write Python's True, False and None where the schema asks for a
// JSON boolean or null. Rewrites those words outside JSON string literals.
std::string PythonLiteralsToJson(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  bool in_string = false;
  for (std::size_t i = 0; i < value.size();) {
    const char c = value[i];
    if (in_string) {
      const std::size_t length = c == '\\' && i + 1 < value.size() ? 2 : 1;
      out += value.substr(i, length);
      in_string = c != '"';
      i += length;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) == 0) {
      out += c;
      in_string = c == '"';
      ++i;
      continue;
    }
    std::size_t word_end = i;
    while (word_end < value.size() &&
           (std::isalnum(static_cast<unsigned char>(value[word_end])) != 0 ||
            value[word_end] == '_')) {
      ++word_end;
    }
    const auto word = value.substr(i, word_end - i);
    out += word == "True"    ? std::string_view{"true"}
           : word == "False" ? std::string_view{"false"}
           : word == "None"  ? std::string_view{"null"}
                             : word;
    i = word_end;
  }
  return out;
}

std::optional<json::Value> TryParseJson(std::string_view value) noexcept {
  try {
    return json::parse(value);
  } catch (...) {
    return std::nullopt;
  }
}

std::string ArgumentsJson(
    std::span<const tokenization::ChatMessage::ToolArgument> arguments) {
  json::Value object = json::Value::object();
  for (const auto& argument : arguments) {
    if (argument.is_string) {
      object[argument.name] = argument.value;
      continue;
    }
    try {
      object[argument.name] = json::parse(argument.value);
    } catch (...) {
      object[argument.name] = argument.value;
    }
  }
  return object.dump();
}

// Native Qwen parameters have no type marker. These are decoding hints, not
// a JSON Schema validator: unsupported rules retain non-strict text semantics.
constexpr unsigned kStringType = 1;
constexpr unsigned kIntegerType = 2;
constexpr unsigned kFractionalType = 4;
constexpr unsigned kBooleanType = 8;
constexpr unsigned kNullType = 16;
constexpr unsigned kArrayType = 32;
constexpr unsigned kObjectType = 64;
constexpr unsigned kAllTypes = 127;

unsigned JsonType(std::string_view type) {
  if (type == "string")
    return kStringType;
  if (type == "integer")
    return kIntegerType;
  if (type == "number")
    return kIntegerType | kFractionalType;
  if (type == "boolean")
    return kBooleanType;
  if (type == "null")
    return kNullType;
  if (type == "array")
    return kArrayType;
  if (type == "object")
    return kObjectType;
  return 0;
}

struct TypeHint {
  enum class Resolution { kResolved, kUnsupported, kCyclic, kBounded };
  unsigned types{kAllTypes};
  Resolution resolution{Resolution::kResolved};

  void Intersect(const TypeHint& other) {
    types &= other.types;
    if (other.resolution != Resolution::kResolved)
      resolution = other.resolution;
  }
  bool PreferString() const {
    return resolution != Resolution::kResolved || types == 0 ||
           (types & kStringType) != 0;
  }
  bool Accepts(const json::Value& value) const {
    const auto type = value.is_number()
                          ? (std::floor(value.as_double()) == value.as_double()
                                 ? kIntegerType
                                 : kFractionalType)
                          : JsonType(value.is_null()     ? "null"
                                     : value.is_bool()   ? "boolean"
                                     : value.is_array()  ? "array"
                                     : value.is_object() ? "object"
                                                         : "string");
    return (types & type) != 0;
  }
};

// ICU is not an ECMAScript engine. Admit only a portable subset and normalize
// dot/end-anchor semantics; ICU-only escapes, groups and set operations remain
// guidance. Search is unanchored, as JSON Schema patternProperties requires.
std::optional<bool> MatchesPropertyPattern(std::string_view pattern,
                                           std::string_view name) {
  if (pattern.size() > 512 || name.size() > 2048)
    return std::nullopt;
  std::string portable;
  bool in_class = false;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const char c = pattern[i];
    if (c == '\\') {
      if (++i == pattern.size() ||
          std::string_view{R"(\.^$|?*+()[]{}-/nrtf)"}.find(pattern[i]) ==
              std::string_view::npos)
        return std::nullopt;
      portable += '\\';
      portable += pattern[i];
    } else if (in_class) {
      if (c == '[' || c == '&' || c == '{' || c == '}' ||
          (c == '-' && i + 1 < pattern.size() && pattern[i + 1] == '-'))
        return std::nullopt;
      in_class = c != ']';
      portable += c;
    } else {
      if (c == '(' && i + 1 < pattern.size() && pattern[i + 1] == '?' &&
          (i + 2 == pattern.size() ||
           std::string_view{":=!"}.find(pattern[i + 2]) ==
               std::string_view::npos))
        return std::nullopt;
      if (c == '+' && i > 0 &&
          std::string_view{"*+?}"}.find(pattern[i - 1]) !=
              std::string_view::npos)
        return std::nullopt;
      in_class = c == '[';
      portable += c == '.'   ? R"([^\r\n\u2028\u2029])"
                  : c == '$' ? R"(\z)"
                             : std::string(1, c);
    }
  }
  UErrorCode status = U_ZERO_ERROR;
  const auto input = icu::UnicodeString::fromUTF8(name);
  const std::unique_ptr<icu::RegexPattern> compiled(
      icu::RegexPattern::compile(icu::UnicodeString::fromUTF8(portable),
                                 UREGEX_ERROR_ON_UNKNOWN_ESCAPES, status));
  if (U_FAILURE(status) || !compiled)
    return std::nullopt;
  const std::unique_ptr<icu::RegexMatcher> matcher(
      compiled->matcher(input, status));
  if (U_FAILURE(status) || !matcher)
    return std::nullopt;
  matcher->setTimeLimit(10, status);
  matcher->setStackLimit(64 * 1024, status);
  const bool matches = matcher->find(status);
  return U_FAILURE(status) ? std::nullopt : std::optional(matches);
}

const json::Value* LocalSchemaReference(const json::Value& root,
                                        std::string_view reference) {
  if (reference == "#")
    return &root;
  if (!reference.starts_with("#/") ||
      reference.find('%') != std::string_view::npos)
    return nullptr;
  reference.remove_prefix(2);
  const auto* schema = &root;
  while (true) {
    const auto end = reference.find('/');
    const auto part = reference.substr(0, end);
    std::string key;
    for (std::size_t i = 0; i < part.size(); ++i) {
      if (part[i] != '~') {
        key += part[i];
      } else {
        if (++i == part.size() || (part[i] != '0' && part[i] != '1'))
          return nullptr;
        key += part[i] == '0' ? '~' : '/';
      }
    }
    schema = schema->is_object() ? schema->find(key) : nullptr;
    if (!schema || end == std::string_view::npos)
      return schema;
    reference.remove_prefix(end + 1);
  }
}

TypeHint ResolveDeclaredTypes(const json::Value& root,
                              const json::Value& schema,
                              std::optional<std::string_view> property,
                              std::vector<const json::Value*> path,
                              std::size_t* budget) {
  using Resolution = TypeHint::Resolution;
  if (path.size() >= 32 || *budget == 0)
    return {.resolution = Resolution::kBounded};
  --*budget;
  if (std::ranges::find(path, &schema) != path.end())
    return {.resolution = Resolution::kCyclic};
  path.push_back(&schema);
  if (!schema.is_object())
    return schema.is_bool() && schema.as_bool()
               ? TypeHint{}
               : TypeHint{.resolution = Resolution::kUnsupported};
  for (const auto* rule :
       {"if", "then", "else", "not", "$dynamicRef", "unevaluatedProperties"})
    if (schema.contains(rule))
      return {.resolution = Resolution::kUnsupported};
  TypeHint hint;
  if (const auto* reference = schema.find("$ref")) {
    const auto* target = reference->is_string()
                             ? LocalSchemaReference(root, reference->str())
                             : nullptr;
    if (!target)
      return {.resolution = Resolution::kUnsupported};
    hint.Intersect(ResolveDeclaredTypes(root, *target, property, path, budget));
  }
  if (property) {
    for (const auto* rule : {"anyOf", "oneOf", "allOf"})
      if (schema.contains(rule))
        return {.resolution = Resolution::kUnsupported};
    bool matched = false;
    if (const auto* properties = schema.find("properties")) {
      if (!properties->is_object())
        return {.resolution = Resolution::kUnsupported};
      if (const auto* named = properties->find(std::string(*property))) {
        matched = true;
        hint.Intersect(ResolveDeclaredTypes(root, *named, {}, path, budget));
      }
    }
    if (const auto* patterns = schema.find("patternProperties")) {
      if (!patterns->is_object() || patterns->size() > 64)
        return {.resolution = Resolution::kBounded};
      for (const auto& [pattern, value] : patterns->members()) {
        const auto match = MatchesPropertyPattern(pattern, *property);
        if (!match)
          return {.resolution = Resolution::kUnsupported};
        if (*match) {
          matched = true;
          hint.Intersect(ResolveDeclaredTypes(root, value, {}, path, budget));
        }
      }
    }
    if (!matched) {
      if (const auto* additional = schema.find("additionalProperties"))
        hint.Intersect(
            ResolveDeclaredTypes(root, *additional, {}, path, budget));
    }
    return hint;
  }
  if (const auto* type = schema.find("type")) {
    unsigned mask = 0;
    if (type->is_string()) {
      mask = JsonType(type->str());
    } else if (type->is_array()) {
      for (const auto& item : type->items()) {
        if (!item.is_string() || JsonType(item.str()) == 0)
          return {.resolution = Resolution::kUnsupported};
        mask |= JsonType(item.str());
      }
    }
    if (mask == 0)
      return {.resolution = Resolution::kUnsupported};
    hint.types &= mask;
  }
  for (const auto* rule : {"anyOf", "oneOf", "allOf"}) {
    if (const auto* choices = schema.find(rule)) {
      if (!choices->is_array() || choices->empty())
        return {.resolution = Resolution::kUnsupported};
      const bool intersection = std::string_view(rule) == "allOf";
      TypeHint combined{.types = intersection ? kAllTypes : 0};
      for (const auto& branch : choices->items()) {
        const auto resolved =
            ResolveDeclaredTypes(root, branch, {}, path, budget);
        if (resolved.resolution != Resolution::kResolved)
          return resolved;
        if (intersection)
          combined.types &= resolved.types;
        else
          combined.types |= resolved.types;
      }
      hint.Intersect(combined);
    }
  }
  return hint;
}

// JSON strings may contain protocol delimiters. Locate framing outside them
// without modifying whitespace or repairing invalid control characters.
std::size_t FindUnquoted(std::string_view text, std::string_view marker) {
  bool quoted = false;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (quoted && text[i] == '\\' && i + 1 < text.size()) {
      ++i;
    } else if (text[i] == '"') {
      quoted = !quoted;
    } else if (!quoted && text.substr(i).starts_with(marker)) {
      return i;
    }
  }
  return std::string_view::npos;
}

struct ToolParsing {
  std::size_t malformed_frames{0};
};

ToolParsing ParseQwenCall(std::string_view frame,
                          std::span<const tokenization::ChatTool> tools,
                          std::vector<ParsedToolCall>* calls) {
  constexpr std::string_view start = "<tool_call>";
  constexpr std::string_view end = "</tool_call>";
  // Framing/recovery belongs to the incremental scanner. A rejected payload
  // cannot promote an example inside its argument string to a second call.
  ToolParsing parsing{.malformed_frames = 1};
  auto body = frame.substr(start.size());
  const auto consume = [&](std::string_view tag) {
    body = Trim(body);
    if (!body.starts_with(tag))
      return false;
    body.remove_prefix(tag.size());
    return true;
  };
  body = Trim(body);
  const auto closing = FindUnquoted(body, end);
  ParsedToolCall call;
  bool complete = false;
  if (consume("<function=")) {
    const auto name_end = body.find('>');
    if (name_end == std::string_view::npos)
      return parsing;
    call.name = std::string(Trim(body.substr(0, name_end)));
    body.remove_prefix(name_end + 1);
    std::optional<json::Value> schema;
    for (const auto& tool : tools) {
      if (tool.name == call.name) {
        schema = TryParseJson(tool.parameters_json);
        break;
      }
    }
    if (call.name.empty() || (!tools.empty() && !schema))
      return parsing;
    bool valid = true;
    while (consume("<parameter=")) {
      const auto name_end = body.find('>');
      if (name_end == std::string_view::npos) {
        valid = false;
        break;
      }
      const std::string name(Trim(body.substr(0, name_end)));
      body.remove_prefix(name_end + 1);
      std::size_t budget = 128;
      const auto hint =
          schema ? ResolveDeclaredTypes(*schema, *schema, name, {}, &budget)
                 : TypeHint{};
      const bool is_string = hint.PreferString();
      // Outer closing tags inside a parameter are data, not structure.
      const auto close = is_string ? body.find("</parameter>")
                                   : FindUnquoted(body, "</parameter>");
      const auto nested =
          is_string ? body.find(start) : FindUnquoted(body, start);
      if (close == std::string_view::npos || nested < close || name.empty()) {
        valid = false;
        break;
      }
      const auto value = StripFramingNewlines(body.substr(0, close));
      // Prefer text if the schema permits it; parsing ambiguous scalars
      // as JSON would silently change a caller's declared string type.
      std::string raw(is_string ? value : Trim(value));
      if (!is_string) {
        auto parsed = TryParseJson(raw);
        if (!parsed || !hint.Accepts(*parsed)) {
          raw = PythonLiteralsToJson(raw);
          parsed = TryParseJson(raw);
        }
        if (!parsed || !hint.Accepts(*parsed)) {
          valid = false;
          break;
        }
      }
      // Models sometimes repeat a parameter. An identical copy is harmless;
      // conflicting copies leave no safe choice.
      const auto previous = std::ranges::find(
          call.arguments, name, [](const auto& arg) { return arg.name; });
      if (previous == call.arguments.end()) {
        call.arguments.push_back(
            {.name = name, .value = std::move(raw), .is_string = is_string});
      } else if (previous->value != raw || previous->is_string != is_string) {
        valid = false;
        break;
      }
      body.remove_prefix(close + std::string_view{"</parameter>"}.size());
    }
    complete = valid && consume("</function>") && consume(end);
  } else {
    // The scanner found the unquoted outer boundary. Decode JSON once;
    // quoted closing markers belong to its argument strings.
    if (closing != std::string_view::npos) {
      const auto parsed = TryParseJson(Trim(body.substr(0, closing)));
      if (parsed && parsed->is_object()) {
        call.name = parsed->member_str("name");
        const auto* arguments = parsed->find("arguments");
        if (!call.name.empty() && arguments && arguments->is_object()) {
          for (const auto& [name, value] : arguments->members())
            call.arguments.push_back(
                {.name = name,
                 .value = value.is_string() ? value.get_str() : value.dump(),
                 .is_string = value.is_string()});
          complete = true;
        }
      }
    }
  }
  if (complete && !tools.empty() &&
      std::ranges::none_of(
          tools, [&](const auto& tool) { return tool.name == call.name; }))
    complete = false;
  if (complete) {
    parsing.malformed_frames = 0;
    call.id = RandomId("call_");
    calls->push_back(std::move(call));
  }
  return parsing;
}

std::optional<std::string> Attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const std::size_t start = tag.find(prefix);
  if (start == std::string_view::npos) {
    return std::nullopt;
  }
  const std::size_t value_start = start + prefix.size();
  const std::size_t end = tag.find('"', value_start);
  if (end == std::string_view::npos) {
    return std::nullopt;
  }
  return std::string(tag.substr(value_start, end - value_start));
}

std::size_t DsmlInvokeEnd(std::string_view text, std::size_t cursor,
                          std::string_view invoke_end,
                          std::string_view parameter_start,
                          std::string_view parameter_end) {
  while (cursor < text.size()) {
    const auto end = text.find(invoke_end, cursor);
    const auto parameter = text.find(parameter_start, cursor);
    if (end < parameter || parameter == std::string_view::npos)
      return end;
    const auto tag_end = text.find('>', parameter);
    if (tag_end == std::string_view::npos)
      return tag_end;
    const auto tag = text.substr(parameter, tag_end - parameter + 1);
    const auto value = text.substr(tag_end + 1);
    const auto close = Attribute(tag, "string").value_or("true") != "false"
                           ? value.find(parameter_end)
                           : FindUnquoted(value, parameter_end);
    if (close == std::string_view::npos)
      return close;
    cursor = tag_end + 1 + close + parameter_end.size();
  }
  return std::string_view::npos;
}

ToolParsing ParseDsmlCalls(std::string_view text,
                           std::vector<ParsedToolCall>* calls, bool complete) {
  constexpr std::array<std::string_view, 4> kInvokeStarts{
      "<｜DSML｜invoke",
      "<DSML｜invoke",
      "<｜DS｜invoke",
      "<DS｜invoke",
  };
  constexpr std::array<std::string_view, 4> kInvokeEnds{
      "</｜DSML｜invoke>",
      "</DSML｜invoke>",
      "</｜DS｜invoke>",
      "</DS｜invoke>",
  };
  constexpr std::array<std::string_view, 4> kParameterStarts{
      "<｜DSML｜parameter",
      "<DSML｜parameter",
      "<｜DS｜parameter",
      "<DS｜parameter",
  };
  constexpr std::array<std::string_view, 4> kParameterEnds{
      "</｜DSML｜parameter>",
      "</DSML｜parameter>",
      "</｜DS｜parameter>",
      "</DS｜parameter>",
  };

  ToolParsing parsing;
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    std::size_t invoke_start = std::string_view::npos;
    std::size_t syntax = 0;
    for (std::size_t index = 0; index < kInvokeStarts.size(); ++index) {
      const std::size_t position = text.find(kInvokeStarts[index], cursor);
      if (position < invoke_start) {
        invoke_start = position;
        syntax = index;
      }
    }
    if (invoke_start == std::string_view::npos) {
      parsing.malformed_frames +=
          complete && !Trim(text.substr(cursor)).empty();
      break;
    }
    parsing.malformed_frames +=
        complete && !Trim(text.substr(cursor, invoke_start - cursor)).empty();
    const std::size_t tag_end = text.find('>', invoke_start);
    const auto invoke_end =
        tag_end == std::string_view::npos
            ? tag_end
            : DsmlInvokeEnd(text, tag_end + 1, kInvokeEnds[syntax],
                            kParameterStarts[syntax], kParameterEnds[syntax]);
    if (tag_end == std::string_view::npos ||
        invoke_end == std::string_view::npos) {
      parsing.malformed_frames += complete;
      break;
    }
    ++parsing.malformed_frames;
    ParsedToolCall call;
    call.id = RandomId("call_");
    const auto name = Attribute(
        text.substr(invoke_start, tag_end - invoke_start + 1), "name");
    call.name = name.value_or("");

    bool valid = true;
    std::size_t parameter_cursor = tag_end + 1;
    while (parameter_cursor < invoke_end) {
      const std::size_t parameter_start =
          text.find(kParameterStarts[syntax], parameter_cursor);
      if (parameter_start == std::string_view::npos ||
          parameter_start >= invoke_end) {
        valid =
            Trim(text.substr(parameter_cursor, invoke_end - parameter_cursor))
                .empty();
        break;
      }
      if (!Trim(text.substr(parameter_cursor,
                            parameter_start - parameter_cursor))
               .empty()) {
        valid = false;
        break;
      }
      const std::size_t parameter_tag_end = text.find('>', parameter_start);
      if (parameter_tag_end == std::string_view::npos) {
        valid = false;
        break;
      }
      const std::string_view tag =
          text.substr(parameter_start, parameter_tag_end - parameter_start + 1);
      const auto parameter_name = Attribute(tag, "name");
      const auto string_value = Attribute(tag, "string");
      const bool is_string = string_value.value_or("true") != "false";
      const auto value_start = parameter_tag_end + 1;
      const auto close =
          is_string
              ? text.substr(value_start).find(kParameterEnds[syntax])
              : FindUnquoted(text.substr(value_start), kParameterEnds[syntax]);
      const auto parameter_end =
          close == std::string_view::npos ? close : value_start + close;
      if (parameter_end == std::string_view::npos ||
          parameter_end > invoke_end) {
        valid = false;
        break;
      }
      if (parameter_name.has_value()) {
        tokenization::ChatMessage::ToolArgument argument{
            .name = *parameter_name,
            .value = std::string(Trim(text.substr(
                parameter_tag_end + 1, parameter_end - parameter_tag_end - 1))),
            .is_string = is_string,
        };
        if (argument.name.empty() ||
            (!is_string && !TryParseJson(argument.value))) {
          valid = false;
          break;
        }
        // As for Qwen calls: drop an identical repeat and reject a
        // conflicting one instead of letting the last value win.
        const auto previous =
            std::ranges::find(call.arguments, argument.name,
                              [](const auto& arg) { return arg.name; });
        if (previous == call.arguments.end()) {
          call.arguments.push_back(std::move(argument));
        } else if (previous->value != argument.value ||
                   previous->is_string != argument.is_string) {
          valid = false;
          break;
        }
      } else {
        valid = false;
        break;
      }
      parameter_cursor = parameter_end + kParameterEnds[syntax].size();
    }
    if (valid && !call.name.empty()) {
      --parsing.malformed_frames;
      calls->push_back(std::move(call));
    }
    cursor = invoke_end + kInvokeEnds[syntax].size();
  }
  return parsing;
}

bool RecognizeTools(std::span<const tokenization::ChatTool> tools,
                    ChatRequest::ToolChoice choice) {
  return !tools.empty() && choice != ChatRequest::ToolChoice::kNone;
}

const char* FinishReason(const TextGenerationBackend::Result& result,
                         bool has_tool_calls) {
  if (result.finish_reason ==
      TextGenerationBackend::FinishReason::kStopSequence)
    return "stop";
  if (has_tool_calls) {
    return "tool_calls";
  }
  if (result.finish_reason == TextGenerationBackend::FinishReason::kLength) {
    return "length";
  }
  return "stop";
}

json::Value Usage(const TextGenerationBackend::Result& result) {
  json::Value usage = json::Value::object();
  usage["prompt_tokens"] = result.prompt_tokens;
  usage["completion_tokens"] = result.completion_tokens;
  usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
  json::Value prompt_details = json::Value::object();
  prompt_details["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_details"] = std::move(prompt_details);

  const double prompt_per_second = PrefillTokensPerSecond(result);
  const double predicted_per_second =
      (result.decode_ms > 0.0 && result.completion_tokens > 0)
          ? (static_cast<double>(result.completion_tokens) /
             (result.decode_ms / 1000.0))
          : 0.0;

  usage["cached_tokens"] = result.cached_prompt_tokens;
  usage["prompt_tokens_per_second"] = prompt_per_second;
  usage["completion_tokens_per_second"] = predicted_per_second;
  usage["draft_tokens"] = result.draft_tokens;
  usage["draft_tokens_accepted"] = result.draft_accepted_tokens;

  json::Value metrics = json::Value::object();
  metrics["cache_hit"] = result.cache_hit;
  if (!result.cache_miss_reason.empty()) {
    metrics["cache_miss_reason"] = result.cache_miss_reason;
    metrics["cache_common_prefix_tokens"] = result.cache_common_prefix_tokens;
    metrics["cache_checkpoint_tokens"] = result.cache_checkpoint_tokens;
  }
  metrics["cache_restore_bytes"] = result.cache_restore_bytes;
  metrics["cache_snapshot_bytes"] = result.cache_snapshot_bytes;
  metrics["cache_disk_queued_bytes"] = result.cache_disk_queued_bytes;
  metrics["cache_shared_bytes"] = result.cache_shared_bytes;
  metrics["cache_restore_ms"] = result.cache_restore_ms;
  metrics["cache_snapshot_ms"] = result.cache_snapshot_ms;
  metrics["cache_disk_enqueue_ms"] = result.cache_disk_enqueue_ms;
  metrics["cache_disk_hit"] = result.cache_disk_hit;
  metrics["cache_shared_prefix_snapshots"] =
      result.cache_shared_prefix_snapshots;
  metrics["cache_shared_prefix_bytes"] = result.cache_shared_prefix_bytes;
  metrics["cache_shared_prefix_ms"] = result.cache_shared_prefix_ms;
  metrics["prefill_tokens"] = result.prefill_tokens;
  metrics["prefill_chunks"] = result.prefill_chunks;
  metrics["active_decode_prefill_chunks"] = result.active_decode_prefill_chunks;
  metrics["max_prefill_chunk_tokens"] = result.max_prefill_chunk_tokens;
  metrics["queue_depth_at_submit"] = result.queue_depth_at_submit;
  metrics["client_queue_depth_at_submit"] = result.client_queue_depth_at_submit;
  metrics["resident_requests_at_admission"] =
      result.resident_requests_at_admission;
  metrics["requested_logical_concurrency"] =
      result.requested_logical_concurrency;
  metrics["physical_execution_width"] = result.physical_execution_width;
  metrics["queue_ms"] = result.queue_ms;
  metrics["prefill_ms"] = result.prefill_ms;
  metrics["decode_ms"] = result.decode_ms;
  metrics["ttft_ms"] = result.ttft_ms;
  metrics["mean_inter_token_ms"] = result.mean_inter_token_ms;
  metrics["max_inter_token_ms"] = result.max_inter_token_ms;
  metrics["execution_plan"] = result.execution_plan;
  usage["gufo"] = std::move(metrics);
  return usage;
}

json::Value ToolCallsJson(std::span<const ParsedToolCall> calls) {
  json::Value output = json::Value::array();
  for (const auto& call : calls) {
    json::Value item = json::Value::object();
    item["id"] = call.id;
    item["type"] = "function";
    json::Value function = json::Value::object();
    function["name"] = call.name;
    function["arguments"] = ArgumentsJson(call.arguments);
    item["function"] = std::move(function);
    output.push_back(std::move(item));
  }
  return output;
}

std::string Sse(const json::Value& value) {
  return "data: " + value.dump() + "\n\n";
}

using StreamClock = std::chrono::steady_clock;

bool KeepAlive(const HttpResponse::BodyWriter& writer,
               StreamClock::time_point last_write) {
  // Send transport keepalives only while generation is producing pieces. SSE
  // comments carry no model output and do not prove a tool call is valid.
  if (StreamClock::now() - last_write >= std::chrono::seconds(10))
    return writer(": keep-alive\n\n");
  return true;
}

json::Value BaseChunk(std::string_view id, long long created,
                      std::string_view model) {
  json::Value chunk = json::Value::object();
  chunk["id"] = std::string(id);
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = std::string(model);
  return chunk;
}

json::Value ChoiceChunk(std::string_view id, long long created,
                        std::string_view model, json::Value delta,
                        const char* finish_reason = nullptr) {
  json::Value chunk = BaseChunk(id, created, model);
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  choice["delta"] = std::move(delta);
  if (finish_reason == nullptr) {
    choice["finish_reason"] = json::Value();
  } else {
    choice["finish_reason"] = finish_reason;
  }
  choices.push_back(std::move(choice));
  chunk["choices"] = std::move(choices);
  return chunk;
}

// One parser owns framing for buffered and streamed output. Only the current
// envelope is buffered; completed calls are decoded once and prose can resume.
class GeneratedTextParser {
public:
  using EmitCallback =
      std::function<bool(std::string_view piece, bool is_reasoning)>;
  using ToolFormat = TextGenerationBackend::ToolFormat;
  using OutputState = TextGenerationBackend::InitialOutputState;

  GeneratedTextParser(OutputState initial, ToolFormat format,
                      std::span<const tokenization::ChatTool> tools,
                      ChatRequest::ToolChoice choice, EmitCallback emit = {})
      : tools_(tools),
        choice_(choice),
        format_(format),
        emit_(std::move(emit)),
        state_(initial == OutputState::kReasoning ? State::kReasoning
               : initial == OutputState::kContent ? State::kContent
                                                  : State::kInitial) {}

  bool Push(std::string_view bytes, bool final = false) {
    pending_.append(decoder_.Push(bytes, final));
    while (connected_ && !pending_.empty()) {
      if (state_ == State::kInitial) {
        const auto view = Trim(pending_);
        if (!final && (view.empty() || kThinkStart.starts_with(view)))
          return true;
        if (view.starts_with(kThinkStart)) {
          pending_.erase(0, pending_.find(kThinkStart) + kThinkStart.size());
          state_ = State::kReasoning;
        } else {
          state_ = State::kContent;
        }
      }
      if (state_ == State::kReasoning) {
        // Some backends return the opener although the prompt opened reasoning.
        if (reasoning_start_) {
          if (!final && kThinkStart.starts_with(pending_))
            return true;
          if (pending_.starts_with(kThinkStart))
            pending_.erase(0, kThinkStart.size());
          reasoning_start_ = false;
        }
        const auto end = pending_.find(kThinkEnd);
        if (end == std::string::npos) {
          const auto ready =
              pending_.size() - (final ? 0 : HeldPrefix(pending_, kThinkEnd));
          Emit(pending_.substr(0, ready), true);
          pending_.erase(0, ready);
          return connected_;
        }
        Emit(pending_.substr(0, end), true);
        pending_.erase(0, end + kThinkEnd.size());
        state_ = State::kContent;
        trim_separator_ = true;
      }
      if (state_ == State::kContent) {
        if (trim_separator_) {
          const auto first = pending_.find_first_not_of("\r\n");
          if (first == std::string::npos) {
            pending_.clear();
            return connected_;
          }
          pending_.erase(0, first);
          trim_separator_ = false;
        }
        if (!RecognizeTools(tools_, choice_)) {
          Emit(pending_, false);
          pending_.clear();
          return connected_;
        }
        std::size_t cursor = 0;
        for (; cursor < pending_.size(); ++cursor) {
          if (pending_[cursor] == '`') {
            const auto end = pending_.find_first_not_of('`', cursor);
            if (end == std::string::npos && !final)
              break;
            const auto count =
                (end == std::string::npos ? pending_.size() : end) - cursor;
            if (code_ticks_ == 0)
              code_ticks_ = count;
            else if (code_ticks_ == count)
              code_ticks_ = 0;
            cursor += count - 1;
          } else if (code_ticks_ == 0 && pending_[cursor] == '<') {
            const auto tail = std::string_view(pending_).substr(cursor);
            bool held = false;
            for (const auto marker : kToolMarkers) {
              if (!Admits(marker))
                continue;
              if (tail.starts_with(marker)) {
                // DeepSeek's formatter owns exactly two newlines before a
                // call block. Hold them until this opener proves to be a call.
                frame_separator_ = marker == "<｜DSML｜tool_calls>" &&
                                   cursor >= 2 &&
                                   pending_.compare(cursor - 2, 2, "\n\n") == 0;
                Emit(pending_.substr(0, cursor - (frame_separator_ ? 2 : 0)),
                     false);
                pending_.erase(0, cursor);
                BeginFrame(marker);
                break;
              }
              held |= !final && marker.starts_with(tail);
            }
            if (state_ == State::kTool || held)
              break;
          }
        }
        if (state_ == State::kContent) {
          // Streaming may split the separator from its opener. Retain only
          // its bounded suffix, including before a partial opener.
          auto ready = cursor;
          if (!final && Admits("<｜DSML｜tool_calls>")) {
            while (ready > 0 && cursor - ready < 2 &&
                   pending_[ready - 1] == '\n')
              --ready;
          }
          Emit(pending_.substr(0, ready), false);
          pending_.erase(0, ready);
          return connected_;
        }
      }
      if (state_ == State::kTool && !ScanFrame())
        return connected_;
    }
    return connected_;
  }

  const ParsedGeneration& Finish(bool enforce_required) {
    Push({}, true);
    if (state_ == State::kTool) {
      // A stop/limit may leave the outer DSML envelope open after complete
      // invokes. JSON-owned suffixes never become independent Qwen calls.
      if (frame_attempted_) {
        if (marker_ != "<tool_call>")
          DecodeFrame(pending_);
      } else if (enforce_required ||
                 !Trim(std::string_view(pending_).substr(marker_.size()))
                      .empty()) {
        EmitFrame(pending_);
      }
      if (!frame_attempted_)
        EmitFrame({});
      frame_separator_ = false;
      pending_.clear();
    }
    if (enforce_required && state_ != State::kReasoning) {
      if (choice_ == ChatRequest::ToolChoice::kRequired &&
          parsed_.tool_calls.empty())
        throw TextGenerationError(
            TextGenerationErrorCode::kToolChoiceUnsatisfied,
            "model did not produce a declared tool call");
      if (malformed_)
        throw TextGenerationError(TextGenerationErrorCode::kMalformedToolCall,
                                  "model produced a malformed tool call");
    }
    return parsed_;
  }

private:
  enum class State { kInitial, kReasoning, kContent, kTool };
  static constexpr std::string_view kThinkStart = "<think>";
  static constexpr std::string_view kThinkEnd = "</think>";

  static std::size_t HeldPrefix(std::string_view text,
                                std::string_view marker) {
    for (auto n = std::min(text.size(), marker.size() - 1); n > 0; --n)
      if (marker.starts_with(text.substr(text.size() - n)))
        return n;
    return 0;
  }

  bool Admits(std::string_view marker) const {
    return format_ == ToolFormat::kUnknown ||
           (format_ == ToolFormat::kQwen && marker == "<tool_call>") ||
           (format_ == ToolFormat::kDeepSeek &&
            marker == "<｜DSML｜tool_calls>");
  }

  void Emit(std::string_view piece, bool reasoning) {
    if (piece.empty() || !connected_)
      return;
    (reasoning ? parsed_.reasoning_content : parsed_.text).append(piece);
    if (emit_)
      connected_ = emit_(piece, reasoning);
  }

  void BeginFrame(std::string_view marker) {
    marker_ = marker;
    frame_cursor_ = marker.size();
    state_ = State::kTool;
    frame_attempted_ = false;
    json_body_ = false;
    quoted_ = false;
    escaped_ = false;
    parameter_end_.clear();
    schema_.reset();
  }

  void EmitFrame(std::string_view piece) {
    if (frame_separator_) {
      Emit("\n\n", false);
      frame_separator_ = false;
    }
    Emit(piece, false);
  }

  void DecodeFrame(std::string_view frame, bool complete = false) {
    const auto first = parsed_.tool_calls.size();
    auto body = frame.substr(marker_.size());
    if (complete)
      body.remove_suffix(marker_.size() + 1);
    const auto result =
        marker_ == "<tool_call>"
            ? ParseQwenCall(frame, tools_, &parsed_.tool_calls)
            : ParseDsmlCalls(body, &parsed_.tool_calls, complete);
    malformed_ |= result.malformed_frames != 0;
    auto begin =
        parsed_.tool_calls.begin() + static_cast<std::ptrdiff_t>(first);
    parsed_.tool_calls.erase(
        std::remove_if(begin, parsed_.tool_calls.end(),
                       [&](const auto& call) {
                         const bool undeclared = std::ranges::none_of(
                             tools_, [&](const auto& tool) {
                               return tool.name == call.name;
                             });
                         malformed_ |= undeclared;
                         return undeclared;
                       }),
        parsed_.tool_calls.end());
  }

  bool ScanFrame() {
    while (frame_cursor_ < pending_.size()) {
      const char ch = pending_[frame_cursor_];
      const bool json =
          json_body_ || (!parameter_end_.empty() && parameter_json_);
      if (json && quoted_) {
        if (escaped_)
          escaped_ = false;
        else if (ch == '\\')
          escaped_ = true;
        else if (ch == '"')
          quoted_ = false;
        ++frame_cursor_;
        continue;
      }
      if (json && ch == '"') {
        quoted_ = true;
        ++frame_cursor_;
        continue;
      }
      if (ch != '<') {
        if (!frame_attempted_ &&
            std::isspace(static_cast<unsigned char>(ch)) == 0) {
          if (marker_ == "<tool_call>" && ch == '{') {
            frame_attempted_ = true;
            json_body_ = true;
          } else {
            // An ordinary mention of the opener is visible prose.
            EmitFrame(pending_.substr(0, frame_cursor_));
            pending_.erase(0, frame_cursor_);
            state_ = State::kContent;
            return true;
          }
        }
        ++frame_cursor_;
        continue;
      }
      const auto tail = std::string_view(pending_).substr(frame_cursor_);
      if (!parameter_end_.empty()) {
        if (tail.starts_with(parameter_end_)) {
          frame_cursor_ += parameter_end_.size();
          parameter_end_.clear();
          continue;
        }
        constexpr std::string_view qwen_start = "<tool_call>";
        const bool qwen = marker_ == qwen_start;
        if (parameter_end_.starts_with(tail) ||
            (qwen && qwen_start.starts_with(tail)))
          return false;
        if (!qwen || !tail.starts_with(qwen_start)) {
          ++frame_cursor_;
          continue;
        }
      }
      frame_attempted_ |= marker_ == "<tool_call>"
                              ? tail.starts_with("<function=")
                              : tail.starts_with("<｜DSML｜invoke") ||
                                    tail.starts_with("<DSML｜invoke") ||
                                    tail.starts_with("<｜DS｜invoke") ||
                                    tail.starts_with("<DS｜invoke");
      const auto end = pending_.find('>', frame_cursor_);
      if (end == std::string::npos)
        return false;
      const auto tag = std::string_view(pending_).substr(
          frame_cursor_, end - frame_cursor_ + 1);
      const auto nested = std::ranges::find(kToolMarkers, tag);
      if (nested != kToolMarkers.end() && Admits(*nested)) {
        // Recover an interrupted native call only outside JSON strings. The
        // old frame has no complete outer boundary, so it is not an EOS error.
        if (!frame_attempted_)
          EmitFrame(pending_.substr(0, frame_cursor_));
        else if (marker_ != "<tool_call>")
          DecodeFrame(std::string_view(pending_).substr(0, frame_cursor_));
        pending_.erase(0, frame_cursor_);
        frame_separator_ = false;
        BeginFrame(*nested);
        continue;
      }
      const std::string closing = "</" + std::string(marker_.substr(1));
      if (tag == closing) {
        if (frame_attempted_) {
          const auto count = parsed_.tool_calls.size();
          DecodeFrame(std::string_view(pending_).substr(0, end + 1), true);
          malformed_ |= count == parsed_.tool_calls.size();
        } else {
          EmitFrame(pending_.substr(0, end + 1));
        }
        frame_separator_ = false;
        pending_.erase(0, end + 1);
        state_ = State::kContent;
        return true;
      }
      if (tag.starts_with("<function=")) {
        frame_attempted_ = true;
        const auto name = Trim(tag.substr(10, tag.size() - 11));
        for (const auto& tool : tools_)
          if (tool.name == name) {
            schema_ = TryParseJson(tool.parameters_json);
            break;
          }
      } else if (tag.starts_with("<parameter=")) {
        const auto name = Trim(tag.substr(11, tag.size() - 12));
        std::size_t budget = 128;
        const auto hint = schema_ ? ResolveDeclaredTypes(*schema_, *schema_,
                                                         name, {}, &budget)
                                  : TypeHint{};
        parameter_json_ = !hint.PreferString();
        parameter_end_ = "</parameter>";
      } else if (tag.find("invoke") != std::string_view::npos ||
                 tag.find("parameter") != std::string_view::npos) {
        // DSML parameter quotes are raw text unless string="false".
        constexpr std::array<std::string_view, 4> stems{"<｜DSML｜", "<DSML｜",
                                                        "<｜DS｜", "<DS｜"};
        for (const auto stem : stems) {
          if (tag.starts_with(std::string(stem) + "invoke"))
            frame_attempted_ = true;
          else if (tag.starts_with(std::string(stem) + "parameter")) {
            parameter_json_ =
                Attribute(tag, "string").value_or("true") == "false";
            parameter_end_ = "</" + std::string(stem.substr(1)) + "parameter>";
          }
        }
      }
      if (!frame_attempted_ && tag != marker_) {
        EmitFrame(pending_.substr(0, end + 1));
        pending_.erase(0, end + 1);
        state_ = State::kContent;
        return true;
      }
      frame_cursor_ = end + 1;
    }
    return false;
  }

  std::span<const tokenization::ChatTool> tools_;
  ChatRequest::ToolChoice choice_;
  ToolFormat format_;
  EmitCallback emit_;
  State state_;
  core::Utf8Decoder decoder_;
  ParsedGeneration parsed_;
  std::string pending_;
  std::string_view marker_;
  std::string parameter_end_;
  std::optional<json::Value> schema_;
  std::size_t frame_cursor_{0};
  std::size_t code_ticks_{0};
  bool frame_attempted_{false};
  bool frame_separator_{false};
  bool json_body_{false};
  bool parameter_json_{false};
  bool quoted_{false};
  bool escaped_{false};
  bool reasoning_start_{true};
  bool trim_separator_{false};
  bool malformed_{false};
  bool connected_{true};
};

// Responses uses semantic SSE events, rather than Chat Completions chunks.
// Build the same output items for streaming and buffered responses.
class ResponsesOutput {
public:
  ResponsesOutput(std::string model, const ChatRequest& chat,
                  bool parallel_tool_calls, HttpResponse::BodyWriter writer)
      : writer_(std::move(writer)) {
    response_ = json::Value::object();
    response_["id"] = RandomId("resp_");
    response_["object"] = "response";
    response_["created_at"] = Now();
    response_["model"] = std::move(model);
    response_["status"] = "in_progress";
    response_["error"] = json::Value();
    response_["incomplete_details"] = json::Value();
    response_["usage"] = json::Value();
    response_["output"] = json::Value::array();
    response_["store"] = false;
    response_["parallel_tool_calls"] =
        !chat.tools.empty() && parallel_tool_calls;
    response_["tool_choice"] =
        chat.tool_choice == ChatRequest::ToolChoice::kRequired ? "required"
        : chat.tools.empty() ||
                chat.tool_choice == ChatRequest::ToolChoice::kNone
            ? "none"
            : "auto";
    response_["tools"] = json::Value::array();
    for (const auto& tool : chat.tools) {
      const auto definition = json::parse(tool.definition_json);
      auto function = *definition.find("function");
      function["type"] = "function";
      response_["tools"].push_back(std::move(function));
    }
  }

  bool Begin() {
    return Lifecycle("response.created") && Lifecycle("response.in_progress");
  }

  bool Append(std::string_view text, bool reasoning) {
    if (text.empty())
      return true;
    if (!active_ || reasoning_ != reasoning) {
      if (!CloseItem("completed"))
        return false;
      reasoning_ = reasoning;
      active_ = true;
      item_ = json::Value::object();
      item_["id"] = RandomId(reasoning ? "rs_" : "msg_");
      item_["type"] = reasoning ? "reasoning" : "message";
      item_["status"] = "in_progress";
      if (reasoning)
        item_["encrypted_content"] = json::Value();
      item_[reasoning ? "summary" : "content"] = json::Value::array();
      if (!reasoning)
        item_["role"] = "assistant";
      auto added = IndexedEvent("response.output_item.added");
      added["item"] = item_;
      if (!Emit(std::move(added)))
        return false;
      text_.clear();
      auto part = PartEvent(reasoning ? "response.reasoning_summary_part.added"
                                      : "response.content_part.added");
      part["part"] = Part();
      if (!Emit(std::move(part)))
        return false;
    }
    text_.append(text);
    auto delta = PartEvent(reasoning ? "response.reasoning_summary_text.delta"
                                     : "response.output_text.delta");
    delta["delta"] = std::string(text);
    if (!reasoning)
      delta["logprobs"] = json::Value::array();
    return Emit(std::move(delta));
  }

  bool FunctionCall(const ParsedToolCall& call) {
    if (!CloseItem("completed"))
      return false;
    auto item = json::Value::object();
    item["id"] = RandomId("fc_");
    item["type"] = "function_call";
    item["status"] = "in_progress";
    item["call_id"] = call.id;
    item["name"] = call.name;
    item["arguments"] = "";
    auto added = IndexedEvent("response.output_item.added");
    added["item"] = item;
    if (!Emit(std::move(added)))
      return false;
    const auto arguments = ArgumentsJson(call.arguments);
    auto delta = IndexedEvent("response.function_call_arguments.delta");
    delta["item_id"] = item.member_str("id");
    delta["delta"] = arguments;
    if (!Emit(std::move(delta)))
      return false;
    auto done = IndexedEvent("response.function_call_arguments.done");
    done["item_id"] = item.member_str("id");
    done["name"] = call.name;
    done["arguments"] = arguments;
    if (!Emit(std::move(done)))
      return false;
    item["arguments"] = arguments;
    item["status"] = "completed";
    auto completed = IndexedEvent("response.output_item.done");
    completed["item"] = item;
    response_["output"].push_back(std::move(item));
    ++output_index_;
    return Emit(std::move(completed));
  }

  json::Value Complete(const TextGenerationBackend::Result& result) {
    const bool limited =
        result.finish_reason == TextGenerationBackend::FinishReason::kLength;
    CloseItem(limited ? "incomplete" : "completed");
    response_["status"] = limited ? "incomplete" : "completed";
    if (limited)
      response_["incomplete_details"]["reason"] = "max_output_tokens";
    auto usage = json::Value::object();
    usage["input_tokens"] = result.prompt_tokens;
    usage["input_tokens_details"]["cached_tokens"] =
        result.cached_prompt_tokens;
    usage["input_tokens_details"]["cache_write_tokens"] = result.prefill_tokens;
    usage["output_tokens"] = result.completion_tokens;
    usage["output_tokens_details"]["reasoning_tokens"] =
        result.reasoning_tokens;
    usage["total_tokens"] = result.prompt_tokens + result.completion_tokens;
    response_["usage"] = std::move(usage);
    response_["timings"] = GenerationTimings(result);
    Lifecycle(limited ? "response.incomplete" : "response.completed");
    return response_;
  }

  bool Fail(std::string_view message) {
    response_["status"] = "failed";
    // Responses defines a closed error-code enum. Keep the specific runtime
    // code in the request log, rather than emitting an invalid wire value.
    response_["error"]["code"] = "server_error";
    response_["error"]["message"] = std::string(message);
    return Lifecycle("response.failed");
  }

private:
  json::Value IndexedEvent(std::string_view type) const {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["response_id"] = response_.member_str("id");
    event["output_index"] = output_index_;
    return event;
  }

  json::Value PartEvent(std::string_view type) const {
    auto event = IndexedEvent(type);
    event["item_id"] = item_.member_str("id");
    event[reasoning_ ? "summary_index" : "content_index"] = 0;
    return event;
  }

  json::Value Part() const {
    auto part = json::Value::object();
    part["type"] = reasoning_ ? "summary_text" : "output_text";
    part["text"] = text_;
    if (!reasoning_) {
      part["annotations"] = json::Value::array();
      part["logprobs"] = json::Value::array();
    }
    return part;
  }

  bool CloseItem(const char* status) {
    if (!active_)
      return connected_;
    auto done = PartEvent(reasoning_ ? "response.reasoning_summary_text.done"
                                     : "response.output_text.done");
    done["text"] = text_;
    if (!reasoning_)
      done["logprobs"] = json::Value::array();
    if (!Emit(std::move(done)))
      return false;
    auto part = Part();
    auto part_done =
        PartEvent(reasoning_ ? "response.reasoning_summary_part.done"
                             : "response.content_part.done");
    part_done["part"] = part;
    if (!Emit(std::move(part_done)))
      return false;
    item_[reasoning_ ? "summary" : "content"].push_back(std::move(part));
    item_["status"] = status;
    auto item_done = IndexedEvent("response.output_item.done");
    item_done["item"] = item_;
    response_["output"].push_back(item_);
    active_ = false;
    ++output_index_;
    return Emit(std::move(item_done));
  }

  bool Lifecycle(std::string_view type) {
    auto event = json::Value::object();
    event["type"] = std::string(type);
    event["response"] = response_;
    return Emit(std::move(event));
  }

  bool Emit(json::Value event) {
    if (!writer_)
      return true;
    if (!connected_)
      return false;
    event["sequence_number"] = sequence_++;
    connected_ =
        writer_("event: " + event.member_str("type") + "\n" + Sse(event));
    return connected_;
  }

  HttpResponse::BodyWriter writer_;
  json::Value response_;
  json::Value item_;
  std::string text_;
  std::size_t sequence_{0};
  std::size_t output_index_{0};
  bool active_{false};
  bool reasoning_{false};
  bool connected_{true};
};

HttpResponse NonStreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    const std::shared_ptr<TextGenerationBackend::GenerationRequest>& generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  const auto result = generation->Wait();
  GeneratedTextParser parser(initial_output_state, generation->tool_format(),
                             request.chat.tools, request.chat.tool_choice);
  parser.Push(result.text);
  const auto& generated = parser.Finish(
      !result.cancelled &&
      result.finish_reason == TextGenerationBackend::FinishReason::kStop);

  json::Value response = json::Value::object();
  response["id"] = RandomId("chatcmpl-");
  response["object"] = "chat.completion";
  response["created"] = Now();
  response["model"] = backend.model_id();
  json::Value choices = json::Value::array();
  json::Value choice = json::Value::object();
  choice["index"] = 0;
  json::Value message = json::Value::object();
  message["role"] = "assistant";
  if (!generated.reasoning_content.empty()) {
    message["reasoning_content"] = generated.reasoning_content;
  }
  if (generated.text.empty() && !generated.tool_calls.empty()) {
    message["content"] = json::Value();
  } else {
    message["content"] = generated.text;
  }
  if (!generated.tool_calls.empty()) {
    message["tool_calls"] = ToolCallsJson(generated.tool_calls);
  }
  choice["message"] = std::move(message);
  choice["finish_reason"] = FinishReason(result, !generated.tool_calls.empty());
  choices.push_back(std::move(choice));
  response["choices"] = std::move(choices);
  response["usage"] = Usage(result);
  response["timings"] = GenerationTimings(result);
  RecordServerMetrics(result);

  std::ostringstream timing;
  timing << std::fixed << std::setprecision(3) << "ttft;dur=" << result.ttft_ms
         << ", inter_token;dur=" << result.mean_inter_token_ms
         << ", max_inter_token;dur=" << result.max_inter_token_ms;

  return {
      .status = 200,
      .reason = "OK",
      .body = response.dump(),
      .headers = {{"Server-Timing", timing.str()}},
      .streaming_body = {},
      .log_details = GenerationLogDetails(result),
  };
}

HttpResponse StreamingResponse(
    const ParsedChatRequest& request, TextGenerationBackend& backend,
    std::shared_ptr<TextGenerationBackend::GenerationRequest> generation,
    TextGenerationBackend::InitialOutputState initial_output_state) {
  const std::string id = RandomId("chatcmpl-");
  const long long created = Now();
  const std::string model = backend.model_id();
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  return {
      .status = 200,
      .reason = "OK",
      .body = {},
      .headers =
          {
              {"Content-Type", "text/event-stream"},
              {"Cache-Control", "no-cache"},
              {"X-Accel-Buffering", "no"},
          },
      .streaming_body =
          [request, generation = std::move(generation), id, created, model,
           initial_output_state,
           stream_log](const HttpResponse::BodyWriter& writer) {
            auto last_write = StreamClock::now();
            const HttpResponse::BodyWriter write = [&](std::string_view piece) {
              last_write = StreamClock::now();
              return writer(piece);
            };
            json::Value role_delta = json::Value::object();
            role_delta["role"] = "assistant";
            if (!write(Sse(
                    ChoiceChunk(id, created, model, std::move(role_delta))))) {
              generation->Cancel();
              return;
            }

            bool connected = true;
            GeneratedTextParser filter(
                initial_output_state, generation->tool_format(),
                request.chat.tools, request.chat.tool_choice,
                [&](std::string_view text, bool is_reasoning) {
                  if (text.empty()) {
                    return true;
                  }
                  json::Value delta = json::Value::object();
                  if (is_reasoning) {
                    delta["reasoning_content"] = std::string(text);
                  } else {
                    delta["content"] = std::string(text);
                  }
                  connected = write(
                      Sse(ChoiceChunk(id, created, model, std::move(delta))));
                  return connected;
                });

            try {
              const auto result = generation->Wait([&](std::string_view piece) {
                connected = connected && filter.Push(piece) &&
                            KeepAlive(write, last_write);
                if (!connected)
                  generation->Cancel();
                return connected;
              });
              stream_log->details = GenerationLogDetails(result);
              RecordServerMetrics(result);
              if (!connected || result.cancelled) {
                return;
              }
              if (!filter.Push({}, true))
                return;

              const auto& generated =
                  filter.Finish(result.finish_reason ==
                                TextGenerationBackend::FinishReason::kStop);
              if (!connected) {
                return;
              }
              for (std::size_t index = 0; index < generated.tool_calls.size();
                   ++index) {
                const auto& call = generated.tool_calls[index];
                json::Value delta = json::Value::object();
                json::Value tool_calls = json::Value::array();
                json::Value item = json::Value::object();
                item["index"] = index;
                item["id"] = call.id;
                item["type"] = "function";
                json::Value function = json::Value::object();
                function["name"] = call.name;
                function["arguments"] = ArgumentsJson(call.arguments);
                item["function"] = std::move(function);
                tool_calls.push_back(std::move(item));
                delta["tool_calls"] = std::move(tool_calls);
                if (!write(Sse(
                        ChoiceChunk(id, created, model, std::move(delta))))) {
                  return;
                }
              }

              json::Value terminal_delta = json::Value::object();
              auto terminal_chunk = ChoiceChunk(
                  id, created, model, std::move(terminal_delta),
                  FinishReason(result, !generated.tool_calls.empty()));
              // llama.cpp reports timings on the terminal choice regardless
              // of the optional OpenAI usage chunk. Proxies need these even
              // when a client does not request stream_options.include_usage.
              terminal_chunk["timings"] = GenerationTimings(result);
              if (!write(Sse(terminal_chunk))) {
                return;
              }
              if (request.include_usage) {
                json::Value usage_chunk = BaseChunk(id, created, model);
                usage_chunk["choices"] = json::Value::array();
                usage_chunk["usage"] = Usage(result);
                usage_chunk["timings"] = GenerationTimings(result);
                if (!write(Sse(usage_chunk))) {
                  return;
                }
              }
              (void)write("data: [DONE]\n\n");
            } catch (const TextGenerationError& exception) {
              stream_log->error_code = exception.stable_code();
              json::Value error = json::Value::object();
              json::Value detail = json::Value::object();
              detail["message"] = exception.what();
              detail["type"] = "server_error";
              detail["code"] = exception.stable_code();
              error["error"] = std::move(detail);
              stream_log->error_event_sent = write(Sse(error));
              (void)write("data: [DONE]\n\n");
            } catch (const std::exception& error) {
              stream_log->error_code = "generation_failed";
              json::Value err = json::Value::object();
              json::Value detail = json::Value::object();
              const char* message = error.what();
              detail["message"] =
                  message && *message ? message : "generation failed";
              detail["type"] = "server_error";
              detail["code"] = "generation_failed";
              err["error"] = std::move(detail);
              stream_log->error_event_sent = write(Sse(err));
              (void)write("data: [DONE]\n\n");
            }
          },
      .stream_log = std::move(stream_log),
  };
}

}  // namespace

bool ParseOpenAiResponseChat(const json::Value& body, ChatRequest* chat,
                             std::string* error) {
  if (const auto* reasoning = body.find("reasoning");
      reasoning != nullptr && !reasoning->is_null()) {
    if (!reasoning->is_object()) {
      *error = "'reasoning' must be an object";
      return false;
    }
    for (const auto& [field, value] : reasoning->members()) {
      if (field == "effort") {
        if (!value.is_null() &&
            (!value.is_string() ||
             !AssignReasoningEffort(&chat->reasoning, value.str(), error))) {
          if (error->empty())
            *error = "'reasoning.effort' must be a string";
          return false;
        }
      } else if (field == "summary") {
        // Gufo exposes local reasoning in summary_text items; it has no
        // separate concise/detailed summarizer.
        if (!value.is_null() && (!value.is_string() || value.str() != "auto")) {
          *error = "'reasoning.summary' supports only auto or null";
          return false;
        }
      } else {
        *error = "unsupported reasoning field: " + field;
        return false;
      }
    }
  }
  if (const auto* include = body.find("include");
      include != nullptr && !include->is_null()) {
    if (!include->is_array() ||
        std::ranges::any_of(include->items(), [](const auto& value) {
          return !value.is_string() ||
                 value.str() != "reasoning.encrypted_content";
        })) {
      *error = "'include' supports only reasoning.encrypted_content";
      return false;
    }
    // This optional-data hint is sent by Responses clients such as Oh My Pi.
    // Local reasoning is already replayable plaintext; encrypted_content is
    // null.
  }
  if (!ParseTools(body.find("tools"), &chat->tools, error) ||
      !ParseToolChoice(body.find("tool_choice"), chat, error))
    return false;
  if (chat->tool_choice == ChatRequest::ToolChoice::kRequired &&
      chat->tools.empty()) {
    *error = "'tool_choice' cannot be required without tools";
    return false;
  }
  if (const auto* parallel = body.find("parallel_tool_calls");
      parallel != nullptr && !parallel->is_bool()) {
    *error = "'parallel_tool_calls' must be a boolean";
    return false;
  }
  if (const auto* instructions = body.find("instructions")) {
    if (!instructions->is_string()) {
      *error = "'instructions' must be a string";
      return false;
    }
    chat->messages.push_back(
        {tokenization::ChatRole::kSystem, instructions->str(), "", ""});
  }
  core::ImageReadBudget image_budget;
  const auto read_content = [&](const json::Value* value,
                                tokenization::ChatMessage* message) {
    return value != nullptr && !value->is_null() &&
           ParseContent(value, message, image_budget, error, true);
  };
  const auto* input = body.find("input");
  if (input != nullptr && input->is_string() && !input->str().empty()) {
    chat->messages.push_back(
        {tokenization::ChatRole::kUser, input->str(), "", ""});
    return true;
  }
  *error =
      "'input' must contain messages, Gufo reasoning items, function "
      "calls or function results with text/image content";
  if (input == nullptr || !input->is_array() || input->empty())
    return false;
  for (const auto& item : input->items()) {
    if (!item.is_object())
      return false;
    const auto type = item.member_str("type", "message");
    tokenization::ChatMessage message;
    if (type == "reasoning") {
      const auto* summary = item.find("summary");
      const auto* encrypted = item.find("encrypted_content");
      if (summary == nullptr || !summary->is_array() ||
          (encrypted != nullptr && !encrypted->is_null()) ||
          item.contains("content"))
        return false;
      message.role = tokenization::ChatRole::kAssistant;
      for (const auto& part : summary->items()) {
        const auto* text = part.find("text");
        if (part.member_str("type") != "summary_text" || text == nullptr ||
            !text->is_string())
          return false;
        message.thought += text->str();
      }
    } else if (type == "function_call") {
      message.role = tokenization::ChatRole::kAssistant;
      tokenization::ChatMessage::ToolCall call;
      call.id = item.member_str("call_id");
      if (call.id.empty() || !ParseHistoricalFunction(item, &call, error))
        return false;
      message.tool_calls.push_back(std::move(call));
    } else if (type == "function_call_output") {
      message.role = tokenization::ChatRole::kTool;
      message.tool_call_id = item.member_str("call_id");
      if (message.tool_call_id.empty() ||
          !read_content(item.find("output"), &message))
        return false;
      // Responses is stateless: resolve the result against a call in input.
      for (const auto& previous : chat->messages) {
        for (const auto& call : previous.tool_calls) {
          if (call.id == message.tool_call_id)
            message.name = call.name;
        }
      }
      if (message.name.empty()) {
        *error =
            "function_call_output requires a matching earlier function_call in "
            "'input'";
        return false;
      }
    } else if (type == "message") {
      const auto role = item.member_str("role");
      if (!IsKnownRole(role) || role == "tool" || item.contains("tool_calls"))
        return false;
      message.role = ParseRole(role);
      if (!read_content(item.find("content"), &message))
        return false;
    } else {
      return false;
    }
    // Reasoning, text and calls from one assistant turn are separate Responses
    // items, but must share one assistant message in the model's chat template.
    if (message.role == tokenization::ChatRole::kAssistant &&
        !chat->messages.empty() &&
        chat->messages.back().role == tokenization::ChatRole::kAssistant) {
      auto& previous = chat->messages.back();
      previous.content += message.content;
      previous.thought += message.thought;
      for (auto& call : message.tool_calls)
        previous.tool_calls.push_back(std::move(call));
    } else {
      chat->messages.push_back(std::move(message));
    }
  }
  error->clear();
  return true;
}

HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream, bool parallel_tool_calls) {
  const auto initial = backend.initial_output_state(chat);
  auto generation = backend.start_chat(chat, max_tokens, sampling,
                                       request.is_cancelled, stream);
  auto stream_log = std::make_shared<HttpResponse::StreamLog>();
  auto timing = std::make_shared<std::string>();
  const auto run = [generation, initial, model = backend.model_id(), stream_log,
                    timing, chat, parallel_tool_calls](
                       const HttpResponse::BodyWriter& writer) {
    auto last_write = StreamClock::now();
    const HttpResponse::BodyWriter write = [&](std::string_view piece) {
      last_write = StreamClock::now();
      return writer(piece);
    };
    ResponsesOutput output(model, chat, parallel_tool_calls,
                           writer ? write : HttpResponse::BodyWriter{});
    if (!output.Begin()) {
      generation->Cancel();
      return json::Value();
    }
    GeneratedTextParser filter(initial, generation->tool_format(), chat.tools,
                               chat.tool_choice,
                               [&](std::string_view piece, bool reasoning) {
                                 if (output.Append(piece, reasoning))
                                   return true;
                                 generation->Cancel();
                                 return false;
                               });
    try {
      const auto result =
          writer ? generation->Wait([&](std::string_view piece) {
            if (filter.Push(piece) && KeepAlive(write, last_write))
              return true;
            generation->Cancel();
            return false;
          })
                 : generation->Wait();
      stream_log->details = GenerationLogDetails(result);
      RecordServerMetrics(result);
      if (!writer) {
        std::ostringstream value;
        value << std::fixed << std::setprecision(3)
              << "ttft;dur=" << result.ttft_ms
              << ", inter_token;dur=" << result.mean_inter_token_ms
              << ", max_inter_token;dur=" << result.max_inter_token_ms;
        *timing = value.str();
      }
      if (result.cancelled)
        return json::Value();
      if (!writer)
        filter.Push(result.text);
      if (!filter.Push({}, true)) {
        generation->Cancel();
        return json::Value();
      }
      const auto& generated = filter.Finish(
          result.finish_reason == TextGenerationBackend::FinishReason::kStop);
      if (!parallel_tool_calls && generated.tool_calls.size() > 1)
        throw TextGenerationError(
            TextGenerationErrorCode::kToolChoiceUnsatisfied,
            "model produced multiple calls with 'parallel_tool_calls' false");
      // Publish tools only after all parsing and policy checks have succeeded.
      for (const auto& call : generated.tool_calls) {
        if (!output.FunctionCall(call)) {
          generation->Cancel();
          return json::Value();
        }
      }
      return output.Complete(result);
    } catch (const std::exception& error) {
      if (!writer)
        throw;
      const auto* generation_error =
          dynamic_cast<const TextGenerationError*>(&error);
      stream_log->error_code = generation_error
                                   ? generation_error->stable_code()
                                   : "generation_failed";
      const char* message = error.what();
      stream_log->error_event_sent =
          output.Fail(message && *message ? message : "generation failed");
      generation->Cancel();
      return json::Value();
    }
  };
  if (stream) {
    return {.status = 200,
            .reason = "OK",
            .body = {},
            .headers = {{"Content-Type", "text/event-stream"},
                        {"Cache-Control", "no-cache"},
                        {"X-Accel-Buffering", "no"}},
            .streaming_body =
                [run](const HttpResponse::BodyWriter& writer) {
                  (void)run(writer);
                },
            .stream_log = std::move(stream_log)};
  }
  auto response = run({});
  return {.status = 200,
          .reason = "OK",
          .body = response.dump(),
          .headers = {{"Server-Timing", *timing}},
          .log_details = stream_log->details};
}

HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend) {
  ParsedChatRequest parsed;
  const auto defaults = backend.sampling_defaults();
  parsed.max_tokens = defaults.max_tokens;
  parsed.sampling = defaults.sampling;
  if (auto error = ParseRequest(request, backend, &parsed); error.has_value()) {
    return std::move(*error);
  }
  try {
    const auto initial_output_state = backend.initial_output_state(parsed.chat);
    auto generation =
        backend.start_chat(parsed.chat, parsed.max_tokens, parsed.sampling,
                           request.is_cancelled, parsed.stream);
    if (parsed.stream) {
      return StreamingResponse(parsed, backend, std::move(generation),
                               initial_output_state);
    }
    return NonStreamingResponse(parsed, backend, generation,
                                initial_output_state);
  } catch (const TextGenerationError& exception) {
    return GenerationError(exception);
  } catch (const std::invalid_argument& exception) {
    return Error(400, "Bad Request", exception.what(), "invalid_prompt");
  } catch (const std::length_error& exception) {
    return Error(400, "Bad Request", exception.what(),
                 "context_length_exceeded");
  }
}

}  // namespace gufo::server
