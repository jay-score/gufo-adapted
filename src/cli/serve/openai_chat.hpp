#ifndef GUFO_SERVER_OPENAI_CHAT_HPP_
#define GUFO_SERVER_OPENAI_CHAT_HPP_

#include "src/cli/serve/http_server.hpp"

namespace gufo::server {

/// Handles the supported OpenAI Chat Completions subset. Streaming responses
/// consume scheduler-published token pieces from HttpResponse::streaming_body;
/// socket writes never own or execute model state.
HttpResponse HandleOpenAiChat(const HttpRequest& request,
                              TextGenerationBackend& backend);

/// Translate stateless Responses input, reasoning controls and function tools
/// into a chat request. The caller supplies defaults for omitted controls.
bool ParseOpenAiResponseChat(const json::Value& body, ChatRequest* chat,
                             std::string* error);

/// Responses output uses the same tool parser, reasoning/UTF-8 filter and
/// scheduler as Chat Completions, including streaming cancellation and cache
/// retention.
HttpResponse CreateOpenAiResponse(const HttpRequest& request,
                                  TextGenerationBackend& backend,
                                  const ChatRequest& chat,
                                  std::size_t max_tokens,
                                  const sampling::SamplingConfig& sampling,
                                  bool stream, bool parallel_tool_calls = true);

}  // namespace gufo::server

#endif  // GUFO_SERVER_OPENAI_CHAT_HPP_
