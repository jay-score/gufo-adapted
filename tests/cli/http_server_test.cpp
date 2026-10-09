#include "src/cli/serve/http_server.hpp"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <semaphore>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include "src/cli/serve/logging.hpp"
#include "src/core/image.hpp"
#include "src/core/platform/socket.hpp"
#include "tests/core/image_fixtures.hpp"

namespace {

using gufo::server::HttpServer;
using gufo::server::TextGenerationBackend;

class FakeBackend final : public TextGenerationBackend {
public:
  struct Call {
    std::string prompt;
    gufo::server::ChatRequest chat;
    std::size_t max_tokens = 0;
    gufo::sampling::SamplingConfig sampling;
    std::string client_id;
    std::vector<std::string> stop_sequences;
  };
  Call LastCall() {
    const std::lock_guard lock(mutex_);
    return last_;
  }
  void SetOutput(std::string text) {
    const std::lock_guard lock(mutex_);
    output_ = std::move(text);
  }
  std::string model_id() const override { return "test"; }
  bool ready() const override { return true; }
  bool supports_image_input() const override { return vision; }
  gufo::ReasoningOptions reasoning_defaults() const override {
    return reasoning;
  }
  std::size_t count_tokens(std::string_view text) const override {
    return text.size();
  }
  Result complete(
      std::string_view prompt, std::size_t limit,
      const gufo::sampling::SamplingConfig& sampling,
      const CancellationCheck& cancel, const TokenCallback& token,
      std::string_view client_id = "anonymous",
      const std::vector<std::string>& stop_sequences = {}) override {
    ++calls;
    if (failure == 1)
      throw std::length_error("context exceeded");
    if (failure == 2)
      throw std::invalid_argument("invalid prompt");
    if (failure == 5)
      throw std::runtime_error("");
    Result result;
    if (wait_for_disconnect) {
      entered.release();
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(2);
      while (!(disconnected = cancel && cancel()) &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      finished.release();
      result.cancelled = disconnected;
      return result;
    }
    {
      const std::lock_guard lock(mutex_);
      last_ = {.prompt = std::string(prompt),
               .max_tokens = limit,
               .sampling = sampling,
               .client_id = std::string(client_id),
               .stop_sequences = stop_sequences};
      result.text = output_;
    }
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = 8;
    result.cache_hit = true;
    result.draft_accepted_tokens = 4;
    result.draft_tokens = 8;
    result.prefill_tokens = 2;
    result.prefill_ms = 4;
    result.completion_tokens = 1;
    result.decode_ms = 2;
    result.finish_reason =
        limit == 1 ? FinishReason::kLength : FinishReason::kStop;
    if (!forced_stop_sequence.empty()) {
      result.finish_reason = FinishReason::kStopSequence;
      result.stop_sequence = forced_stop_sequence;
    }
    if (token)
      (void)token(result.text);
    return result;
  }
  Result chat(const gufo::server::ChatRequest& request, std::size_t limit,
              const gufo::sampling::SamplingConfig& sampling,
              const CancellationCheck& cancel,
              const TokenCallback& token) override {
    auto result = complete("", limit, sampling, cancel, token,
                           request.client_id, request.stop_sequences);
    {
      const std::lock_guard lock(mutex_);
      last_.chat = request;
    }
    return result;
  }
  std::atomic<int> calls{0};
  bool vision{false};
  std::atomic<int> failure{0};
  std::string forced_stop_sequence;
  gufo::ReasoningOptions reasoning;
  bool wait_for_disconnect{false};
  std::atomic<bool> disconnected{false};
  std::binary_semaphore entered{0};
  std::binary_semaphore finished{0};

private:
  std::mutex mutex_;
  Call last_;
  std::string output_{"ok"};
};

class RunningServer {
public:
  explicit RunningServer(gufo::server::HttpServerOptions options = {},
                         bool handle_signals = false)
      : backend(std::make_shared<FakeBackend>()),
        server("127.0.0.1", 0, backend, nullptr, nullptr, nullptr,
               std::move(options)) {
    server.add("POST", "/echo", [](const auto& request, auto&) {
      return gufo::server::HttpResponse{.body = request.body};
    });
    server.add("POST", "/stream-error", [](const auto&, auto&) {
      return gufo::server::HttpResponse{
          .streaming_body = [](const auto& write) {
            (void)write("first chunk");
            throw std::runtime_error("injected stream failure");
          }};
    });
    server.add("POST", "/stream", [](const auto& request, auto&) {
      auto log = std::make_shared<gufo::server::HttpResponse::StreamLog>();
      return gufo::server::HttpResponse{
          .streaming_body =
              [log, fail = !request.body.empty(),
               reported = request.body == "reported"](const auto& write) {
                (void)write(std::string_view("a\0b", 3));
                (void)write("");
                (void)write("end");
                if (fail)
                  log->error_code = "injected";
                log->error_event_sent = reported;
              },
          .stream_log = log,
      };
    });
    std::string error;
    assert(server.start(&error));
    worker = std::jthread([this, handle_signals] {
      server.run(handle_signals);
      run_finished.release();
    });
  }
  ~RunningServer() {
    server.stop();
    worker.join();
  }

  int Connect() {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    assert(gufo::platform::SetSocketTimeout(fd, SO_RCVTIMEO,
                                            std::chrono::seconds{3}) == 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    assert(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr) == 1);
    assert(::connect(fd, reinterpret_cast<const sockaddr*>(&address),
                     sizeof(address)) == 0);
    return fd;
  }
  std::string Send(std::string_view request, bool half_close = false) {
    const int fd = Connect();
    while (!request.empty()) {
      const auto count =
          ::send(fd, request.data(), request.size(), MSG_NOSIGNAL);
      assert(count > 0);
      request.remove_prefix(static_cast<std::size_t>(count));
    }
    // A half-close allows malformed/truncated-body tests to complete without
    // timing-dependent sleeps.
    if (half_close)
      ::shutdown(fd, SHUT_WR);
    std::string response;
    char buffer[4096];
    for (;;) {
      const auto count = gufo::platform::SocketRead(fd, buffer, sizeof(buffer));
      assert(count >= 0);
      if (count == 0) {
        break;
      }
      response.append(buffer, static_cast<std::size_t>(count));
    }
    gufo::platform::CloseSocket(fd);
    return response;
  }

  std::string Post(std::string_view path, std::string_view body) {
    return Send("POST " + std::string(path) + " HTTP/1.1\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\n\r\n" + std::string(body));
  }

  std::shared_ptr<FakeBackend> backend;
  HttpServer server;
  std::binary_semaphore run_finished{0};
  std::jthread worker;
};

void ExpectStatus(const std::string& response, int status) {
  if (!response.starts_with("HTTP/1.1 " + std::to_string(status) + " ")) {
    std::cerr << response << '\n';
    std::abort();
  }
}

void TestVisionDiscovery() {
  RunningServer server;
  for (const bool vision : {false, true, false}) {
    server.backend->vision = vision;
    const auto models = server.Send("GET /v1/models HTTP/1.1\r\n\r\n");
    ExpectStatus(models, 200);
    const auto body =
        gufo::json::parse(models.substr(models.find("\r\n\r\n") + 4));
    const auto& model = body.find("data")->items().front();
    const auto& input =
        model.find("architecture")->find("input_modalities")->items();
    assert(model.member_str("id") == "test" && input[0].str() == "text");
    assert(input.size() == (vision ? 2 : 1));
    if (vision)
      assert(input[1].str() == "image");
    for (const auto* path : {"/props", "/props?model=test"}) {
      const auto response =
          server.Send("GET " + std::string(path) + " HTTP/1.1\r\n\r\n");
      ExpectStatus(response, 200);
      const auto props =
          gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
      assert(props.member_str("model") == "test");
      assert(props.find("modalities")->find("vision")->as_bool() == vision);
    }
  }
  ExpectStatus(server.Send("GET /props?model=other HTTP/1.1\r\n\r\n"), 404);
  assert(server.backend->calls == 0);
}

void TestAuthorization() {
  RunningServer secured({.api_key = "test-secret"});
  for (const std::string path :
       {"/health", "/ready", "/v1/models", "/v1/chat/completions",
        "/v1/responses", "/v1/messages", "/v1/audio/speech",
        "/v1/audio/transcriptions", "/v1/video/generations", "/echo"}) {
    ExpectStatus(secured.Send("POST " + path + " HTTP/1.1\r\n\r\n"), 401);
  }
  for (const std::string header :
       {"", "Authorization: Bearer wrong\r\n",
        "Authorization: Basic test-secret\r\n",
        "Authorization: Bearer test-secret\r\nAuthorization: Bearer "
        "test-secret\r\n"}) {
    const auto response =
        secured.Send("GET /v1/models HTTP/1.1\r\n" + header + "\r\n");
    ExpectStatus(response, 401);
    assert(response.find("WWW-Authenticate: Bearer") != std::string::npos);
    assert(response.find("test-secret") == std::string::npos);
  }
  assert(secured.backend->calls == 0);
  ExpectStatus(secured.Send("GET /v1/models HTTP/1.1\r\naUtHoRiZaTiOn: bEaReR  "
                            "test-secret\r\n\r\n"),
               200);
  ExpectStatus(secured.Send("OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"),
               204);
  const std::string body = R"({"prompt":"hi","max_tokens":1})";
  ExpectStatus(secured.Send("POST /v1/completions HTTP/1.1\r\nAuthorization: "
                            "Bearer test-secret\r\n"
                            "Content-Length: " +
                            std::to_string(body.size()) + "\r\n\r\n" + body),
               200);
  assert(secured.backend->calls == 1);
}

void TestRequestLogging() {
  std::ostringstream output;
  auto* previous = std::clog.rdbuf(output.rdbuf());
  std::string request_id;
  {
    RunningServer server;
    ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
    const auto response = server.Post(
        "/v1/chat/completions?private-query",
        R"({"model":"test","messages":[{"role":"user","content":"private-prompt"}],"stream":true})");
    ExpectStatus(response, 200);
    const auto header = response.find("X-Request-ID: ");
    assert(header != std::string::npos);
    const auto begin = header + std::string("X-Request-ID: ").size();
    request_id = response.substr(begin, response.find("\r\n", begin) - begin);

    const auto failed = server.Post("/stream-error", "");
    ExpectStatus(failed, 200);
    assert(failed.find("first chunk") != std::string::npos);
    assert(failed.find("HTTP/1.1", 1) == std::string::npos);
  }
  gufo::server::Logger::Info("test", "escaped\n\x1b[31m");
  std::clog.rdbuf(previous);
  const auto log = output.str();
  assert(log.find("request=" + request_id + " event=received") !=
         std::string::npos);
  assert(log.find("request=" + request_id + " event=completed") !=
         std::string::npos);
  assert(log.find("cache=memory") != std::string::npos);
  assert(log.find("acceptance_pct=50.0") != std::string::npos);
  assert(log.find("rss_mib=") != std::string::npos);
  assert(log.find("error_code=server_exception") != std::string::npos);
  assert(log.find("path=/health") == std::string::npos);
  assert(log.find("private-query") == std::string::npos);
  assert(log.find("private-prompt") == std::string::npos);
  assert(log.find("escaped\\x0a\\x1b[31m") != std::string::npos);
}

void TestFramingAndMetrics() {
  RunningServer server({.max_request_body_bytes = 8192});
  ExpectStatus(server.Send("GET /health HTTP/1.1\r\n\r\n"), 200);
  for (const std::string header :
       {"Content-Length: nope", "Content-Length: -1", "Content-Length: 4junk",
        "Content-Length: 18446744073709551616",
        "Content-Length: 4\r\nContent-Length: 3", "Transfer-Encoding: chunked",
        "broken-header"}) {
    ExpectStatus(server.Send("POST /echo HTTP/1.1\r\n" + header + "\r\n\r\n"),
                 400);
  }
  ExpectStatus(server.Send("POST /echo\r\n\r\n"), 400);
  ExpectStatus(server.Send("POST /echo HTTP/1.1 extra\r\n\r\n"), 400);
  ExpectStatus(
      server.Send("POST /echo HTTP/1.1\r\nContent-Length: 4\r\n\r\nx", true),
      400);
  for (const auto* path :
       {"/echo", "/v1/chat/completions", "/v1/responses", "/v1/messages"}) {
    // Headers alone must suffice: an oversized request never reaches inference.
    const auto response =
        server.Send("POST " + std::string(path) +
                    " HTTP/1.1\r\nContent-Length: 8193\r\n\r\n");
    ExpectStatus(response, 413);
    const auto body =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    const auto* error = body.find("error");
    assert(error != nullptr);
    assert(error->member_str("type") == "invalid_request_error");
    assert(error->member_str("code") == "payload_too_large");
    // OpenCode matches the first phrase; Pi matches the second, even when its
    // provider adapter exposes only the message rather than the full JSON body.
    const auto message = error->member_str("message");
    assert(message.find("Request entity too large") != std::string::npos);
    assert(message.find("reduce the length of the messages") !=
           std::string::npos);
    assert(response.find("Retry-After") == std::string::npos);
  }
  assert(server.backend->calls == 0);
  const std::string payload(8192, 'x');
  const auto echo = server.Send(
      "POST /echo HTTP/1.1\r\nContent-Length: 8192\r\nContent-Length: "
      "8192\r\n\r\n" +
      payload);
  ExpectStatus(echo, 200);
  assert(echo.substr(echo.find("\r\n\r\n") + 4) == payload);

  const std::string body = R"({"prompt":"hi","n_predict":1})";
  const auto response =
      server.Send("POST /completion HTTP/1.1\r\nContent-Length: " +
                  std::to_string(body.size()) + "\r\n\r\n" + body);
  ExpectStatus(response, 200);
  const auto parsed =
      gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
  const auto* timings = parsed.find("timings");
  assert(timings != nullptr);
  assert(timings->member_double("prompt_n") == 2);
  assert(timings->member_double("cache_n") == 8);
  assert(timings->member_double("prompt_per_second") == 500);
  assert(timings->member_double("prompt_per_token_ms") == 2);
}

void TestFallbackBackendMetrics() {
  namespace metrics = gufo::server::detail;
  RunningServer server;
  struct Endpoint {
    const char* path;
    const char* body;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/chat/completions",
                    R"({"messages":[{"role":"user","content":"hi"}]})"},
           Endpoint{"/v1/completions", R"({"prompt":"hi"})"},
           Endpoint{"/v1/responses", R"({"input":"hi"})"},
       }) {
    for (const bool stream : {false, true}) {
      const auto prompt_before = metrics::TotalPromptTokens().load();
      const auto generated_before = metrics::TotalGenTokens().load();
      auto body = gufo::json::parse(endpoint.body);
      body["model"] = "test";
      body["stream"] = stream;
      // This fork still rejects streamed raw completions before admission.
      const bool supported =
          !stream || std::string_view(endpoint.path) != "/v1/completions";
      ExpectStatus(server.Post(endpoint.path, body.dump()),
                   supported ? 200 : 400);
      const auto response = server.Send("GET /metrics HTTP/1.1\r\n\r\n");
      ExpectStatus(response, 200);
      // The backend reports 10 prompt tokens, of which 8 were cached.
      assert(response.find("llamacpp:prompt_tokens_total " +
                           std::to_string(prompt_before + (supported ? 2 : 0)) +
                           "\n") != std::string::npos);
      assert(
          response.find("llamacpp:tokens_predicted_total " +
                        std::to_string(generated_before + (supported ? 1 : 0)) +
                        "\n") != std::string::npos);
    }
  }
  struct Case {
    std::size_t prefill, cached;
    bool cancelled;
    std::size_t expected_prompt;
  };
  for (const auto& test : {
           Case{6, 4, false, 6},
           Case{0, 4, false, 6},
           Case{0, 10, false, 0},
           Case{0, 20, false, 0},
           Case{0, 0, true, 0},
           Case{3, 0, true, 3},
       }) {
    const auto prompt_before = metrics::TotalPromptTokens().load();
    const auto generated_before = metrics::TotalGenTokens().load();
    TextGenerationBackend::Result result;
    result.prompt_tokens = 10;
    result.cached_prompt_tokens = test.cached;
    result.prefill_tokens = test.prefill;
    result.completion_tokens = 2;
    result.cancelled = test.cancelled;
    gufo::server::RecordServerMetrics(result);
    assert(metrics::TotalPromptTokens().load() ==
           prompt_before + test.expected_prompt);
    assert(metrics::TotalGenTokens().load() == generated_before + 2);
  }
}

void TestMalformedToolCalls() {
  RunningServer server;
  server.backend->SetOutput(
      "<tool_call>{\"name\":\"f\",\"arguments\":{\"s\":\"raw\nnewline\"}}</"
      "tool_call>");
  for (bool responses : {false, true}) {
    auto body = gufo::json::parse(R"({"model":"test","tools":[
      {"type":"function","function":{"name":"f"}}]})");
    if (responses)
      body["input"] = "call f";
    else
      body["messages"] =
          gufo::json::parse(R"([{"role":"user","content":"call f"}])");
    const char* path = responses ? "/v1/responses" : "/v1/chat/completions";
    const auto buffered = server.Post(path, body.dump());
    ExpectStatus(buffered, 502);
    assert(buffered.find("malformed_tool_call") != std::string::npos &&
           buffered.find("Retry-After") == std::string::npos);
    body["stream"] = true;
    const auto streamed = server.Post(path, body.dump());
    ExpectStatus(streamed, 200);
    assert(streamed.find("HTTP/1.1", 1) == std::string::npos);
    assert(streamed.find(responses
                             ? "event: response.failed"
                             : "malformed_tool_call") != std::string::npos);
    assert(streamed.find(responses ? "event: response.completed"
                                   : "\"finish_reason\":\"stop\"") ==
           std::string::npos);
    body["stream"] = false;
    body[responses ? "max_output_tokens" : "max_tokens"] = 1;
    body["tool_choice"] = "required";
    ExpectStatus(server.Post(path, body.dump()), 200);
  }
}

void TestCompatibilityRequests() {
  RunningServer server;
  using gufo::json::parse;
  const auto response_body = [](const std::string& response) {
    ExpectStatus(response, 200);
    return parse(response.substr(response.find("\r\n\r\n") + 4));
  };
  struct Endpoint {
    const char *path, *body, *limit;
  };
  for (const auto& endpoint : {
           Endpoint{"/v1/completions", R"({"prompt":"hi"})", "max_tokens"},
           Endpoint{"/v1/responses", R"({"input":"hi"})", "max_output_tokens"},
           Endpoint{"/v1/messages",
                    R"({"messages":[{"role":"user","content":"hi"}]})",
                    "max_tokens"},
           Endpoint{"/completion", R"({"prompt":"hi"})", "n_predict"},
       }) {
    auto body = parse(endpoint.body);
    response_body(server.Post(endpoint.path, body.dump()));
    assert(server.backend->LastCall().max_tokens == 0);
    body["model"] = "test";
    body[endpoint.limit] = 1;
    body["temperature"] = 0.6;
    body["top_k"] = 40;
    body["top_p"] = 0.9;
    body["seed"] = 123;
    body["repeat_penalty"] = 1.1;
    const auto output = response_body(server.Post(endpoint.path, body.dump()));
    const auto last = server.backend->LastCall();
    for (int failure : {1, 2}) {
      server.backend->failure = failure;
      const auto rejected = server.Post(endpoint.path, body.dump());
      ExpectStatus(rejected, 400);
      assert(rejected.find(failure == 1
                               ? "context_length_exceeded"
                               : "invalid_prompt") != std::string::npos);
    }
    server.backend->failure = 0;
    assert(last.client_id == "127.0.0.1");
    assert(last.max_tokens == 1 && last.sampling.temperature == 0.6F &&
           last.sampling.top_k == 40 && last.sampling.top_p == 0.9F &&
           last.sampling.seed == 123 && last.sampling.repeat_penalty == 1.1F);
    if (std::string_view(endpoint.path) == "/v1/responses") {
      assert(output.member_str("status") == "incomplete");
      assert(output.find("incomplete_details")->member_str("reason") ==
             "max_output_tokens");
      assert(output.find("usage")->contains("input_tokens_details"));
    } else if (std::string_view(endpoint.path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "max_tokens");
    } else if (std::string_view(endpoint.path) == "/completion") {
      assert(output.find("stopped_length")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "length");
    }
    const int calls = server.backend->calls;
    for (const auto value : {"0", "-1", "1.5", "1e100", "\"1\"", "null"}) {
      auto invalid = body;
      invalid[endpoint.limit] = parse(value);
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    for (const auto field :
         {"stream", "echo", "store", "background", "tools", "stop", "reasoning",
          "output_config", "logit_bias"}) {
      if (std::string_view(endpoint.path) == "/v1/responses" &&
          std::string_view(field) == "stream")
        continue;
      auto invalid = body;
      invalid[field] = true;
      ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    }
    auto invalid = body;
    invalid["n"] = 1.4;
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 400);
    invalid = body;
    invalid["model"] = "wrong";
    ExpectStatus(server.Post(endpoint.path, invalid.dump()), 404);
    ExpectStatus(server.Post(endpoint.path, "[]"), 400);
    ExpectStatus(server.Post(endpoint.path, "{"), 400);
    assert(server.backend->calls == calls);
  }
  for (const bool stream : {false, true}) {
    auto image =
        gufo::json::parse(R"({"input":[{"role":"user","content":[
      {"type":"input_text","text":"describe"},
      {"type":"input_image","image_url":")" +
                          std::string(gufo::test::kLosslessWebP) + R"("}]}]})");
    image["stream"] = stream;
    ExpectStatus(server.Post("/v1/responses", image.dump()), 200);
    const auto message = server.backend->LastCall().chat.messages.front();
    assert(message.content == "describe" && message.images.size() == 1 &&
           message.images[0].offset == 8 &&
           *message.images[0].bytes ==
               gufo::core::ReadImageUrl(gufo::test::kLosslessWebP));
  }
  for (const bool responses : {false, true}) {
    for (const bool stream : {false, true}) {
      auto body = gufo::json::parse(responses ? R"({"input":[
        {"role":"user","content":"Read the file."},
        {"type":"custom_tool_call","call_id":"read1","name":"read","input":"file.png"},
        {"type":"custom_tool_call_output","call_id":"read1","output":[
          {"type":"input_text","text":" before "},
          {"type":"input_image","image_url":"data:image/png;base64,AQID"},
          {"type":"input_text","text":" after "}]}]})"
                                              : R"({"messages":[
        {"role":"user","content":"Read the file."},
        {"role":"assistant","content":null,"tool_calls":[
          {"id":"read1","type":"function","function":{"name":"read","arguments":"{}"}}]},
        {"role":"tool","tool_call_id":"read1","content":[
          {"type":"text","text":" before "},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,AQID"}},
          {"type":"text","text":" after "}]}]})");
      body["model"] = "test";
      body["stream"] = stream;
      ExpectStatus(
          server.Post(responses ? "/v1/responses" : "/v1/chat/completions",
                      body.dump()),
          200);
      const auto chat = server.backend->LastCall().chat;
      const auto& tool = chat.messages[2];
      assert(tool.tool_call_id == "read1" &&
             tool.content == " before  after " && tool.images.size() == 1 &&
             tool.images[0].offset == 8 &&
             *tool.images[0].bytes == std::vector<std::uint8_t>({1, 2, 3}));
      if (responses) {
        const auto& call = chat.messages[1].tool_calls.front();
        assert(call.id == "read1" && call.name == "read" &&
               call.arguments.front().value == "file.png" &&
               tool.name == "read");
      }
    }
  }
  const int calls = server.backend->calls;
  ExpectStatus(server.Post("/v1/completions", R"({"prompt":["one","two"]})"),
               400);
  ExpectStatus(server.Post("/v1/responses",
                           R"({"input":[{"role":"user","content":[
                           {"type":"input_text","text":"describe"},
                           {"type":"input_image","file_id":"file_123"}]}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"tool","content":"hi"}]})"),
               400);
  ExpectStatus(server.Post("/v1/messages", R"({"messages":[]})"), 400);
  ExpectStatus(
      server.Post("/infill", R"({"input_prefix":"one","input_suffix":"two"})"),
      501);
  ExpectStatus(server.Post("/v1/messages/count_tokens",
                           R"({"messages":[{"role":"user","content":"hi"}]})"),
               501);
  assert(server.backend->calls == calls);

  const auto response = response_body(server.Post(
      "/v1/responses",
      R"({"instructions":"Be concise.","input":[{"role":"user","content":[
          {"type":"input_text","text":"hi"}]}],"max_output_tokens":2,"store":false})"));
  assert(response.member_str("status") == "completed");
  const auto messages = server.backend->LastCall().chat.messages;
  assert(messages.size() == 2);
  assert(messages[0].role == gufo::tokenization::ChatRole::kSystem &&
         messages[0].content == "Be concise." && messages[1].content == "hi");
  for (const auto limit : {1, 2}) {
    const auto streaming = server.Post(
        "/v1/responses",
        std::string(R"({"input":"hi","stream":true,"max_output_tokens":)") +
            std::to_string(limit) + "}");
    ExpectStatus(streaming, 200);
    assert(streaming.find("text/event-stream") != std::string::npos);
    assert(streaming.find("event: response.created") != std::string::npos);
    assert(streaming.find("event: response.output_text.delta") !=
           std::string::npos);
    assert(streaming.find(limit == 1 ? "event: response.incomplete"
                                     : "event: response.completed") !=
           std::string::npos);
  }
  for (const int failure : {1, 5}) {
    server.backend->failure = failure;
    const std::string message =
        failure == 1 ? "context exceeded" : "generation failed";
    const auto failed_stream =
        server.Post("/v1/responses", R"({"input":"hi","stream":true})");
    ExpectStatus(failed_stream, 200);  // Fake backend fails after headers.
    assert(failed_stream.find("event: response.failed") != std::string::npos);
    assert(failed_stream.find("event: response.completed") ==
           std::string::npos);
    const auto failed_data = failed_stream.find(
        "data: ", failed_stream.find("event: response.failed"));
    assert(failed_data != std::string::npos);
    const auto failed_event = gufo::json::parse(
        std::string_view(failed_stream)
            .substr(failed_data + 6,
                    failed_stream.find("\n\n", failed_data) - failed_data - 6));
    const auto failed_error = failed_event.find("response")->find("error");
    assert(failed_error->member_str("code") == "server_error");
    assert(failed_error->member_str("message") == message);
    assert(failed_stream.ends_with("0\r\n\r\n"));
    const auto failed_chat = server.Post(
        "/v1/chat/completions",
        R"({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":true})");
    ExpectStatus(failed_chat, 200);
    assert(failed_chat.find("\"message\":\"" + message + "\"") !=
           std::string::npos);
    assert(failed_chat.find("\"code\":\"generation_failed\"") !=
           std::string::npos);
    assert(failed_chat.find("data: [DONE]\n\n") != std::string::npos);
  }
  server.backend->failure = 0;
  const auto continued = response_body(
      server.Post("/v1/responses",
                  R"({"input":[{"role":"user","content":"First question"},
        {"type":"reasoning","id":"rs_1","status":"completed",
         "summary":[{"type":"summary_text","text":"Thoughts"}]},
        {"type":"message","role":"assistant","content":[
          {"type":"output_text","text":"Answer","annotations":[]}]},
        {"role":"user","content":"Next question"}]})"));
  assert(continued.member_str("status") == "completed");
  const auto replay_messages = server.backend->LastCall().chat.messages;
  assert(replay_messages.size() == 3 &&
         replay_messages[1].thought == "Thoughts" &&
         replay_messages[1].content == "Answer");

  const auto tools =
      parse(R"([{"type":"function","name":"status","parameters":{}}])");
  auto tool_request = parse(
      R"({"input":"Check status","tool_choice":"required","parallel_tool_calls":false})");
  tool_request["tools"] = tools;
  server.backend->SetOutput(
      "<tool_call><function=status></function></tool_call>");
  const auto tool_response =
      response_body(server.Post("/v1/responses", tool_request.dump()));
  const auto& call = tool_response.find("output")->items()[0];
  assert(call.member_str("type") == "function_call" &&
         call.member_str("name") == "status");
  assert(server.backend->LastCall().chat.tools[0].name == "status");
  tool_request["stream"] = true;
  const auto tool_stream = server.Post("/v1/responses", tool_request.dump());
  ExpectStatus(tool_stream, 200);
  assert(tool_stream.find("response.function_call_arguments.delta") !=
         std::string::npos);
  tool_request["stream"] = false;
  tool_request["tool_choice"] = "auto";
  tool_request["input"] =
      parse(R"([{"role":"user","content":"Check status"}])");
  tool_request["input"].push_back(call);
  auto result = parse(R"({"type":"function_call_output","output":"ready"})");
  result["call_id"] = call.member_str("call_id");
  tool_request["input"].push_back(result);
  server.backend->SetOutput("Ready.");
  ExpectStatus(server.Post("/v1/responses", tool_request.dump()), 200);
  const auto tool_messages = server.backend->LastCall().chat.messages;
  assert(tool_messages.size() == 3 &&
         tool_messages[2].tool_call_id == call.member_str("call_id") &&
         tool_messages[2].content == "ready");
  tool_request["tool_choice"] = "required";
  ExpectStatus(server.Post("/v1/responses", tool_request.dump()), 502);
  const int tool_calls = server.backend->calls;
  result["call_id"] = "unknown";
  tool_request["input"].push_back(result);
  ExpectStatus(server.Post("/v1/responses", tool_request.dump()), 400);
  assert(server.backend->calls == tool_calls);
  server.backend->SetOutput("ok");

  const auto anthropic = response_body(
      server.Post("/v1/messages",
                  R"({"system":[{"type":"text","text":"Be concise."}],
          "messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],
          "max_tokens":2})"));
  assert(anthropic.member_str("stop_reason") == "end_turn");
  assert(server.backend->LastCall().chat.messages[0].content == "Be concise.");
}

void TestInvalidBindSettings() {
  for (const int port : {-1, 65536}) {
    HttpServer server("127.0.0.1", port, nullptr);
    std::string error;
    assert(!server.start(&error));
    assert(!error.empty());
  }
  HttpServer invalid("bad.address", 0, nullptr);
  std::string error;
  assert(!invalid.start(&error));
  assert(!error.empty());
}

void TestCompatibilityUtf8() {
  RunningServer server;
  server.backend->SetOutput("é中😀\xE2\x94!\xF0\x9F");
  const std::string expected = "é中😀\xEF\xBF\xBD!\xEF\xBF\xBD";
  for (const auto& [path, input] : {
           std::pair{"/v1/completions", R"({"prompt":"hi"})"},
           std::pair{"/v1/responses", R"({"input":"hi"})"},
           std::pair{"/v1/messages",
                     R"({"messages":[{"role":"user","content":"hi"}]})"},
           std::pair{"/completion", R"({"prompt":"hi"})"},
       }) {
    const auto response = server.Post(path, input);
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    std::string text;
    if (std::string_view(path) == "/v1/completions")
      text = output.find("choices")->items()[0].member_str("text");
    else if (std::string_view(path) == "/v1/responses")
      text = output.find("output")
                 ->items()[0]
                 .find("content")
                 ->items()[0]
                 .member_str("text");
    else if (std::string_view(path) == "/v1/messages")
      text = output.find("content")->items()[0].member_str("text");
    else
      text = output.member_str("content");
    assert(text == expected);
  }
}

void TestQueryParameters() {
  gufo::server::HttpRequest request;
  request.query = "notafter=wrong&note=after=wrong&after=right+value%26x";
  assert(request.query_param("after") == "right value&x");
  request.query = "notafter=wrong&note=after=wrong";
  assert(request.query_param("after").empty());
  request.query = "%61fter=encoded&broken=%xz&empty";
  assert(request.query_param("after") == "encoded");
  assert(request.query_param("broken") == "%xz");
  assert(request.query_param("empty").empty());
}

void TestPeerDisconnect() {
  RunningServer server;
  server.backend->wait_for_disconnect = true;
  const int fd = server.Connect();
  const std::string body = R"({"prompt":"hi","max_tokens":128})";
  const std::string request =
      "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
      "Content-Type: application/json\r\nContent-Length: " +
      std::to_string(body.size()) + "\r\n\r\n" + body;
  assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
         static_cast<ssize_t>(request.size()));
  assert(server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
  gufo::platform::CloseSocket(fd);
  assert(server.backend->finished.try_acquire_for(std::chrono::seconds(2)));
  assert(server.backend->disconnected);
}

void TestCompatibilityStopSequences() {
  RunningServer server;
  server.backend->forced_stop_sequence = "END";
  for (
      const auto& [path, body] : {
          std::pair{"/v1/completions", R"({"prompt":"hello","stop":"END"})"},
          std::pair{"/completion", R"({"prompt":"hello","stop":["END"]})"},
          std::pair{
              "/v1/messages",
              R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":["END"]})"},
      }) {
    const auto response = server.Post(path, body);
    ExpectStatus(response, 200);
    assert(server.backend->LastCall().stop_sequences ==
           std::vector<std::string>{"END"});
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    if (std::string_view(path) == "/v1/messages") {
      assert(output.member_str("stop_reason") == "stop_sequence");
      assert(output.member_str("stop_sequence") == "END");
    } else if (std::string_view(path) == "/completion") {
      assert(output.find("stopped_word")->as_bool());
      assert(!output.find("stopped_eos")->as_bool());
      assert(output.member_str("stopping_word") == "END");
    } else {
      assert(output.find("choices")->items()[0].member_str("finish_reason") ==
             "stop");
    }
  }
  const auto before = server.backend->calls.load();
  for (const auto* stop : {"null", "\"END\"", "[null]", "[\"\"]", "true"}) {
    auto body = gufo::json::parse(
        R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})");
    body["stop_sequences"] = gufo::json::parse(stop);
    ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  }
  ExpectStatus(server.Post("/v1/responses", R"({"input":"hi","stop":"END"})"),
               400);
  assert(server.backend->calls == before);
  auto body = gufo::json::parse(
      R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32,"stop_sequences":[]})");
  for (int i = 0; i < 65; ++i)
    body["stop_sequences"].push_back(std::to_string(i));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  body["stop_sequences"] = gufo::json::Value::array();
  for (int i = 0; i < 5; ++i)
    body["stop_sequences"].push_back(std::string(4096, 'x'));
  ExpectStatus(server.Post("/v1/messages", body.dump()), 400);
  server.backend->forced_stop_sequence.clear();
  for (const int limit : {1, 32}) {
    body["stop_sequences"] = gufo::json::Value::array();
    body["max_tokens"] = limit;
    const auto response = server.Post("/v1/messages", body.dump());
    ExpectStatus(response, 200);
    const auto output =
        gufo::json::parse(response.substr(response.find("\r\n\r\n") + 4));
    assert(output.member_str("stop_reason") ==
           (limit == 1 ? "max_tokens" : "end_turn"));
    assert(output.find("stop_sequence")->is_null());
  }
}

void TestCompatibilityThinkingDefaults() {
  RunningServer server;
  for (const bool enabled : {false, true}) {
    server.backend->reasoning = {
        .enabled = enabled,
        .effort = gufo::ReasoningEffort::kHigh,
        .preserve_thinking = true,
    };
    for (
        const auto& [path, body] : {
            std::pair{"/v1/responses", R"({"input":"hello"})"},
            std::pair{
                "/v1/messages",
                R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":32})"},
        }) {
      ExpectStatus(server.Post(path, body), 200);
      const auto reasoning = server.backend->LastCall().chat.reasoning;
      assert(reasoning.enabled == enabled);
      assert(reasoning.effort == gufo::ReasoningEffort::kHigh);
      assert(reasoning.preserve_thinking == true);
    }
  }

  for (const bool stream : {false, true}) {
    for (const auto& [effort, expected] :
         {std::pair{"low", gufo::ReasoningEffort::kLow},
          std::pair{"medium", gufo::ReasoningEffort::kMedium},
          std::pair{"xhigh", gufo::ReasoningEffort::kXHigh}}) {
      server.backend->reasoning.enabled = false;
      auto body =
          gufo::json::parse(R"({"input":"hello","reasoning":{"summary":"auto"},
        "include":["reasoning.encrypted_content"]})");
      body["reasoning"]["effort"] = effort;
      body["stream"] = stream;
      ExpectStatus(server.Post("/v1/responses", body.dump()), 200);
      const auto reasoning = server.backend->LastCall().chat.reasoning;
      assert(reasoning.enabled == true && reasoning.effort == expected &&
             reasoning.preserve_thinking == true);
    }
    server.backend->reasoning.enabled = true;
    auto disabled =
        gufo::json::parse(R"({"input":"hello","reasoning":{"effort":"none"}})");
    disabled["stream"] = stream;
    ExpectStatus(server.Post("/v1/responses", disabled.dump()), 200);
    assert(server.backend->LastCall().chat.reasoning.enabled == false);
  }
  server.backend->reasoning = {};
  // The reasoning fields Claude Code sends on every Messages request.
  // Adaptive keeps the server's thinking default; effort never enables it.
  // This endpoint returns text only, so the checks below cover acceptance
  // and the applied reasoning, not thinking-block rendering.
  for (const bool server_thinking : {false, true}) {
    server.backend->reasoning = {
        .enabled = server_thinking,
        .effort = gufo::ReasoningEffort::kLow,
        .preserve_thinking = true,
    };
    const auto adaptive = server.Post("/v1/messages",
                                      R"({"max_tokens":32,
        "thinking":{"type":"adaptive","display":"omitted"},
        "output_config":{"effort":"xhigh"},
        "messages":[{"role":"user","content":"hello"}]})");
    ExpectStatus(adaptive, 200);
    const auto reasoning = server.backend->LastCall().chat.reasoning;
    assert(reasoning.enabled == server_thinking);
    assert(reasoning.effort == gufo::ReasoningEffort::kXHigh);
  }
  server.backend->reasoning = {};
  ExpectStatus(server.Post("/v1/messages",
                           R"({"messages":[{"role":"user","content":"hello"}],
      "thinking":{"type":"disabled"},"output_config":{"effort":"low"}})"),
               200);
  assert(server.backend->LastCall().chat.reasoning.enabled == false);
  assert(server.backend->LastCall().chat.reasoning.effort ==
         gufo::ReasoningEffort::kLow);
  for (const auto* invalid :
       {R"({"messages":[{"role":"user","content":"hi"}],"thinking":true})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"enabled","budget_tokens":0}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "thinking":{"type":"adaptive","display":"full"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"adaptive"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"effort":"minimal"}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "output_config":{"format":{"type":"json_schema"}}})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "reasoning_effort":"high"})",
        R"({"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":true}})"}) {
    ExpectStatus(server.Post("/v1/messages", invalid), 400);
  }
  for (
      const auto* body :
      {R"({"input":"hello"})", R"({"input":"hello","reasoning":null})",
       R"({"input":"hello","reasoning":{"effort":null,"summary":"auto"},"include":[]})"}) {
    ExpectStatus(server.Post("/v1/responses", body), 200);
    const auto reasoning = server.backend->LastCall().chat.reasoning;
    const auto options = gufo::tokenization::ResolveQwenChatOptions(reasoning);
    assert(!reasoning.enabled.has_value() && !reasoning.effort.has_value());
    assert(options.enable_thinking &&
           options.reasoning_effort ==
               gufo::tokenization::QwenReasoningEffort::kXHigh);
  }
  const auto calls = server.backend->calls.load();
  for (const auto* body :
       {R"({"input":"hello","reasoning":true})",
        R"({"input":"hello","reasoning":{"effort":1}})",
        R"({"input":"hello","reasoning":{"effort":"invalid"}})",
        R"({"input":"hello","reasoning":{"mode":"pro"}})",
        R"({"input":"hello","reasoning":{"summary":"concise"}})",
        R"({"input":"hello","reasoning":{"summary":true}})",
        R"({"input":"hello","reasoning_effort":"low"})",
        R"({"input":"hello","include":"reasoning.encrypted_content"})",
        R"({"input":"hello","include":["unsupported"]})"}) {
    ExpectStatus(server.Post("/v1/responses", body), 400);
  }
  assert(server.backend->calls == calls);
}

void TestStreamingFraming() {
  RunningServer server;
  const std::string chunks = std::string("3\r\na\0b\r\n", 8) + "3\r\nend\r\n";
  for (const std::string body : {"", "fail", "reported"}) {
    const auto response = server.Post("/stream", body);
    ExpectStatus(response, 200);
    assert(response.find("Transfer-Encoding: chunked\r\n") !=
           std::string::npos);
    assert(response.substr(response.find("\r\n\r\n") + 4) ==
           chunks + (body == "fail" ? "" : "0\r\n\r\n"));
  }
  const auto thrown = server.Post("/stream-error", "");
  assert(thrown.substr(thrown.find("\r\n\r\n") + 4) == "b\r\nfirst chunk\r\n");
  // Existing HTTP/1.0 clients retain close-delimited framing.
  const auto legacy = server.Send("POST /stream HTTP/1.0\r\n\r\n");
  assert(legacy.find("Transfer-Encoding:") == std::string::npos);
  assert(legacy.substr(legacy.find("\r\n\r\n") + 4) ==
         std::string("a\0bend", 6));
}

void TestSignalShutdown() {
  // Process signals must never terminate the test runner itself. Prove that
  // both idle listeners and active generation return through normal cleanup.
  for (const int signal : {SIGINT, SIGTERM}) {
    for (const bool active : {false, true}) {
      const auto run_case = [&] {
        RunningServer server({}, true);
        // Accepting a request proves run() installed its handlers.
        ExpectStatus(server.Post("/echo", "ready"), 200);
        int fd = -1;
        if (active) {
          server.backend->wait_for_disconnect = true;
          fd = server.Connect();
          const std::string body = R"({"prompt":"hello"})";
          const std::string request =
              "POST /v1/completions HTTP/1.1\r\nContent-Length: " +
              std::to_string(body.size()) + "\r\n\r\n" + body;
          assert(::send(fd, request.data(), request.size(), MSG_NOSIGNAL) ==
                 static_cast<ssize_t>(request.size()));
          assert(
              server.backend->entered.try_acquire_for(std::chrono::seconds(2)));
        }
#ifdef _WIN32
        assert(::raise(signal) == 0);
#else
        assert(::kill(::getpid(), signal) == 0);
#endif
        assert(server.run_finished.try_acquire_for(std::chrono::seconds(2)));
        if (active) {
          assert(server.backend->disconnected);
          gufo::platform::CloseSocket(fd);
        }
      };
#ifdef _WIN32
      // No fork() on Windows: run in-process. The CRT calls the handler
      // synchronously on this thread, and without one SIGINT/SIGTERM would
      // end the runner, which fails the test just as loudly.
      run_case();
#else
      const pid_t child = ::fork();
      assert(child >= 0);
      if (child == 0) {
        ::alarm(5);
        run_case();
        ::_exit(0);
      }
      int status = 0;
      assert(::waitpid(child, &status, 0) == child);
      assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
#endif
    }
  }
}

}  // namespace

int main() {
  TestMalformedToolCalls();
  TestRequestLogging();
  TestInvalidBindSettings();
  TestQueryParameters();
  TestAuthorization();
  TestVisionDiscovery();
  TestFramingAndMetrics();
  TestFallbackBackendMetrics();
  TestCompatibilityRequests();
  TestCompatibilityStopSequences();
  TestCompatibilityThinkingDefaults();
  TestCompatibilityUtf8();
  TestPeerDisconnect();
  TestStreamingFraming();
  TestSignalShutdown();
  std::cout << "HTTP transport checks passed.\n";
}
