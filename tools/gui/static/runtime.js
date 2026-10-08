import { text, rawText, locale } from "./i18n.js";
import { $, api } from "./api.js";

export function createRuntime(onChange) {
  let runtime = { state: "stopped" }, closed = false;

  function renderMetrics(state) {
    const metrics = state.state === "ready" ? state.metrics : null;
    for (const [id, key, decimals, nonzero] of [
      ["live-prefill-speed", "live_prefill_tps", 1], ["live-generation-speed", "live_generation_tps", 1],
      ["prefill-speed", "prefill_tps", 1, true], ["generation-speed", "generation_tps", 1, true],
      ["prompt-total", "prompt_tokens_total", 0], ["generated-total", "generated_tokens_total", 0],
      ["requests-processing", "requests_processing", 0], ["requests-deferred", "requests_deferred", 0],
    ]) {
      const value = metrics?.[key];
      $(id).textContent = Number.isFinite(value) && (!nonzero || value > 0)
        ? value.toLocaleString(locale(), { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : "—";
    }
    const live = Number.isFinite(metrics?.requests_processing) && Number.isFinite(metrics?.requests_deferred);
    text($("metrics-note"), state.state === "ready"
      ? !metrics ? "Performance unavailable. Retrying…"
        : !live ? "Update the Gufo executable to enable live rates and request counts."
        : metrics.prompt_tokens_total === 0 && metrics.generated_tokens_total === 0
          && metrics.requests_processing === 0 && metrics.requests_deferred === 0
          ? "Waiting for requests."
          : "Live totals and request counts update about once per second."
      : ({ loading: "Performance will appear when the model is ready.",
           stopping: "Stopping Gufo…", failed: "Performance unavailable while Gufo is stopped.",
           disconnected: "Performance unavailable. Launcher disconnected." }[state.state] || "Start Gufo to see performance."));
    text($("metrics-detail"), metrics && !live
      ? "This executable reports totals after requests finish and includes cached prompt tokens. Completed speeds retain the latest nonzero measurement."
      : "Live rates cover all requests between polls. Prompt totals and live prefill exclude cached tokens. Completed speeds retain the latest nonzero measurement.");
  }

  function renderStatus(state) {
    runtime = state;
    renderMetrics(state);
    const active = state.pid != null || ["loading", "ready", "stopping"].includes(state.state);
    const launch = state.running;
    text($("running-preset"), launch ? "{state}: {name}{modified}" : "", launch ? {
      state: active ? "Running" : "Last launch", name: launch.preset_name,
      modified: launch.modified ? " (modified)" : "",
    } : {});
    const title = state.state[0].toUpperCase() + state.state.slice(1);
    text($("status"), state.state === "loading" ? "{state} · {seconds}s" : title,
      { state: title, seconds: state.elapsed_seconds });
    $("status").className = `status ${state.state}`;
    $("api-url").textContent = state.base_url || `http://127.0.0.1:${$("port").value || 8080}/v1`;
    text($("active-model"), state.model_name ? "Model: {name}" : "", { name: state.model_name });
    text($("runtime-note"), state.error || ({
      stopped: "Start Gufo to connect your OpenAI-compatible client.",
      loading: "Loading weights. This can take a few minutes; the process log shows progress.",
      ready: "Ready for requests · PID {pid}. Closing this tab keeps Gufo running.",
      stopping: "Stopping Gufo and releasing its GPU memory…",
    }[state.state] || "Check the process log."), { pid: state.pid });
    if (state.state === "failed") $("logs-details").open = true;
    const output = state.logs.join("\n") || "No process output yet.";
    if ($("logs").textContent !== output) {
      const atBottom = $("logs").scrollTop + $("logs").clientHeight >= $("logs").scrollHeight - 30;
      if (state.logs.length) rawText($("logs"), output); else text($("logs"), output);
      if (atBottom) $("logs").scrollTop = $("logs").scrollHeight;
    }
    onChange(state);
  }

  async function poll() {
    if (closed) return;
    try {
      const state = await api("status");
      if (!closed) renderStatus(state);
    } catch (error) {
      if (closed) return;
      text($("status"), "Disconnected");
      renderMetrics({ state: "disconnected" });
      $("status").className = "status failed";
      text($("runtime-note"), "Cannot reach the launcher. {detail}", { detail: error.message });
    }
    if (!closed) setTimeout(poll, 1000);
  }

  return {
    get state() { return runtime; },
    render: renderStatus,
    poll,
    close() {
      closed = true;
      renderMetrics({ state: "stopped" });
      text($("status"), "Launcher closed");
      $("status").className = "status stopped";
      text($("runtime-note"), "Gufo has stopped. You can close this tab.");
      text($("running-preset"), "");
    },
  };
}
