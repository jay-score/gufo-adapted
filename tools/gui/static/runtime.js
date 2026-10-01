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
        ? value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals }) : "—";
    }
    const live = Number.isFinite(metrics?.requests_processing) && Number.isFinite(metrics?.requests_deferred);
    $("metrics-note").textContent = state.state === "ready"
      ? !metrics ? "Performance unavailable. Retrying…"
        : !live ? "Update the Gufo executable to enable live rates and request counts."
        : metrics.prompt_tokens_total === 0 && metrics.generated_tokens_total === 0
          && metrics.requests_processing === 0 && metrics.requests_deferred === 0
          ? "Waiting for requests."
          : "Live totals and request counts update about once per second."
      : ({ loading: "Performance will appear when the model is ready.",
           stopping: "Stopping Gufo…", failed: "Performance unavailable while Gufo is stopped.",
           disconnected: "Performance unavailable. Launcher disconnected." }[state.state] || "Start Gufo to see performance.");
    $("metrics-detail").textContent = metrics && !live
      ? "This executable reports totals after requests finish and includes cached prompt tokens. Completed speeds retain the latest nonzero measurement."
      : "Live rates cover all requests between polls. Prompt totals and live prefill exclude cached tokens. Completed speeds retain the latest nonzero measurement.";
  }

  function renderStatus(state) {
    runtime = state;
    renderMetrics(state);
    const active = state.pid != null || ["loading", "ready", "stopping"].includes(state.state);
    const launch = state.running;
    $("running-preset").textContent = launch
      ? `${active ? "Running" : "Last launch"}: ${launch.preset_name}${launch.modified ? " (modified)" : ""}` : "";
    const title = state.state[0].toUpperCase() + state.state.slice(1);
    $("status").textContent = state.state === "loading" ? `${title} · ${state.elapsed_seconds}s` : title;
    $("status").className = `status ${state.state}`;
    $("api-url").textContent = state.base_url || `http://127.0.0.1:${$("port").value || 8080}/v1`;
    $("active-model").textContent = state.model_name ? `Model: ${state.model_name}` : "";
    $("runtime-note").textContent = state.error || ({
      stopped: "Start Gufo to connect your OpenAI-compatible client.",
      loading: "Loading weights. This can take a few minutes; the process log shows progress.",
      ready: `Ready for requests · PID ${state.pid}. Closing this tab keeps Gufo running.`,
      stopping: "Stopping Gufo and releasing its GPU memory…",
    }[state.state] || "Check the process log.");
    if (state.state === "failed") $("logs-details").open = true;
    const output = state.logs.join("\n") || "No process output yet.";
    if ($("logs").textContent !== output) {
      const atBottom = $("logs").scrollTop + $("logs").clientHeight >= $("logs").scrollHeight - 30;
      $("logs").textContent = output;
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
      $("status").textContent = "Disconnected";
      renderMetrics({ state: "disconnected" });
      $("status").className = "status failed";
      $("runtime-note").textContent = `Cannot reach the launcher. ${error.message}`;
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
      $("status").textContent = "Launcher closed";
      $("status").className = "status stopped";
      $("runtime-note").textContent = "Gufo has stopped. You can close this tab.";
      $("running-preset").textContent = "";
    },
  };
}
