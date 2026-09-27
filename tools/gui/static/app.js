"use strict";

const $ = (id) => document.getElementById(id);
const form = $("settings-form");
const token = document.querySelector('meta[name="gufo-token"]').content;
let saved = {}, loaded = false, busy = false, closed = false;
let hasSaved = false, homeFolder = "", previousModel = "";
let runtime = { state: "stopped" }, previewTimer, previewVersion = 0;
let pickerTarget, pickerKind, listing, browseVersion = 0;

async function api(path, body) {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-Gufo-Token": token, "Content-Type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function values() {
  return Object.fromEntries(Object.keys(saved).map((key) => {
    const input = $(key);
    return [key, input.type === "checkbox" ? input.checked : input.type === "number" ? input.valueAsNumber : input.value];
  }));
}

function notice(message, success = false) {
  $("notice").textContent = message;
  $("notice").className = `notice${success ? " success" : ""}`;
  $("notice").hidden = !message;
}

function updateControls() {
  const active = ["loading", "ready", "stopping"].includes(runtime.state);
  $("start-button").disabled = !loaded || busy || active || closed;
  $("stop-button").disabled = busy || !active || runtime.state === "stopping" || closed;
  $("save-button").disabled = !loaded || busy || closed;
  $("exit-button").disabled = busy || closed;
  $("mtp-options").disabled = !$("mtp").checked;
  $("reasoning_effort").disabled = $("think").value === "off";
  $("mtp-files").hidden = !$("mtp").checked;
  for (const key of ["model", "mtp_model", "mmproj"]) {
    $(key).title = $(key).value;
    $(`${key}-filename`).textContent = $(key).value.split(/[\\/]/).pop();
  }
  if (loaded) $("saved-state").textContent = JSON.stringify(values()) === JSON.stringify(saved) ? (hasSaved ? "Settings saved" : "Not saved yet") : "Unsaved changes";
}

async function preview() {
  const version = ++previewVersion;
  try {
    const result = await api("preview", values());
    if (version === previewVersion) $("command").textContent = result.command;
  } catch (error) {
    if (version === previewVersion) $("command").textContent = error.message;
  }
}

function edited() {
  updateControls();
  clearTimeout(previewTimer);
  previewTimer = setTimeout(preview, 250);
}

function renderMetrics(state) {
  const metrics = state.state === "ready" ? state.metrics : null;
  for (const [id, key, speed] of [
    ["prefill-speed", "prefill_tps", true], ["generation-speed", "generation_tps", true],
    ["prompt-total", "prompt_tokens_total", false], ["generated-total", "generated_tokens_total", false],
  ]) {
    const value = metrics?.[key];
    $(id).textContent = Number.isFinite(value) && (!speed || value > 0)
      ? value.toLocaleString(undefined, { minimumFractionDigits: speed ? 1 : 0, maximumFractionDigits: speed ? 1 : 0 }) : "—";
  }
  $("metrics-note").textContent = state.state === "ready"
    ? !metrics ? "Performance unavailable. Retrying…"
      : metrics.prompt_tokens_total === 0 && metrics.generated_tokens_total === 0
        ? "Waiting for the first completed request."
        : "Updates about once per second after requests finish."
    : ({ loading: "Performance will appear when the model is ready.",
         stopping: "Stopping Gufo…", failed: "Performance unavailable while Gufo is stopped.",
         disconnected: "Performance unavailable. Launcher disconnected." }[state.state] || "Start Gufo to see performance.");
}

function renderStatus(state) {
  runtime = state;
  renderMetrics(state);
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
  updateControls();
}

async function action(callback) {
  busy = true; updateControls(); notice("");
  try { await callback(); }
  catch (error) { notice(error.message); }
  finally { busy = false; updateControls(); }
}

form.addEventListener("input", edited);
form.addEventListener("change", edited);
$("model").addEventListener("change", () => {
  if ($("model").value !== previousModel) {
    $("mtp_model").value = ""; $("mmproj").value = ""; $("mtp").checked = false;
    previousModel = $("model").value;
  }
});
form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || closed) return;
  action(async () => renderStatus(await api("start", values())));
});
$("save-button").addEventListener("click", () => action(async () => {
  if (!form.reportValidity()) return;
  saved = (await api("settings", values())).settings;
  hasSaved = true;
  notice("Settings saved for your next visit.", true);
}));
$("stop-button").addEventListener("click", () => action(async () => renderStatus(await api("stop", {}))));
$("exit-button").addEventListener("click", () => action(async () => {
  await api("exit", {});
  closed = true;
  renderMetrics({ state: "stopped" });
  $("status").textContent = "Launcher closed";
  $("status").className = "status stopped";
  $("runtime-note").textContent = "Gufo has stopped. You can close this tab.";
  notice("Launcher closed. Saved settings will be restored next time.", true);
}));
$("copy-url").addEventListener("click", () => action(async () => {
  await navigator.clipboard.writeText($("api-url").textContent);
  notice("API address copied.", true);
}));

async function browse(path) {
  const version = ++browseVersion;
  $("picker-error").textContent = "";
  $("picker-entries").textContent = "Loading folder…";
  $("select-folder").disabled = true;
  try {
    const result = await api(`browse?${new URLSearchParams({ path, kind: pickerKind })}`);
    if (version !== browseVersion) return;
    listing = result;
    $("picker-path").value = listing.path;
    $("picker-filter").value = "";
    $("picker-roots").replaceChildren(...listing.roots.map((root, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.textContent = index === 0 ? "Home" : root;
      button.addEventListener("click", () => browse(root));
      return button;
    }));
    $("select-folder").disabled = false;
    renderFiles();
  } catch (error) {
    if (version !== browseVersion) return;
    listing = null;
    $("picker-entries").textContent = "";
    $("picker-error").textContent = error.message;
  }
}

function choose(path) {
  $(pickerTarget).value = path;
  $(pickerTarget).dispatchEvent(new Event("change", { bubbles: true }));
  $("file-dialog").close();
  edited();
}

function renderFiles() {
  const filter = $("picker-filter").value.toLocaleLowerCase();
  const entries = listing?.entries.filter((entry) => entry.name.toLocaleLowerCase().includes(filter)) || [];
  $("picker-entries").replaceChildren(...entries.map((entry) => {
    const button = document.createElement("button"); button.type = "button";
    const icon = document.createElement("span"); icon.className = "file-icon";
    icon.textContent = entry.directory ? "DIR" : pickerKind === "gguf" ? "GGUF" : "EXE";
    const name = document.createElement("span"); name.textContent = entry.name;
    button.append(icon, name);
    if (!entry.directory && entry.shards > 1) {
      const shards = document.createElement("small"); shards.textContent = `${entry.shards} shards`;
      button.append(shards);
    }
    button.addEventListener("click", () => entry.directory ? browse(entry.path) : choose(entry.path));
    return button;
  }));
  if (!entries.length) $("picker-entries").textContent = "No matching files or folders.";
}

document.querySelectorAll("[data-browse]").forEach((button) => button.addEventListener("click", () => {
  pickerTarget = button.dataset.browse; pickerKind = button.dataset.kind;
  $("picker-title").textContent = pickerKind === "folder" ? "Choose your models folder" : `Choose ${pickerTarget === "executable" ? "Gufo executable" : pickerTarget === "mtp_model" ? "MTP sidecar" : pickerTarget === "mmproj" ? "vision projector" : "main model"}`;
  $("select-folder").hidden = pickerKind !== "folder";
  $("picker-roots").replaceChildren();
  const homeButton = document.createElement("button");
  homeButton.type = "button"; homeButton.textContent = "Home";
  homeButton.addEventListener("click", () => browse(homeFolder));
  $("picker-roots").append(homeButton);
  $("file-dialog").showModal();
  const current = $(pickerTarget).value;
  let parent = current.replace(/[\\/][^\\/]*$/, "");
  if (/^[a-z]:$/i.test(parent)) parent += "\\";
  const path = pickerKind === "folder" ? current : current && parent !== current ? parent : $("models_dir").value;
  browse(path);
}));
$("close-picker").addEventListener("click", () => $("file-dialog").close());
$("file-dialog").addEventListener("close", () => { ++browseVersion; });
$("picker-filter").addEventListener("input", renderFiles);
$("picker-up").addEventListener("click", () => { if (listing) browse(listing.parent); });
$("picker-go").addEventListener("click", () => browse($("picker-path").value));
$("picker-path").addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); browse(event.target.value); } });
$("select-folder").addEventListener("click", () => { if (listing) choose(listing.path); });

async function poll() {
  if (closed) return;
  try {
    const state = await api("status");
    if (!closed) renderStatus(state);
  }
  catch (error) {
    if (closed) return;
    $("status").textContent = "Disconnected";
    renderMetrics({ state: "disconnected" });
    $("status").className = "status failed";
    $("runtime-note").textContent = `Cannot reach the launcher. ${error.message}`;
  }
  if (!closed) setTimeout(poll, 1000);
}

(async () => {
  try {
    const result = await api("settings"); saved = result.settings;
    hasSaved = result.saved; homeFolder = result.home; previousModel = saved.model;
    for (const [key, value] of Object.entries(saved)) {
      if ($(key).type === "checkbox") $(key).checked = value;
      else $(key).value = value;
    }
    $("settings-location").textContent = `Settings · ${result.path}`;
    loaded = true; updateControls(); preview();
    if (result.warning) notice(`${result.warning} Defaults are shown; your saved file has not been changed.`);
    poll();
  } catch (error) { notice(`Could not load settings. ${error.message}`); }
})();
