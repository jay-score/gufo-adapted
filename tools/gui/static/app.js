import { $, api, notice } from "./api.js";
import { createSettingsForm, sameSettings } from "./settings-form.js";
import { createPresets } from "./presets.js";
import { initFilePicker } from "./file-picker.js";
import { createRuntime } from "./runtime.js";

let form, presets, busy = false, closed = false;
let previewTimer, previewVersion = 0;
const runtime = createRuntime(updateControls);

function updateControls() {
  const state = runtime.state;
  const active = state.pid != null || ["loading", "ready", "stopping"].includes(state.state);
  $("settings-form").inert = !form || busy || closed;
  $("start-button").disabled = !form || busy || active || closed;
  $("stop-button").disabled = busy || !active || state.state === "stopping" || closed;
  $("exit-button").disabled = busy || closed;
  presets?.render();
  const different = active && state.running && form && !sameSettings(form.values(), state.running.settings);
  $("next-launch-note").textContent = different
    ? "These settings differ from the running server. Changes apply to the next launch."
    : "Changes apply to the next launch.";
}

async function preview(version) {
  try {
    const result = await api("preview", form.values());
    if (version === previewVersion) $("command").textContent = result.command;
  } catch (error) {
    if (version === previewVersion) $("command").textContent = error.message;
  }
}

function edited() {
  updateControls();
  clearTimeout(previewTimer);
  const version = ++previewVersion;
  previewTimer = setTimeout(() => preview(version), 250);
}

async function action(callback) {
  if (busy || closed) return;
  busy = true; updateControls(); notice("");
  try { return await callback(); }
  catch (error) { notice(error.message); }
  finally { busy = false; updateControls(); }
}

$("settings-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action(async () => runtime.render(await api("start", { settings: form.values(), preset_id: presets.id })));
});
$("stop-button").addEventListener("click", () => action(async () => runtime.render(await api("stop", {}))));
$("exit-button").addEventListener("click", () => action(async () => {
  await api("exit", {});
  closed = true;
  runtime.close();
  notice("Launcher closed. Saved presets will be restored next time.", true);
}));
$("copy-url").addEventListener("click", () => action(async () => {
  await navigator.clipboard.writeText($("api-url").textContent);
  notice("API address copied.", true);
}));
window.addEventListener("beforeunload", (event) => {
  if (!closed && presets?.dirty()) { event.preventDefault(); event.returnValue = ""; }
});

updateControls();
(async () => {
  try {
    const result = await api("settings");
    form = createSettingsForm(result.settings, result.families, result.modes, edited);
    presets = createPresets(result, form, action, edited);
    initFilePicker(result.home);
    $("settings-location").textContent = `Settings · ${result.path}`;
    edited();
    if (result.warning) notice(`${result.warning} Defaults are shown; your saved file has not been changed.`);
    runtime.poll();
  } catch (error) { notice(`Could not load settings. ${error.message}`); }
})();
