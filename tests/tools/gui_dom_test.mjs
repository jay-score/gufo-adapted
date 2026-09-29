// npm install --prefix build/gui-test-js --no-save --package-lock=false jsdom@26.1.0
// node --experimental-vm-modules --test tests/tools/gui_dom_test.mjs
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import test from "node:test";

const root = fileURLToPath(new URL("../../", import.meta.url));
const require = createRequire(path.join(root, "build/gui-test-js/package.json"));
const { JSDOM } = require("jsdom");
const python = process.env.GUI_TEST_PYTHON || path.join(root, process.platform === "win32"
  ? "build/gui-env/Scripts/python.exe" : "build/gui-env/bin/python");
const fixture = JSON.parse(execFileSync(python, ["-c", `
import json, re, sys, tempfile, threading
from pathlib import Path
sys.path.insert(0, 'tools/gui')
from server import create_app
from launcher_process import ProcessManager
with tempfile.TemporaryDirectory() as directory:
    app = create_app(Path(directory) / 'settings.json', ProcessManager(), threading.Event())
    client = app.test_client()
    html = client.get('/').text
    token = re.search('name="gufo-token" content="([^"]+)"', html)[1]
    initial = client.get('/api/settings', headers={'X-Gufo-Token': token}).json
    print(json.dumps({'html': html, 'initial': initial}))
`], { cwd: root, encoding: "utf8" }));

async function setup(t) {
  const dom = new JSDOM(fixture.html, { url: "http://localhost/", runScripts: "outside-only" });
  t.after(() => dom.window.close());
  const { window } = dom;
  window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  window.HTMLDialogElement.prototype.close = function (value = this.returnValue) {
    this.returnValue = value; this.open = false; this.dispatchEvent(new window.Event("close"));
  };
  const modules = new Map();
  async function load(name) {
    if (modules.has(name)) return modules.get(name);
    const code = await readFile(path.join(root, "tools/gui/static", name), "utf8");
    const module = new vm.SourceTextModule(code, { context: dom.getInternalVMContext(), identifier: name });
    modules.set(name, module);
    await module.link((specifier) => load(specifier.replace(/^\.\//, "")));
    return module;
  }
  async function use(name) {
    const module = await load(name);
    if (module.status !== "evaluated") await module.evaluate();
    return module.namespace;
  }
  return { window, use, $: (id) => window.document.getElementById(id),
           initial: structuredClone(fixture.initial) };
}

function edit(env, id, value) {
  env.$(id).value = value;
  env.$(id).dispatchEvent(new env.window.Event("change", { bubbles: true }));
}
const tick = () => new Promise((resolve) => setImmediate(resolve));

test("restoring a preset keeps sidecars; changing its model clears them", async (t) => {
  const env = await setup(t);
  const { createSettingsForm } = await env.use("settings-form.js");
  const { settings, families, modes } = env.initial;
  const form = createSettingsForm(settings, families, modes, () => {});
  const values = { ...settings, model_family: "qwen38_flash_next", model: "C:\\main.gguf",
    mtp_model: "C:\\draft.gguf", mmproj: "C:\\mmproj.gguf", speculative: "mtp", temperature: 0.42 };
  form.apply(values);
  assert.equal(form.values().mtp_model, values.mtp_model);
  assert.equal(form.values().mmproj, values.mmproj);
  assert.equal(env.$("mtp-options").hidden, false);
  edit(env, "model_family", "qwen38_27b");
  assert.equal(form.values().speculative, "off");
  assert.equal(form.values().temperature, 0.42);
  assert.deepEqual(Array.from(env.$("speculative").options, (option) => option.value), ["off", "dflash2"]);
  form.apply(values);
  edit(env, "model", "C:\\different.gguf");
  for (const key of ["mtp_model", "dflash_model", "mmproj"]) assert.equal(form.values()[key], "");
  assert.equal(form.values().speculative, "off");
});

test("preset switching supports cancel, discard, and save without starting the server", async (t) => {
  const env = await setup(t);
  const { createSettingsForm } = await env.use("settings-form.js");
  const { createPresets } = await env.use("presets.js");
  const { settings, families, modes } = env.initial;
  const document = env.initial.document;
  document.presets.push({ id: "second", name: "Second", settings: {
    ...document.presets[0].settings, context: 65536, model: "C:\\second.gguf",
    speculative: "mtp", mtp_model: "C:\\sidecar.gguf", mmproj: "C:\\projector.gguf",
  } });
  let presets;
  const form = createSettingsForm(settings, families, modes, () => presets?.render());
  const requests = [];
  env.window.fetch = async (url, options) => {
    requests.push([url, JSON.parse(options.body)]);
    const body = requests.at(-1)[1];
    assert.equal(url, "/api/presets");
    assert.equal(body.action, "save");
    document.presets[0].settings = { ...document.presets[0].settings, context: body.settings.context };
    document.revision++;
    return { ok: true, json: async () => ({ document: structuredClone(document) }) };
  };
  const errors = [];
  const action = async (fn) => { try { await fn(); } catch (error) { errors.push(error); } };
  presets = createPresets(env.initial, form, action, () => presets.render());
  presets.render();
  edit(env, "context", "8192");
  edit(env, "preset-select", "second");
  assert.equal(env.$("unsaved-dialog").open, true);
  env.$("unsaved-dialog").close("cancel"); await tick();
  assert.equal(presets.id, "default");
  assert.equal(form.values().context, 8192);
  edit(env, "preset-select", "second");
  env.$("unsaved-dialog").close("discard"); await tick();
  assert.equal(presets.id, "second");
  assert.equal(form.values().mtp_model, "C:\\sidecar.gguf");
  assert.equal(form.values().mmproj, "C:\\projector.gguf");
  assert.equal(presets.dirty(), false);
  edit(env, "preset-select", "default"); await tick();
  edit(env, "context", "16384");
  edit(env, "preset-select", "second");
  env.$("unsaved-dialog").close("save"); await tick();
  assert.equal(presets.id, "second");
  assert.equal(requests.length, 1);
  assert.equal(requests[0][1].settings.context, 16384);
  assert.deepEqual(errors, []);
});

test("file picker selection follows the same model-change rules", async (t) => {
  const env = await setup(t);
  const { createSettingsForm } = await env.use("settings-form.js");
  const { initFilePicker } = await env.use("file-picker.js");
  const form = createSettingsForm(env.initial.settings, env.initial.families, env.initial.modes, () => {});
  form.apply({ ...env.initial.settings, model: "C:\\old.gguf", mtp_model: "C:\\old-mtp.gguf", speculative: "mtp" });
  env.window.fetch = async () => ({ ok: true, json: async () => ({ path: "C:\\", parent: "C:\\", roots: ["C:\\"],
    entries: [{ name: "new.gguf", path: "C:\\new.gguf", directory: false, shards: 1 }] }) });
  initFilePicker("C:\\");
  env.window.document.querySelector('[data-browse="model"]').click(); await tick();
  env.$("picker-entries").querySelector("button").click();
  assert.equal(form.values().model, "C:\\new.gguf");
  assert.equal(form.values().mtp_model, "");
  assert.equal(form.values().speculative, "off");
  assert.equal(env.$("file-dialog").open, false);
});

test("runtime display reports the launch snapshot independently of form edits", async (t) => {
  const env = await setup(t);
  const { createRuntime } = await env.use("runtime.js");
  const runtime = createRuntime(() => {});
  const state = { state: "ready", pid: 123, logs: [], base_url: "http://127.0.0.1:8080/v1", model_name: "gufo",
    running: { preset_name: "Flash 32K", modified: true, settings: env.initial.settings } };
  runtime.render(state);
  edit(env, "context", "65536");
  assert.equal(env.$("running-preset").textContent, "Running: Flash 32K (modified)");
  runtime.render({ ...state, state: "stopped", pid: null });
  assert.equal(env.$("running-preset").textContent, "Last launch: Flash 32K (modified)");
});

test("save as, rename, failed save, and delete keep the correct draft", async (t) => {
  const env = await setup(t);
  const { createSettingsForm } = await env.use("settings-form.js");
  const { createPresets } = await env.use("presets.js");
  const { settings, families, modes } = env.initial;
  let document = structuredClone(env.initial.document);
  let failSave = false, presets;
  const form = createSettingsForm(settings, families, modes, () => presets?.render());
  const errors = [];
  const action = async (fn) => { try { await fn(); } catch (error) { errors.push(error.message); } };
  env.window.fetch = async (url, options) => {
    assert.equal(url, "/api/presets");
    const body = JSON.parse(options.body);
    if (failSave) return { ok: false, json: async () => ({ error: "disk full" }) };
    if (body.action === "create") {
      const values = { ...body.settings };
      for (const key of ["executable", "models_dir", "port"]) delete values[key];
      document.presets.push({ id: "copy", name: body.name, settings: values });
      document.selected_preset = "copy";
    } else if (body.action === "rename") {
      document.presets[1].name = body.name;
    } else if (body.action === "delete") {
      document.presets.pop(); document.selected_preset = "default";
    } else assert.fail(`Unexpected action ${body.action}`);
    document.revision++;
    return { ok: true, json: async () => ({ document: structuredClone(document) }) };
  };
  presets = createPresets(env.initial, form, action, () => presets.render());
  presets.render();
  edit(env, "context", "8192");
  env.$("save-as-preset").click();
  env.$("preset-name").value = "Copy";
  env.$("preset-name-dialog").close("save"); await tick();
  assert.equal(presets.id, "copy");
  assert.equal(presets.dirty(), false);
  edit(env, "context", "16384");
  env.$("rename-preset").click();
  env.$("preset-name").value = "Renamed";
  env.$("preset-name-dialog").close("save"); await tick();
  assert.equal(form.values().context, 16384);
  assert.equal(presets.dirty(), true);
  assert.equal(env.$("preset-select").selectedOptions[0].textContent, "Renamed");
  failSave = true;
  env.$("save-button").click(); await tick();
  assert.equal(presets.dirty(), true);
  assert.equal(form.values().context, 16384);
  assert.deepEqual(errors, ["disk full"]);
  failSave = false;
  env.$("delete-preset").click();
  env.$("unsaved-dialog").close("discard"); await tick();
  env.$("delete-dialog").close("delete"); await tick();
  assert.equal(presets.id, "default");
  assert.equal(env.$("delete-preset").disabled, true);
  assert.equal(form.values().context, settings.context);
});

test("application initializes and previews edits without launching or saving", async (t) => {
  const env = await setup(t);
  const requests = [];
  env.window.fetch = async (url, options) => {
    requests.push(url);
    const response = url === "/api/settings" ? structuredClone(env.initial)
      : url === "/api/status" ? { state: "stopped", logs: [], pid: null }
      : url === "/api/preview" ? { command: `gufo --context ${JSON.parse(options.body).context}` }
      : assert.fail(`Unexpected request ${url}`);
    return { ok: true, json: async () => response };
  };
  await env.use("app.js"); await tick();
  assert.equal(env.$("notice").textContent, "");
  assert.equal(env.$("settings-form").inert, false);
  assert.equal(env.$("start-button").disabled, false);
  assert.equal(env.$("preset-select").selectedOptions[0].textContent, "Default");
  edit(env, "context", "8192");
  await new Promise((resolve) => setTimeout(resolve, 300));
  assert.equal(env.$("command").textContent, "gufo --context 8192");
  assert.equal(env.$("saved-state").textContent, "Unsaved changes");
  assert.deepEqual([...new Set(requests)].sort(), ["/api/preview", "/api/settings", "/api/status"]);
});
