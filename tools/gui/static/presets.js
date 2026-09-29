import { $, api, notice } from "./api.js";
import { sameSettings } from "./settings-form.js";

function ask(dialog) {
  return new Promise((resolve) => {
    dialog.returnValue = "cancel";
    dialog.addEventListener("close", () => resolve(dialog.returnValue), { once: true });
    dialog.showModal();
  });
}

export function createPresets(initial, form, action, onChange) {
  let document = initial.document;
  let selectedId = document.selected_preset;
  let hasSaved = initial.saved;
  let renderedRevision = -1;

  function selected() { return document.presets.find((preset) => preset.id === selectedId); }
  function savedValues() { return { ...selected().settings, ...document.launcher }; }
  function dirty() { return !sameSettings(form.values(), savedValues()); }

  function render() {
    if (renderedRevision !== document.revision) {
      $("preset-select").replaceChildren(...document.presets.map((preset) => new Option(preset.name, preset.id)));
      renderedRevision = document.revision;
    }
    $("preset-select").value = selectedId;
    $("delete-preset").disabled = document.presets.length === 1;
    $("saved-state").textContent = dirty() ? "Unsaved changes" : hasSaved ? "Preset saved" : "Not saved yet";
  }

  async function change(request) {
    const result = await api("presets", { ...request, revision: document.revision });
    document = result.document;
    hasSaved = true;
  }

  async function save() {
    if (!form.valid()) return false;
    await change({ action: "save", id: selectedId, settings: form.values() });
    form.apply(savedValues());
    onChange();
    notice("Preset and shared launcher settings saved for your next visit.", true);
    return true;
  }

  async function mayLeave() {
    if (!dirty()) return true;
    const choice = await ask($("unsaved-dialog"));
    return choice === "discard" || (choice === "save" && await save());
  }

  $("preset-select").addEventListener("change", () => {
    const next = $("preset-select").value;
    $("preset-select").value = selectedId;
    action(async () => {
      if (!await mayLeave()) return;
      selectedId = next;
      form.apply(savedValues());
      onChange();
    });
  });
  $("save-button").addEventListener("click", () => action(save));

  for (const [button, operation] of [["save-as-preset", "create"], ["rename-preset", "rename"]]) {
    $(button).addEventListener("click", () => action(async () => {
      if (operation === "create" && !form.valid()) return;
      $("preset-name-title").textContent = operation === "create" ? "Save as a new preset" : "Rename preset";
      $("preset-name").value = operation === "create" ? "" : selected().name;
      if (await ask($("preset-name-dialog")) !== "save") return;
      await change({ action: operation, id: selectedId, name: $("preset-name").value,
                     ...(operation === "create" ? { settings: form.values() } : {}) });
      if (operation === "create") {
        selectedId = document.selected_preset;
        form.apply(savedValues());
      }
      onChange();
      notice(operation === "create" ? "New preset saved." : "Preset renamed. Unsaved form edits are unchanged.", true);
    }));
  }

  $("delete-preset").addEventListener("click", () => action(async () => {
    if (!await mayLeave()) return;
    $("delete-preset-name").textContent = selected().name;
    if (await ask($("delete-dialog")) !== "delete") return;
    await change({ action: "delete", id: selectedId });
    selectedId = document.selected_preset;
    form.apply(savedValues());
    onChange();
    notice("Preset deleted. Model files and the running server are unchanged.", true);
  }));

  return { get id() { return selectedId; }, dirty, render };
}
