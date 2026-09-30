import { $, api } from "./api.js";

export function initFilePicker(homeFolder) {
  let pickerTarget, pickerKind, listing, browseVersion = 0;

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
    $("picker-title").textContent = button.getAttribute("aria-label") || (pickerKind === "folder" ? "Choose a folder" : "Choose a file");
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

}
