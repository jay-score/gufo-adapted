# Local Gufo launcher

A small browser control panel for one Gufo text server. Select existing model
files, configure a curated set of server and sampler options, save named presets,
and start or stop inference. The engine remains a separate executable.

## Start on Windows

For first-time setup, double-click **[setup-windows.bat](../../setup-windows.bat)**
to install missing prerequisites, build Gufo, and open the GUI. See the
[Windows instructions](../../docs/WINDOWS.md#guided-setup) for details and manual setup.
On subsequent runs, double-click **[launch-gui.bat](../../launch-gui.bat)**
in the repository folder, or create a desktop shortcut to it. If startup fails,
the terminal stays open so you can read the error.

You can still launch from a terminal and pass options:

```powershell
.\launch-gui.bat -Port 8090
```

The first run creates `build/gui-env` and installs Flask and Waitress. Subsequent
runs reuse it; a changed requirements file triggers a dependency update. The
launcher opens at **http://127.0.0.1:8090**. Keep its terminal open.

1. Choose your **models folder**, then select a main GGUF. Browse into subfolders
   or paste a full path. Split models appear once; select their first shard.
2. Choose the **Model family**, then **Speculative decoding**: **Off**, **MTP**
   for Flash-Next, or **DFlash2** for Qwen3.8-27B. MTP uses Flash-Next's matching shared Q8_0 sidecar;
   DFlash2 uses a matching Qwen3.8-27B DFlash2 draft. Selecting a file does not
   change the mode. The picker shows local files; Gufo still decides which
   models and tensor formats it supports.
3. Optionally choose a **vision projector**. A blank field means **auto-detect**,
   not disabled vision. An explicitly selected incompatible projector fails
   during model loading.
4. Check the **Gufo executable**. It defaults to this checkout's
   `build/release/gufo.exe`; another built copy can be selected. Leave that
   executable alongside its runtime DLLs and kernel libraries.
5. Adjust settings, click **Save preset** (or **Save as…** for a new name), then **Start Gufo**. Wait for
   **Ready**, and connect your chat client to the displayed API address and model
   name. The defaults are `http://127.0.0.1:8080/v1` and `gufo`.

The launcher starts with 32K context, one session, a 2,048-token output limit,
and Flash-Next thinking sampler values (temperature 1, top-p 0.95, top-k 20,
min-p 0). These are editable launcher defaults, not universal model settings.
**Default thinking level** offers Auto, Low, Medium, and Xhigh. Auto omits
`--reasoning-effort`, leaving Qwen's default of xhigh. The other levels set a
server default that clients can override per request. The selection is saved;
when Thinking is Off, the level is disabled and its flag is omitted.
Changing the thinking controls does not rewrite your sampler settings. API
clients can override the server's generation defaults per request.

Settings live in `%LOCALAPPDATA%\Gufo\launcher.json`. Saving is explicit and
works while the engine is running. **Start** uses the current form without saving
it. Reopening the launcher restores the last saved preset without loading a model.

Choose a **Launch preset**, then click **Load preset** to apply its complete model configuration: model and
sidecar paths, API model name, context, sessions, sampling, reasoning, and
speculative options. **Save preset** updates it; **Save as…** creates an independent
copy. Save, Rename, and Delete act on the loaded preset, identified below the buttons.
**Rename** keeps its identity and **Delete** removes only the preset, never
model files. At least one preset must remain. Switching or deleting with unsaved
edits offers Save, Discard, or Cancel. Selecting a preset leaves the form unchanged
until you click **Load preset**. Loading does not save or start the server; only
choosing Save in the unsaved-edits prompt writes changes to disk. Preset names
must be unique, and up to 100 presets can be saved.

The executable, models folder, and API port are shared across presets. **Save
preset** and **Save as…** also save these shared values. The form describes the
next launch; the Server panel separately identifies the running preset and
whether it launched with unsaved changes. Saving, renaming, or deleting a preset
does not change a running process. Concurrent edits from a stale browser tab are
rejected; reload that tab before saving again.

Choosing or entering a different main model clears sidecars and resets speculative
decoding to Off. Loading a preset restores all its fields together, including its
sidecars. Changing the model family limits available speculative modes without
rewriting sampler values. **Use family defaults** explicitly applies the family's
curated defaults: Flash-Next's thinking sampler, or Qwen3.8-27B's model-default
thinking controls (leaving sampling unchanged). The **Other / existing
configuration** family preserves the existing options for unclassified models.
Family selection guides the controls; Gufo still validates artifact compatibility.

Version 1 settings, including the older MTP checkbox, appear as a **Default**
preset in memory. They are not rewritten on load. The next explicit save writes
version 2 and preserves the original as `launcher.json.bak` (an existing backup
is never replaced). Corrupt files likewise remain untouched until an explicit
save, which preserves a backup before replacement.

The button at the top right of the bar switches between the light paper palette
and a dark navy one; cards, fields, the status pill, metrics and the file picker
all follow it. Until you press it, the launcher follows your operating-system
theme. The choice is remembered in that browser's local storage, not in
`launcher.json`, so it is not part of **Save preset** and never reaches the
engine.

**Stop** releases the launched Gufo process. Closing a browser tab keeps it
running; **Exit launcher** or Ctrl+C in the launcher terminal stops both. On
Windows, Stop terminates the owned process tree and waits for it to exit.
A Windows Job Object also terminates that tree if the launcher crashes or its
terminal is closed abruptly. Failed shutdowns remain visible and can be retried
with Stop; Exit launcher does not report success while shutdown has failed.
Termination does not flush pending disk continuation cache writes. Snapshots
already written remain reusable.

## Reasoning preservation and disk caching

**Preserve reasoning in history** defaults to on, matching Qwen's template
default. The GUI explicitly passes `--preserve-thinking on` or `off`. This is
independent of generating new thoughts with **Thinking**. It preserves reasoning
supplied by the client; it cannot recover reasoning omitted from the request.
Chat Completions clients may override it with
`chat_template_kwargs.preserve_thinking`.

**More options → Enable disk continuation cache** defaults to off. When enabled,
the GUI passes `--cache-disk`, `--cache-disk-bytes`, and
`--cache-disk-staging-bytes`. The suggested folder is `%LOCALAPPDATA%\Gufo\cache`
on Windows (`~/.cache/gufo` otherwise); Gufo creates it when needed. The disk
limit defaults to 8 GiB and the RAM staging limit to 0 (automatic, at most 1 GiB).
Large snapshots need larger limits: Flash-Next/MTP at 262K context needs 8 GiB
staging if RAM permits. See [server cache behavior](../../docs/SERVER.md#hip-execution)
for skip behavior and budget details. Stop/Exit do not drain pending writes.

These controls are saved per preset. Existing presets receive the defaults in
memory without rewriting the saved file. Use **Save preset** to persist changes;
they apply on the next engine launch. RAM continuation caching remains active
with disk caching off.

## Options and checks

The **Performance** section polls Gufo's `/metrics` endpoint about once per
second while the server is ready, including when requests come from another
chat client. It shows prefill and generation tokens/second plus cumulative
prompt and generated token counts for the current server run. Speeds are the
last completed nonzero measurements, not live rates during generation; a fully
cached prompt retains the previous prefill speed. Prefill speed excludes cached
tokens, while the prompt total includes them. Missing measurements appear as
dashes, and unavailable metrics are retried without adding process-log entries.
Restart the launcher after updating its code to load the new panel.

- The control panel and Gufo both bind to localhost. Their ports must differ.
  `gui.ps1 -Port 8091` changes the control panel's port; the form sets Gufo's port.
- `gui.ps1 -Config C:\path\launcher.json` uses another settings file.
- `gui.ps1 -NoBrowser` starts without opening a browser.
- More options exposes disk caching, sessions, penalties and Flash-Next MTP options. MTP
  options are passed only when MTP is enabled. Latin draft vocabulary is
  intended for English/code; survival and lookup target single-session use.
- DFlash2 uses the engine's adaptive defaults. MTP options, including its draft
  token cap, are not passed to DFlash2. If Qwen3.8-27B reports `MTP HTTP decoding
  requires a Qwen Flash-Next model`, select DFlash2 and choose its draft in the
  DFlash2 field. Older GUI versions only expose MTP; update and restart the
  launcher to see the mode selector.
- Start validates paths, GGUF magic, split-file presence and numeric options.
  It does not establish model/sidecar compatibility or estimate available VRAM.
- Command preview shows the next native argument list. It uses Windows native
  quoting, not PowerShell command syntax. Logs retain the last 300 lines.
- The GUI uses the current shell's environment, including any `GUFO_*`
  platform-tuning overrides.

To run directly, install `tools/gui/requirements.txt` in a Python environment
and run `python tools/gui/server.py`. The Python code also supports POSIX paths;
Windows is the locally tested launch platform.

Run the focused tests with the GUI environment:

```powershell
.\build\gui-env\Scripts\python.exe -m unittest discover -s tests/tools -p "gui_*_test.py"
```

The tests cover persistence, file selection, argument validation, real child
processes and local HTTP controls. They do not load a model or require a GPU.

The JavaScript DOM tests use the actual Flask-rendered templates and browser
modules. Install their isolated test dependency and run them with Node.js 22+:

```powershell
npm install --prefix build/gui-test-js --no-save --package-lock=false jsdom@26.1.0
node --experimental-vm-modules --test tests/tools/gui_dom_test.mjs
```

Set `GUI_TEST_PYTHON` to a Python with the GUI requirements if the launcher
environment is elsewhere. These tests cover preset switching, sidecar restoration,
file selection, and runtime display; they do not verify browser layout.

## Module boundaries

- `server.py` owns local HTTP controls, security, and the running configuration
  snapshot. `launcher_process.py` and `windows_job.py` own process lifetime.
- `settings.py` owns common defaults and validation; `models.py` owns curated
  family definitions exposed to both backend validation and the browser.
- `presets.py` owns preset operations, schema migration, and atomic JSON writes.
- `command.py` builds native Gufo arguments; `files.py` handles browsing and
  lightweight artifact checks. Neither launches a process.
- Browser modules separate form editing, presets, file picking, and runtime
  display. `app.js` wires them together through direct calls and callbacks.
  `theme.js` remains independent and runs before first paint.

To add another text-model family, define its supported modes, reasoning levels,
and explicit defaults in `models.py`. Existing controls are reused. A genuinely
new option also needs validation, command construction, a small template section,
and focused tests. The family definitions describe UI choices, not detected
runtime capabilities. The GUI continues to manage one `serve llm` process.
