# Local Gufo launcher

A small browser control panel for one Gufo text server. Select existing model
files, configure a curated set of server and sampler options, save the settings,
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
2. Choose **Speculative decoding**: **Off**, **MTP · Qwen Flash-Next**, or
   **DFlash2 · Qwen3.8-27B**. MTP uses Flash-Next's matching shared Q8_0 sidecar;
   DFlash2 uses a matching Qwen3.8-27B DFlash2 draft. Selecting a file does not
   change the mode. The picker shows local files; Gufo still decides which
   models and tensor formats it supports.
3. Optionally choose a **vision projector**. A blank field means **auto-detect**,
   not disabled vision. An explicitly selected incompatible projector fails
   during model loading.
4. Check the **Gufo executable**. It defaults to this checkout's
   `build/release/gufo.exe`; another built copy can be selected. Leave that
   executable alongside its runtime DLLs and kernel libraries.
5. Adjust settings, click **Save settings**, then **Start Gufo**. Wait for
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
works while the engine is running. Changes apply on the next launch; **Start**
does not overwrite the saved configuration. Reopening the launcher restores
the form without automatically loading a model. Choosing or entering a different
model clears the previous sidecar selections and resets speculative decoding to
Off. Older saved MTP checkbox settings restore as MTP or Off without rewriting
the file until you explicitly save.

The button at the top right of the bar switches between the light paper palette
and a dark navy one; cards, fields, the status pill, metrics and the file picker
all follow it. Until you press it, the launcher follows your operating-system
theme. The choice is remembered in that browser's local storage, not in
`launcher.json`, so it is not part of **Save settings** and never reaches the
engine.

**Stop** releases the launched Gufo process. Closing a browser tab keeps it
running; **Exit launcher** or Ctrl+C in the launcher terminal stops both. On
Windows, Stop terminates the owned process tree and waits for it to exit.
A Windows Job Object also terminates that tree if the launcher crashes or its
terminal is closed abruptly. Failed shutdowns remain visible and can be retried
with Stop; Exit launcher does not report success while shutdown has failed.
Termination does not flush Gufo's optional disk continuation cache. The GUI
does not enable that cache.

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
- More options exposes sessions, penalties and Flash-Next MTP options. MTP
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
