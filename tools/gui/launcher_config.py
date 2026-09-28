"""Curated Gufo launch settings, local files, and atomic persistence."""

import json
import math
import os
from pathlib import Path
import re
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
PATH_FIELDS = ("executable", "models_dir", "model", "mtp_model", "mmproj")
DEFAULTS = {
    "executable": str(ROOT / "build" / "release" / ("gufo.exe" if os.name == "nt" else "gufo")),
    "models_dir": str(Path.home()),
    "model": "",
    "mtp_model": "",
    "mmproj": "",  # Empty means engine auto-discovery, not disabled vision.
    "context": 32768,
    "max_tokens": 2048,
    "think": "on",
    "reasoning_effort": "auto",
    "port": 8080,
    "served_model_name": "gufo",
    "sessions": 1,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "seed": -1,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "mtp": False,
    "draft_tokens": 7,
    "mtp_policy": "length",
    "mtp_draft_vocab": "full",
    "prompt_lookup": False,
}
INTEGER_LIMITS = {
    "context": (0, 2**32 - 1), "max_tokens": (-1, 2**32 - 1),
    "port": (1, 65535), "sessions": (1, 64), "top_k": (0, 2**31 - 1),
    "seed": (-1, 2**53 - 1), "repeat_last_n": (0, 2**32 - 1),
    "draft_tokens": (1, 7),
}
FLOAT_LIMITS = {
    "temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
    "repeat_penalty": (0, 100), "frequency_penalty": (-2, 2),
    "presence_penalty": (-2, 2),
}
ENUMS = {"think": ("auto", "on", "off"), "reasoning_effort": ("auto", "low", "medium", "xhigh"),
         "mtp_policy": ("length", "survival"),
         "mtp_draft_vocab": ("full", "latin")}
CLI_FIELDS = (
    "context", "max_tokens", "think", "served_model_name", "temperature",
    "top_k", "top_p", "min_p", "seed", "repeat_penalty", "repeat_last_n",
    "frequency_penalty", "presence_penalty",
)


def settings_path():
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".config"))
    return base / "Gufo" / "launcher.json"


def validate_settings(values):
    if not isinstance(values, dict) or set(values) - DEFAULTS.keys():
        raise ValueError("Settings must contain only the supported launcher fields.")
    result = DEFAULTS | values
    for key, default in DEFAULTS.items():
        value = result[key]
        if isinstance(default, str):
            if not isinstance(value, str) or any(ord(c) < 32 for c in value):
                raise ValueError(f"{key}: enter a single line of text.")
            result[key] = value = value.strip()
        elif isinstance(default, bool) and type(value) is not bool:
            raise ValueError(f"{key}: expected a checkbox value.")
        if key in INTEGER_LIMITS:
            low, high = INTEGER_LIMITS[key]
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key}: enter a whole number from {low} to {high}.")
        if key in FLOAT_LIMITS:
            low, high = FLOAT_LIMITS[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key}: enter a number from {low} to {high}.")
        if key in ENUMS and value not in ENUMS[key]:
            raise ValueError(f"{key}: choose one of {', '.join(ENUMS[key])}.")
        if key in PATH_FIELDS and value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise ValueError(f"{key}: use a full filesystem path.")
            result[key] = str(path)
    if result["max_tokens"] == 0:
        raise ValueError("max_tokens: use -1 for unlimited output, or a positive limit.")
    if result["repeat_penalty"] == 0:
        raise ValueError("repeat_penalty: must be greater than zero; 1 disables it.")
    if result["top_p"] == 0:
        raise ValueError("top_p: must be greater than zero; 1 disables it.")
    if not result["served_model_name"]:
        raise ValueError("served_model_name: enter the model name your API client will use.")
    return result


def load_settings(path):
    if not path.exists():
        return dict(DEFAULTS)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("version") != 1:
            raise ValueError("Unsupported launcher settings version.")
        return validate_settings(document["settings"])
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Cannot read saved settings at {path}: {exc}") from exc


def save_settings(path, values):
    settings = validate_settings(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="launcher-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump({"version": 1, "settings": settings}, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return settings


def check_gguf(value, label, *, sharded=False):
    path = Path(value)
    if not value or not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError(f"{label}: select an existing GGUF file.")
    with path.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError(f"{label}: the file does not have a GGUF header.")
    match = SHARD.match(path.name)
    if match and sharded:
        prefix, index, count = match.groups()
        if int(index) != 1 or not 1 <= int(count) <= 99999:
            raise ValueError(f"{label}: select the first shard (-00001-of-...).")
        for number in range(1, int(count) + 1):
            sibling = path.with_name(f"{prefix}-{number:05d}-of-{count}{path.suffix}")
            if not sibling.is_file():
                raise ValueError(f"{label}: missing shard {sibling.name}.")


def build_command(values, *, check_files=False):
    settings = validate_settings(values)
    if check_files:
        if not settings["executable"] or not Path(settings["executable"]).is_file():
            raise ValueError("Select gufo.exe from a built Gufo directory, with its runtime DLLs.")
        check_gguf(settings["model"], "Model", sharded=True)
        if settings["mmproj"]:
            check_gguf(settings["mmproj"], "Vision projector")
        if settings["mtp"]:
            check_gguf(settings["mtp_model"], "MTP sidecar")
    command = [settings["executable"], "serve", "--host", "127.0.0.1",
               "--port", str(settings["port"]), "--sessions", str(settings["sessions"]),
               "llm", "--model", settings["model"]]
    for key in CLI_FIELDS:
        command.extend(["--" + key.replace("_", "-"), str(settings[key])])
    if settings["think"] != "off" and settings["reasoning_effort"] != "auto":
        command.extend(["--reasoning-effort", settings["reasoning_effort"]])
    if settings["mmproj"]:
        command.extend(["--mmproj", settings["mmproj"]])
    command.extend(["--speculative", "mtp" if settings["mtp"] else "off"])
    if settings["mtp"]:
        for key in ("mtp_model", "draft_tokens", "mtp_policy", "mtp_draft_vocab"):
            command.extend(["--" + key.replace("_", "-"), str(settings[key])])
        if settings["prompt_lookup"]:
            command.append("--prompt-lookup")
    return command


def browse_directory(value, kind="gguf"):
    if kind not in ("gguf", "executable", "folder"):
        raise ValueError("Unknown file picker type.")
    path = Path(value or Path.home()).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ValueError("Choose an existing folder using its full path.")
    path = path.resolve()
    entries = []
    with os.scandir(path) as children:
        for child in children:
            if len(entries) >= 2000:
                raise ValueError("This folder has too many entries. Enter a more specific folder.")
            try:
                is_dir = child.is_dir()
                shard = SHARD.match(child.name)
                if not is_dir:
                    if kind == "folder" or not child.is_file():
                        continue
                    if kind == "gguf" and (not child.name.lower().endswith(".gguf") or
                                           (shard and int(shard[2]) != 1)):
                        continue
                    if kind == "executable" and not (child.name.lower().endswith(".exe") or child.name == "gufo"):
                        continue
                entries.append({"name": child.name, "path": str(path / child.name),
                                "directory": is_dir, "shards": int(shard[3]) if shard else 1})
            except OSError:
                continue  # A deleted or inaccessible entry need not hide the folder.
    entries.sort(key=lambda entry: (not entry["directory"], entry["name"].casefold()))
    roots = [str(Path.home())]
    if os.name == "nt":
        roots.extend(f"{letter}:\\" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if Path(f"{letter}:\\").is_dir())
    else:
        roots.append("/")
    return {"path": str(path), "parent": str(path.parent), "roots": roots, "entries": entries}
