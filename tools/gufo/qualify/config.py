"""qualify.json loading and validation.

Each text model declares the qualification gate next to its bench.json:

    docs/models/<model>/artifacts/qualify.json

The declaration is intentionally explicit: a scope maps changed path prefixes to
the checks and captures that must run. Paths are repository-relative prefixes;
the tool does not infer ownership from file names.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = "gufo-model-qualify/1"
COMPARATORS = ("logit-sha256", "file-sha256", "none")
PLATFORMS = ("windows", "linux")
SCOPES = (
    "arithmetic",
    "loader",
    "tokenizer",
    "prefill",
    "sampling",
    "state",
    "speculation",
    "vision",
    "audio",
    "video",
    "image",
    "api",
)

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_METRIC_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_BUILTINS = {
    "gufo",
    "python",
    "repo",
    "out",
    "exe",
    "docs_model",
    "source",
    "config",
}


class QualifyError(RuntimeError):
    """A user-facing qualification declaration error."""


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = "."
    platforms: tuple[str, ...] = PLATFORMS
    timeout: float | None = None
    optional: bool = False


@dataclass(frozen=True)
class Check:
    id: str
    description: str
    commands: tuple[Command, ...]


@dataclass(frozen=True)
class Capture:
    id: str
    description: str
    baseline: bool
    commands: tuple[Command, ...]
    outputs: tuple[str, ...]
    compare: str


@dataclass(frozen=True)
class Trigger:
    scope: str
    paths: tuple[str, ...]
    requires: tuple[str, ...]


@dataclass(frozen=True)
class Limit:
    capture: str
    metric: str
    maximum: float | None
    minimum: float | None
    source: str


@dataclass
class QualifyConfig:
    path: Path
    root: Path
    data: dict[str, Any]
    checks: dict[str, Check]
    captures: dict[str, Capture]
    triggers: list[Trigger]
    limits: list[Limit]
    profiles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return self.data["model"]["id"]

    @property
    def docs(self) -> str:
        return self.data["model"]["docs"]

    @property
    def source(self) -> str:
        return self.data["model"]["source"]

    @property
    def category(self) -> str:
        return self.data["model"]["category"]

    @property
    def model_dir(self) -> Path:
        return self.root / "docs" / "models" / self.docs

    @property
    def config_path(self) -> Path:
        return self.path

    @property
    def config_rel(self) -> str:
        try:
            return self.path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return self.path.name

    @property
    def files(self) -> dict[str, str]:
        return self.data["files"]

    @property
    def tools(self) -> dict[str, str]:
        return self.data.get("tools", {})

    @property
    def corpus(self) -> dict[str, str]:
        return self.data.get("corpus", {})

    def corpus_path(self, name: str) -> Path:
        try:
            relative = self.corpus[name]
        except KeyError as error:
            raise QualifyError(f"{self.config_rel}: unknown corpus {name!r}") from error
        return (self.model_dir / relative).resolve()

    def required_ids(self) -> tuple[set[str], set[str]]:
        checks: set[str] = set()
        captures: set[str] = set()
        for trigger in self.triggers:
            for item in trigger.requires:
                if item in self.checks:
                    checks.add(item)
                else:
                    captures.add(item)
        return checks, captures

    def unused_ids(self) -> tuple[list[str], list[str]]:
        used_checks, used_captures = self.required_ids()
        return (
            [item for item in self.checks if item not in used_checks],
            [item for item in self.captures if item not in used_captures],
        )

def path_matches(path: str, prefix: str) -> bool:
    """A trigger path is a repository-relative prefix; "re:" opts into a regex."""
    if prefix.startswith("re:"):
        return re.search(prefix[3:], path) is not None
    return path == prefix or path.startswith(prefix)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualifyError(message)


def _require_string(value: Any, where: str) -> str:
    _require(isinstance(value, str) and value.strip() != "", f"{where} must be a non-empty string")
    return value


def _require_mapping(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where} must be an array")
    return value


def _command(raw: Any, where: str) -> Command:
    if isinstance(raw, list):
        raw = {"argv": raw}
    data = _require_mapping(raw, where)
    argv = _require_list(data.get("argv"), f"{where}.argv")
    _require(argv, f"{where}.argv must not be empty")
    for index, part in enumerate(argv):
        _require_string(part, f"{where}.argv[{index}]")
    env = data.get("env", {})
    if env is not None:
        env = _require_mapping(env, f"{where}.env")
        for key, value in env.items():
            _require_string(key, f"{where}.env key")
            _require_string(value, f"{where}.env[{key}]")
    cwd = data.get("cwd", ".")
    _require_string(cwd, f"{where}.cwd")
    _require(not Path(cwd).is_absolute() and ".." not in Path(cwd).parts,
             f"{where}.cwd must stay inside the repository")
    platforms = data.get("platforms", list(PLATFORMS))
    platforms = _require_list(platforms, f"{where}.platforms")
    _require(platforms, f"{where}.platforms must not be empty")
    for platform in platforms:
        _require(platform in PLATFORMS, f"{where}.platforms has unknown value {platform!r}")
    timeout = data.get("timeout")
    if timeout is not None:
        _require(isinstance(timeout, (int, float)) and math.isfinite(timeout) and timeout > 0,
                 f"{where}.timeout must be a positive number")
    optional = data.get("optional", False)
    _require(isinstance(optional, bool), f"{where}.optional must be boolean")
    return Command(
        argv=tuple(argv),
        env=dict(env or {}),
        cwd=cwd,
        platforms=tuple(platforms),
        timeout=float(timeout) if timeout is not None else None,
        optional=optional,
    )


def _item_commands(raw: Any, where: str) -> tuple[Command, ...]:
    commands = _require_list(raw.get("commands"), f"{where}.commands")
    _require(commands, f"{where}.commands must not be empty")
    return tuple(
        _command(command, f"{where}.commands[{index}]")
        for index, command in enumerate(commands)
    )


def _check_command_placeholders(
    command: Command,
    where: str,
    file_roles: set[str],
    corpus_names: set[str],
    tool_names: set[str],
) -> None:
    def inspect(text: str, context: str) -> None:
        for raw in _PLACEHOLDER_RE.findall(text):
            if raw in _BUILTINS:
                continue
            kind, separator, name = raw.partition(":")
            if not separator:
                raise QualifyError(f"{where} {context}: unknown placeholder {{{raw}}}")
            if kind == "files":
                raise QualifyError(
                    f"{where} {context}: {{files:{name}}} must be a complete argv element"
                )
            if kind == "file":
                if name not in file_roles:
                    raise QualifyError(
                        f"{where} {context}: undeclared file role {name!r}"
                    )
            elif kind == "corpus":
                if name not in corpus_names:
                    raise QualifyError(
                        f"{where} {context}: undeclared corpus {name!r}"
                    )
            elif kind == "tool":
                if name not in tool_names:
                    raise QualifyError(
                        f"{where} {context}: undeclared tool {name!r}"
                    )
            elif kind == "env":
                if not name:
                    raise QualifyError(f"{where} {context}: empty env placeholder")
            elif kind == "test":
                if not name:
                    raise QualifyError(f"{where} {context}: empty test placeholder")
            else:
                raise QualifyError(
                    f"{where} {context}: unknown placeholder kind {kind!r}"
                )

    for index, part in enumerate(command.argv):
        whole = re.fullmatch(r"\{files:([^{}]+)\}", part)
        if whole:
            if whole.group(1) not in file_roles:
                raise QualifyError(
                    f"{where}.argv[{index}]: undeclared file role {whole.group(1)!r}"
                )
            continue
        inspect(part, f"argv[{index}]")
    for key, value in command.env.items():
        inspect(value, f"env[{key}]")
    inspect(command.cwd, "cwd")


def _trigger(raw: Any, where: str, item_ids: set[str]) -> Trigger:
    data = _require_mapping(raw, where)
    scope = _require_string(data.get("scope"), f"{where}.scope")
    _require(scope in SCOPES, f"{where}.scope: unknown scope {scope!r}")
    paths = _require_list(data.get("paths"), f"{where}.paths")
    _require(paths, f"{where}.paths must not be empty")
    normalized: list[str] = []
    for index, path in enumerate(paths):
        path = _require_string(path, f"{where}.paths[{index}]")
        if not path.startswith("re:"):
            _require(not Path(path).is_absolute(), f"{where}.paths[{index}] must be relative")
            _require(".." not in Path(path).parts, f"{where}.paths[{index}] must not escape the repository")
        normalized.append(path)
    requires = _require_list(data.get("requires"), f"{where}.requires")
    _require(requires, f"{where}.requires must not be empty")
    for index, item in enumerate(requires):
        item = _require_string(item, f"{where}.requires[{index}]")
        _require(item in item_ids, f"{where}.requires[{index}]: unknown check or capture {item!r}")
    return Trigger(scope=scope, paths=tuple(normalized), requires=tuple(requires))


def _limit(raw: Any, where: str, captures: dict[str, Capture]) -> Limit:
    data = _require_mapping(raw, where)
    capture = _require_string(data.get("capture"), f"{where}.capture")
    _require(capture in captures, f"{where}.capture: unknown capture {capture!r}")
    metric = _require_string(data.get("metric"), f"{where}.metric")
    _require(_METRIC_RE.fullmatch(metric) is not None, f"{where}.metric is not a metric name")
    maximum = data.get("max")
    minimum = data.get("min")
    _require((maximum is None) != (minimum is None), f"{where} needs exactly one of max/min")
    bound = maximum if maximum is not None else minimum
    _require(isinstance(bound, (int, float)) and math.isfinite(bound),
             f"{where}.max/min must be a finite number")
    source = _require_string(data.get("source"), f"{where}.source")
    return Limit(
        capture=capture,
        metric=metric,
        maximum=float(maximum) if maximum is not None else None,
        minimum=float(minimum) if minimum is not None else None,
        source=source,
    )


def load_config(path: Path, root: Path) -> QualifyConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as error:
        raise QualifyError(f"missing qualification config: {path}") from error
    except json.JSONDecodeError as error:
        raise QualifyError(f"invalid JSON in {path}: {error}") from error
    data = _require_mapping(data, str(path))
    _require(data.get("schema") == SCHEMA, f"{path}: expected schema {SCHEMA}")

    model = _require_mapping(data.get("model"), f"{path}: model")
    _require_string(model.get("id"), f"{path}: model.id")
    docs = _require_string(model.get("docs"), f"{path}: model.docs")
    source = _require_string(model.get("source"), f"{path}: model.source")
    _require_string(model.get("category"), f"{path}: model.category")
    _require(
        path.parent.parent.name == docs,
        f"{path}: model.docs {docs!r} must match its docs/models directory",
    )

    files = _require_mapping(data.get("files"), f"{path}: files")
    _require(files, f"{path}: files must declare at least one artifact role")
    for role, description in files.items():
        _require(_ID_RE.fullmatch(role) is not None, f"{path}: invalid file role {role!r}")
        _require_string(description, f"{path}: files[{role}]")

    corpus = data.get("corpus", {})
    corpus = _require_mapping(corpus, f"{path}: corpus")
    for name, relative in corpus.items():
        _require(_ID_RE.fullmatch(name) is not None, f"{path}: invalid corpus name {name!r}")
        relative = _require_string(relative, f"{path}: corpus[{name}]")
        _require(not Path(relative).is_absolute(), f"{path}: corpus[{name}] must be relative")
        resolved = (path.parent.parent / relative).resolve()
        _require(resolved.is_file(), f"{path}: corpus[{name}] does not exist: {resolved}")

    tools = data.get("tools", {})
    tools = _require_mapping(tools, f"{path}: tools")
    for name, description in tools.items():
        _require(_ID_RE.fullmatch(name) is not None, f"{path}: invalid tool name {name!r}")
        _require_string(description, f"{path}: tools[{name}]")

    checks_raw = _require_mapping(data.get("checks", {}), f"{path}: checks")
    captures_raw = _require_mapping(data.get("captures", {}), f"{path}: captures")
    _require(checks_raw or captures_raw, f"{path}: declare at least one check or capture")
    item_ids = set(checks_raw) | set(captures_raw)
    _require(len(item_ids) == len(checks_raw) + len(captures_raw),
             f"{path}: an id is used by both a check and a capture")

    checks: dict[str, Check] = {}
    for item_id, raw in checks_raw.items():
        _require(_ID_RE.fullmatch(item_id) is not None, f"{path}: invalid check id {item_id!r}")
        raw = _require_mapping(raw, f"{path}: checks[{item_id}]")
        description = _require_string(raw.get("description"), f"{path}: checks[{item_id}].description")
        checks[item_id] = Check(
            item_id,
            description,
            _item_commands(raw, f"{path}: checks[{item_id}]"),
        )

    captures: dict[str, Capture] = {}
    for item_id, raw in captures_raw.items():
        _require(_ID_RE.fullmatch(item_id) is not None, f"{path}: invalid capture id {item_id!r}")
        raw = _require_mapping(raw, f"{path}: captures[{item_id}]")
        description = _require_string(raw.get("description"), f"{path}: captures[{item_id}].description")
        baseline = raw.get("baseline", False)
        _require(isinstance(baseline, bool), f"{path}: captures[{item_id}].baseline must be boolean")
        compare = raw.get("compare", "none")
        _require(compare in COMPARATORS, f"{path}: captures[{item_id}].compare has unknown value {compare!r}")
        outputs = _require_list(raw.get("outputs", []), f"{path}: captures[{item_id}].outputs")
        for index, output in enumerate(outputs):
            output = _require_string(output, f"{path}: captures[{item_id}].outputs[{index}]")
            if compare != "none":
                _require(
                    not any(character in output for character in "*?["),
                    f"{path}: captures[{item_id}].outputs[{index}] must be an exact path "
                    "when a comparator is used",
                )
        captures[item_id] = Capture(
            item_id,
            description,
            baseline,
            _item_commands(raw, f"{path}: captures[{item_id}]"),
            tuple(outputs),
            compare,
        )

    triggers_raw = _require_list(data.get("triggers"), f"{path}: triggers")
    _require(triggers_raw, f"{path}: triggers must not be empty")
    triggers = [
        _trigger(raw, f"{path}: triggers[{index}]", item_ids)
        for index, raw in enumerate(triggers_raw)
    ]
    _require(
        any(
            any(prefix.startswith(f"src/models/{source}/") for prefix in trigger.paths)
            for trigger in triggers
        ),
        f"{path}: no trigger owns src/models/{source}/",
    )

    profiles_raw = data.get("profiles", {})
    profiles_raw = _require_mapping(profiles_raw, f"{path}: profiles")
    profiles: dict[str, tuple[str, ...]] = {}
    for name, items in profiles_raw.items():
        _require(_ID_RE.fullmatch(name) is not None, f"{path}: invalid profile name {name!r}")
        items = _require_list(items, f"{path}: profiles[{name}]")
        _require(items, f"{path}: profiles[{name}] must not be empty")
        for index, item in enumerate(items):
            item = _require_string(item, f"{path}: profiles[{name}][{index}]")
            _require(item in item_ids, f"{path}: profiles[{name}][{index}]: unknown item {item!r}")
        profiles[name] = tuple(items)

    limits_raw = data.get("limits", [])
    limits_raw = _require_list(limits_raw, f"{path}: limits")
    limits = [
        _limit(raw, f"{path}: limits[{index}]", captures)
        for index, raw in enumerate(limits_raw)
    ]

    for item_id in sorted(item_ids):
        for index, command in enumerate(
            checks[item_id].commands if item_id in checks else captures[item_id].commands
        ):
            _check_command_placeholders(
                command,
                f"{path}: {item_id}.commands[{index}]",
                set(files),
                set(corpus),
                set(tools),
            )

    config = QualifyConfig(
        path=path.resolve(),
        root=root.resolve(),
        data=data,
        checks=checks,
        captures=captures,
        triggers=triggers,
        limits=limits,
        profiles=profiles,
    )
    unused_checks, unused_captures = config.unused_ids()
    for item in unused_checks:
        config.warnings.append(f"check {item!r} is declared but not required by a trigger")
    for item in unused_captures:
        config.warnings.append(f"capture {item!r} is declared but not required by a trigger")
    return config


def load_configs(root: Path) -> dict[str, QualifyConfig]:
    """Load every docs/models/*/artifacts/qualify.json declaration."""
    root = root.resolve()
    paths = sorted((root / "docs" / "models").glob("*/artifacts/qualify.json"))
    if not paths:
        raise QualifyError(f"no qualify.json declarations under {root / 'docs/models'}")
    configs: dict[str, QualifyConfig] = {}
    for path in paths:
        config = load_config(path, root)
        if config.docs in configs:
            raise QualifyError(f"duplicate qualification config for {config.docs}")
        configs[config.docs] = config
    return configs


def config_for(root: Path, model: str) -> QualifyConfig:
    path = root.resolve() / "docs" / "models" / model / "artifacts" / "qualify.json"
    return load_config(path, root)


def ordered_union(groups: Iterable[Sequence[str]], order: Iterable[str]) -> list[str]:
    """Union item ids while preserving declaration order."""
    wanted = {item for group in groups for item in group}
    return [item for item in order if item in wanted]
