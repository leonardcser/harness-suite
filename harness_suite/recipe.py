"""Harness recipes: data that tells the suite how to launch a harness.

A recipe only connects a harness to the fake provider and the fixture tool.
It must not change the harness's own policies (retry, compaction, limits).
"""

import json
import re
import string
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APIS = ("chat", "responses", "anthropic")
TRANSPORTS = ("stdio", "http", "sse")  # How an agent reaches an MCP server: `[mcp.TRANSPORT]`.
MODELS = ("suite-model", "suite-model-b")
MAX_OUTPUT = 32768


def render(value, variables: dict):
    """Fill ${name} placeholders. A whole-string placeholder keeps its type."""
    if isinstance(value, dict):
        return {render(key, variables): render(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [render(item, variables) for item in value]
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([a-z_]+)\}", value)
    if match and match[1] in variables:
        return variables[match[1]]
    words = {key: str(item) for key, item in variables.items()}
    words |= {key + "_json": json.dumps(item) for key, item in variables.items()}
    return string.Template(value).substitute(words)


def words(values: list) -> list:
    """Command line words: a number, as from a setting's value, becomes text."""
    return [str(v) if isinstance(v, int | float) and not isinstance(v, bool) else v for v in values]


def merge_json(into: dict, extra: dict) -> None:
    """Merge the JSON `extra` into `into`, deeply; on a conflict, `extra`
    wins, lists too."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            merge_json(into[key], value)
        else:
            into[key] = value


@dataclass(frozen=True)
class Prepared:
    """What an agent needs for one launch, before the sandbox runs it."""

    argv: list[str]
    model: str  # The model ID after settings, which may pick their own model.
    environment: dict
    stdin: str | None  # Prompt text for stdin, if the recipe reads it there.
    files: list[tuple[str, str, str]]  # (path as the harness sees it, text, mode); see `file_mode`.


def file_mode(spec: dict) -> str:
    """How a launch writes a recipe file: "replace" it, "append" to it, or
    "merge" its JSON into the file already on disk."""
    if spec.get("merge"):
        return "merge"
    return "append" if spec.get("append") else "replace"


@dataclass(frozen=True)
class Recipe:
    path: Path
    data: dict

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def title(self) -> str:
        """The name people know the agent by."""
        return self.data.get("title", self.name)

    @property
    def about(self) -> dict:
        """The agent as an export shows it: its name, and where its releases
        are."""
        return {"title": self.title, "repo": self.data.get("repo"), "tag_prefix": self.data.get("tag_prefix", "v")}

    @property
    def version(self) -> str:
        return self.data.get("version", "")

    @property
    def install(self) -> dict | None:
        """How `images` installs the harness in its image (see images.py)."""
        return self.data.get("install")

    @property
    def image(self) -> str | None:
        """The pinned image the harness runs in, if the recipe installs it."""
        return f"harness-suite-{self.name}:{self.version}" if self.install else None

    @property
    def apis(self) -> tuple[str, ...]:
        return tuple(api for api in APIS if api in self.data.get("apis", {}))

    @property
    def native(self) -> dict:
        return self.data.get("native", {})

    @property
    def subagent(self) -> list[dict]:
        """The calls that hand a task to a sub-agent and bring back its
        answer: one, or one that starts it and others that wait for it."""
        spec = self.native.get("subagent", [])
        return spec if isinstance(spec, list) else [spec]

    def bootstrap(self, server: str) -> list[dict]:
        """The calls that load an MCP server's deferred tools, for the model."""
        return render(self.data.get("mcp", {}).get("bootstrap", []), {"server": server})

    def route(self, server: str) -> dict | None:
        """How the model reaches an MCP server's tool behind the agent's own tool."""
        spec = self.data.get("mcp", {}).get("route")
        if spec is None:
            return None
        return {**spec, "find": render(spec["find"], {"server": server})} if "find" in spec else spec

    def transport(self, name: str) -> bool | str:
        """True if the recipe connects MCP servers over a transport; else the
        reason it cannot, or False."""
        if name in self.data.get("mcp", {}):
            return True
        return self.data.get("unsupported", {}).get(f"mcp-{name}", False)

    @property
    def instructions(self) -> list[str]:
        """The instruction files the agent reads from the workspace."""
        return self.data.get("instructions", [])

    def supports(self, setting: str) -> bool | str:
        """True if the recipe configures a setting; else the reason it cannot."""
        if setting in self.data.get("settings", {}):
            return True
        return self.data.get("unsupported", {}).get(setting, False)

    def lacks(self, feature: str) -> str | None:
        """Why the agent lacks a feature it has unless its recipe says so,
        such as images: the reason `[unsupported]` documents, or None."""
        return self.data.get("unsupported", {}).get(feature)

    @property
    def tui(self) -> dict | None:
        """How to start the harness's TUI, if the recipe says."""
        return self.data.get("tui")

    def tools(self, source: str) -> bool:
        if source == "mcp":
            return self.transport("stdio") is True
        if source == "native":
            return "shell" in self.native
        return True

    def prepare(
        self,
        variables: dict,
        api: str,
        tools: str,
        settings: dict,
        prompt: str,
        first: bool,
        frontend: str = "headless",
        servers: list[dict] | None = None,
    ) -> Prepared:
        """The launch for `api` with per-launch `settings` like {"effort": "high"}.

        With MCP tools, `[mcp]` applies once, and the template of each server's
        transport, `[mcp.stdio]` for one, once per server with its `${server}`
        name and, over HTTP, its `${server_url}`. JSON files that layers write
        to one path merge.

        In the TUI, `[tui].args` replace the headless `args`, or a setting's
        `tui_args` do, `[tui]` adds its files and environment, and the suite
        pastes the prompt."""
        tui = frontend == "tui"
        data, scope = self.data, self.data["apis"][api]
        variables = dict(variables)
        for key, value in {**data.get("variables", {}), **scope.get("variables", {})}.items():
            variables[key] = render(value, variables)
        # Each layer with its own variables: a setting's ${value}, or a
        # server's ${server}, applies only to its own arguments and files.
        # The top level's `args` are the base arguments, which come first.
        layers: list[tuple[dict, dict]] = [(without_args(data), {}), (scope, {})]
        if tools == "mcp":
            layers.append((data.get("mcp", {}), {}))
            for server in servers or [{"server": "suite", "transport": "stdio"}]:
                layers.append((data["mcp"][server["transport"]], server))
        if tui:
            layers.append((without_args(data["tui"]), {}))
        for name, value in settings.items():
            spec = data.get("settings", {}).get(name)
            if spec is None:
                raise ValueError(f"{self.name} has no setting {name}")
            for key, item in spec.get("variables", {}).items():
                variables[key] = render(item, variables | {"value": value})
            layers.append((spec, {"value": value}))
        # The base arguments: the top level's, or `[tui].args` in the TUI; the
        # last setting with `base_args` (headless) or `tui_args` (TUI) replaces them.
        replaced = "tui_args" if tui else "base_args"
        base = (data["tui"] if tui else data).get("args", []), {}
        base = next(((layer[replaced], extra) for layer, extra in reversed(layers) if replaced in layer), base)
        argv = [*data["command"], *words(render(base[0], variables | base[1]))]
        environment: dict = {}
        files: list = []
        documents: dict[str, dict] = {}  # Path: the JSON that layers merge into it.
        for layer, extra in layers:
            local = variables | extra
            argv += words(render(layer.get("args", []), local))
            environment |= render(layer.get("environment", {}), local)
            for spec in layer.get("files", []):
                if spec.get("once") and not first:
                    continue
                path = render(spec["path"], local)
                if "json" in spec and path in documents:
                    merge_json(documents[path], render(spec["json"], local))
                elif "json" in spec:
                    documents[path] = render(spec["json"], local)
                    files.append((path, documents[path], file_mode(spec)))
                else:
                    files.append((path, render(spec["text"], local), file_mode(spec)))
        files = [(path, json.dumps(t, indent=2) if isinstance(t, dict) else t, mode) for path, t, mode in files]
        mode = "paste" if tui else data.get("input", "file")
        if mode == "argv":
            argv.append(prompt)
        return Prepared(argv, variables["model"], environment, prompt if mode == "stdin" else None, files)


def without_args(layer: dict) -> dict:
    return {key: value for key, value in layer.items() if key != "args"}


def load(name: str) -> Recipe:
    path = Path(name)
    if path.suffix != ".toml":
        path = ROOT / "harnesses" / f"{name}.toml"
    data = tomllib.loads(path.read_text())
    if not re.fullmatch(r"[a-z0-9-]+", data.get("name", "")):
        raise ValueError(f"{path}: name must be lowercase letters, digits, dashes")
    if not data.get("command") or not set(data.get("apis", {})) <= set(APIS):
        raise ValueError(f"{path}: needs a command and apis from {APIS}")
    if "install" in data and not ({"installer", "command"} <= data["install"].keys() and data.get("version")):
        raise ValueError(f"{path}: [install] needs an installer and a command, and the recipe a version")
    return Recipe(path.resolve(), data)


def available() -> list[str]:
    return sorted(path.stem for path in (ROOT / "harnesses").glob("*.toml"))
