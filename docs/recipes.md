# Add a harness

A harness is one TOML file in `harnesses/`. A recipe only connects the agent to
the suite. It must not change the agent's own behavior or trust the workspace.

## Format

```toml
name = "example"
title = "Example"                            # the name by default
repo = "https://github.com/example/example"  # links each version to its release
# tag_prefix = "v"                           # before the version in a release tag
version = "1.2.0"
command = ["example"]
instructions = ["AGENTS.md"]                 # the workspace files it reads
input = "stdin"                              # or "argv", or read ${prompt_file}
args = ["--print", "--model", "${model}"]

[install]                                    # the image; without it, the harness runs on the host
installer = "https://example.com/install.sh" # frozen in installers/example.sh
command = "curl -fsSL ${installers}/example.sh | sh -s -- ${version}"

[apis.chat.variables]
base = "${provider_url}/v1"

[environment]
EXAMPLE_API_KEY = "suite-placeholder"

[[files]]                                    # config written before launch
path = "${home}/.example/config.json"
json = { base_url = "${base}", context_window = "${context_window}" }

[mcp]                                        # added once when tools come from MCP
args = ["--mcp-config", "${home}/.example/mcp.json"]
[[mcp.stdio.files]]                          # added for each server over stdio
path = "${home}/.example/mcp.json"
json = { servers = { "${server}" = { command = "${python}", args = ["${fixture}", "mcp", "${fixture_dir}", "${server}"] } } }
[[mcp.http.files]]                           # each over Streamable HTTP; [mcp.sse]: the older SSE
path = "${home}/.example/mcp.json"
json = { servers = { "${server}" = { url = "${server_url}" } } }

[native.shell]                               # the harness's own shell tool
name = "bash"
arguments = { command = "${command}" }
# [native.read], [native.write], [native.edit]: its file tools, with ${path},
# ${content}, ${old} and ${new}
[native.subagent]                            # its sub-agent tool, with ${prompt}
name = "task"
arguments = { prompt = "${prompt}" }
# Or [[native.subagent]] calls: one that starts it, then ones that wait; a
# later call takes ${found}, what its `find` pattern matched in a result.

[tui]                                        # the interactive UI, in tmux
args = ["--model", "${model}"]               # replace the headless args
# ready = "suite-model"                       # text the screen shows once the TUI can take the prompt
# quit = "/quit"                              # the default

[settings.continue]                          # what session checks may use
args = ["--continue"]
[settings.effort]
args = ["--effort", "${value}"]
[settings.read-only]                         # a mode that changes no file
base_args = ["--print", "--plan", "--model", "${model}"]   # replace the headless args
[settings.normal-mode]                       # its default permissions, for workspace checks
base_args = ["--print", "--model", "${model}"]   # without the flag that skips every prompt
[settings.json]                              # its machine-readable output, for cli.json
args = ["--format", "json"]
[[settings.untrusted.files]]                 # undo the base files' trust, for workspace checks
path = "${home}/.example/trust.json"
json = { trusted = [] }
[[settings.workspace-mcp.files]]             # a project MCP config, as a cloned repository holds it
path = "${workspace}/.example/mcp.json"
json = { servers = { "${value}" = { command = "${python}", args = ["${fixture}", "mcp", "${fixture_dir}", "${value}"] } } }
[[settings.workspace-trust.files]]           # the user trusts the workspace
path = "${home}/.example/trust.json"
merge = true                                 # merge into the file the harness keeps
json = { trusted = ["${workspace}"] }

[unsupported]                                # settings, tools or features it lacks, with the source
effort = "example has no effort setting"
images = "example sends only text to the model"
```

## Placeholders

| Placeholder                           | Value                          |
| ------------------------------------- | ------------------------------ |
| `home`, `workspace`, `prompt_file`    | Paths of the run               |
| `provider_url`, `model`, `model_b`    | The scripted provider          |
| `context_window`, `max_output_tokens` | Limits of the scenario         |
| `python`, `fixture`, `fixture_dir`    | The fixture tool               |
| `free_port`                           | A free host port               |
| `server`, `server_url`                | An MCP server, in its template |

A placeholder that fills a whole string keeps its type. `${name_json}` gives the
JSON form.

## Files

- `once`: write on the first launch only.
- `append`: add to the file.
- `merge`: merge into a JSON file the agent already wrote.

JSON files from several layers merge, so each MCP server adds its own entry.

## MCP

- A transport without a template is unsupported. Say so in `[unsupported]` as
  `mcp-http` or `mcp-sse`.
- If the agent defers MCP tools, `[[mcp.bootstrap]]` lists the calls that load
  them.
- If it hides them behind its own tool, `[mcp.route]` says how to reach a server
  through it.

See `claude-code.toml`, `omp.toml` and `fx.toml`.

## Image

Give the recipe an `[install]` and run `python -m harness_suite images NAME`. A
new version freezes its installer in `installers/`. Commit it.
