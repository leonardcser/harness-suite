# Run the suite

You need [uv](https://docs.astral.sh/uv/), Python 3.11+, podman and `openssl`.
Agents with an `[install]` run in a pinned image, `harness-suite-NAME:VERSION`.

## Commands

```sh
uv run python -m harness_suite images                     # build the images
uv run python -m harness_suite harnesses                  # recipes and images
uv run python -m harness_suite list                       # checks and levels

uv run python -m harness_suite run --quick --harness pi   # required checks
uv run python -m harness_suite run                        # everything
uv run python -m harness_suite run --harness pi --check tools --api chat
uv run python -m harness_suite run --harness pi --host    # pi from PATH
uv run python -m harness_suite run --command 'pi=node ~/pi/cli.js'

uv run python -m harness_suite report results/RUN --details
uv run python -m harness_suite export results/RUN [results/RERUN] RUN.json
```

## Options

- `--check`: a check ID (`tools.round-trip`) or an area (`tools`).
- `--api`: `chat`, `responses` or `anthropic`.
- `--tools`: `mcp` or `native`.
- `--frontend`: `headless` or `tui`.
- `--jobs`: cases at once.

The exit code is 1 if a check fails or the suite errors. Ctrl-C stops every
agent the run started.

## New agent

1. Run with `--quick`: each required check once, headless, in about a minute. It
   prints the required grid and "Fix these first", in the spec's order.
2. Run everything headless.
3. Add `--frontend tui`.

## Own agent

Write a [recipe](recipes.md), then run it alone:

```sh
uv run python -m harness_suite run --quick --harness path/to/mine.toml
uv run python -m harness_suite run --harness path/to/mine.toml --check tools
```

`tests/fake_harness.py` is a small agent that passes every check. When a check
fails, read how it handles the same case.

## Limits

- A run holds its memory in the systemd user slice `harnesssuite.slice`, capped
  at 75% of the machine. Each agent has its own 4 GB cap. Without a systemd user
  manager, the run has no limits and warns.
- A run starts a case only while agents use under 60% of the cap and CPU
  pressure is under 30%.

## Evidence

`results/RUN/` holds `run.json` (versions, images, hashes) and `results.jsonl`
(a row per check). Each case folder holds:

- `provider/`: every request, header and reply, and `trace.jsonl`
- `fixture-calls.jsonl`: every tool run and MCP message
- `launch-N.*`: how the agent started, its output and exit
- in the TUI: `terminal-N.log`, `launch-N.screen.txt`, `launch-N.shown.txt`
