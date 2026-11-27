<p align="center">
  <img src="assets/logo.svg" width="56" alt="" />
</p>

<h1 align="center">harness suite</h1>

<p align="center">
  Black-box tests for coding agents.
  <br />
  <a href="https://leonardcser.github.io/harness-lab/suite/"><strong>See the results</strong></a>
</p>

The suite plays the model provider. It serves a scripted model over the real
wire APIs (OpenAI Chat, OpenAI Responses, Anthropic Messages), gives the agent
one tool, and watches what comes back: the requests it sends and the tool calls
it runs. It never reads the agent's logs or source.

Each check is a rule at one level:

| Level       | The agent...                                              | Result                  |
| ----------- | --------------------------------------------------------- | ----------------------- |
| Required    | must do it: run a tool once, send valid requests          | pass, fail              |
| Recommended | should do it: honor a retry delay, stop on a dead key     | pass, gap               |
| Optional    | may do it, and if it does, it must work: images, effort   | pass, fail, unsupported |
| Info        | does one of several fine things: how it clips long output | a short label           |

Every case comes from the [spec](SPEC.md). Verdicts are repeatable: the suite
scripts every step, so the same agent version gives the same result.

## Quick start

You need [uv](https://docs.astral.sh/uv/) and Python 3.11+. For the pinned
images, also podman.

```sh
uv run python -m harness_suite harnesses                 # the agents it knows
uv run python -m harness_suite run --quick --harness pi  # required checks, in a minute
uv run python -m harness_suite report results/RUN --details
```

## Test your own agent

Write a recipe: one TOML file that tells the suite how to start the agent and
point it at a provider URL. Then run it.

```sh
uv run python -m harness_suite run --quick --harness path/to/mine.toml
uv run python -m harness_suite run --harness path/to/mine.toml
```

## Docs

- [Run the suite](docs/running.md): options, frontends, limits and saved
  evidence
- [Add a harness](docs/recipes.md): the recipe format
- [Reading the results](docs/results.md): gaps, fails, unsupported and not
  measured
- [How it works](docs/internals.md): provider, scenarios and development
- [Spec](SPEC.md): every case the suite checks or plans

## License

[MIT](LICENSE)
