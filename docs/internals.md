# How it works

## Provider

`provider.py` serves OpenAI Chat, OpenAI Responses and Anthropic Messages
(`apis/`).

- Parses each request into one `Request` (`messages.py`).
- Asks the scripted model (`model.py`) for a `Reply` and writes it in the API's
  format, with real error bodies and headers.
- Injects stream faults: a cut, an error event, a bad event, odd SSE framing.
- Rejects what the real API rejects, which fails `request.valid`.
- Keeps state as the real APIs do: thinking signatures, encrypted reasoning,
  stored Responses items.

The model tells task requests from side requests (titles, summaries) and
sub-agent requests, and follows a script for each. It counts 4 bytes as one
token and enforces the context window.

## Fixture

`fixture.py` is the one tool, as an MCP server or a shell command. Every call is
logged in `fixture-calls.jsonl`. It can run slowly, flood output, print invalid
bytes, ask on the terminal, leave a child behind, or kill itself like the
out-of-memory killer. As an MCP server, `fixture.PROFILE` changes the protocol
version, startup, pagination, tool list and client requests.

## Scenarios

A scenario is one class in `scenarios/AREA.py`: a script and its checks.

```python
class Session(Scenario):
    id = "cache.tool-loop"
    title = "Each request in a tool loop extends the one before"
    tools = True

    def script(self):
        return [use("text"), use("text"), use("text"), say("SUITE_FIRST_DONE"), use("text"), DONE]

    @check(RECOMMENDED)
    def tool_loop(self, run): ...
```

- `script()` lists the model's steps (`script.py`): call the tool, send text,
  fail, or stream with a fault.
- `launches()` starts the agent more than once to continue, switch or kill it.
- `task()` and `aside()` answer main and side requests.
- A `@check` reads only the evidence in `run` (`evidence.py`) and returns a
  result its level allows (`results.py`).

## Drivers

`drivers.py` runs the launches. Headless is one process each. The TUI runs in a
120x40 tmux pane (`terminal.py`) and checks read every line the screen showed.
`sandbox.py` starts agents on the host or in their container. `live.py` stops
what is left on an error or Ctrl-C.

## Develop

```sh
uv run python -m unittest discover -s tests
uv run ruff check . && uv run ruff format --check .
```

- `tests/fake_harness.py` is a small agent with flags that add one defect each.
  `tests/test_checks.py` runs every scenario against it, so each check passes a
  good agent and catches a broken one.
- One test runs every case on the clean fake from its recipe alone, so a new
  agent needs only its TOML file.
- A new check follows
  [What makes a good case](../SPEC.md#what-makes-a-good-case).
