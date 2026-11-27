# Harness Suite spec

What a coding agent must, should and may do when it talks to a model provider,
runs tools and keeps sessions, as far as a black-box test can see. The suite
plays the provider and the tool, so every case is something it can provoke and
observe from outside.

### Levels

- **required**: the agent must do it. Pass or fail.
- **recommended**: the agent should do it. Pass or gap.
- **optional**: the agent may do it, and if it does, it must work. Pass, fail
  or not supported.
- **info**: there is no single right answer. A label shows what the agent does.

Two more outcomes are not levels:

- **not supported**: the agent says so itself, with a recipe that lacks the
  setting, an `[unsupported]` reason, or an undeclared protocol capability. A
  feature it claims that does not work fails.
- **not measured**: the run never reached what the case judges.

A case that cannot apply, such as an Anthropic stream shape on Chat, is not
run.

### Frontends

Each area names one frontend, and its cases run only there. The subject of a
case decides it, never what agents support.

- **headless**: the agent loop and the command line. Most areas.
- **TUI**: what only a person at the keyboard does, such as a paste.
- **headless and TUI**: a user action each frontend does its own way, such as
  continuing a session.

### Marks

`[x]` the suite checks it, with the check ID. `[ ]` planned. `[-]` not
measurable as a black box, with the reason. A test keeps the `[x]` lines and
the checks in step.

### What makes a good case

A case holds for any agent, however it is built or words things. A recipe sets
the agent up as a user would: plain output, a JSON mode only through its `json`
setting, and no workspace trusted in advance.

- **Known in advance.** The behavior to provoke and the assertion come before
  the code.
- **Structural evidence only.** Requests on the wire, fixture tool runs, MCP
  messages, files, exit codes, signals. Text counts only as an exact marker the
  suite planted: no word lists or regexes, and nothing that a rewording of the
  agent's messages would change.
- **Proof that it happened.** A case first checks that what it provokes
  reached the agent. If not, it is not measured. An agent that does nothing
  passes nothing.
- **Real provider behavior.** The suite sends only what a real API or server
  sends. A quirk of one deployment is recommended at most.
- **Order from the suite, not the clock.** The suite holds each step, so the
  same agent gives the same verdict on every run. Work at the same time is out
  of scope, and so is what the agent did in the background while the suite
  waited.
- **Generous timing.** A wait is a bound, never a ratio or a shape, wide enough
  for a correct agent under load.
- **One cause, one verdict.** A case that provokes a rejected request names the
  kinds of rule in `provokes`, and that request decides only its own checks.
  `request.valid` judges every other rule.
- **A control for every "no".** A case that passes when nothing happens also
  runs the setup where it must happen. Without that, it is not measured.
- **A control for every recipe action.** A setting the recipe applies, such as
  trust, is proven to take effect in the same case.
- **One baseline.** Every tool case builds on `tools.round-trip` `text`. Where
  it does not pass, the other tool cases of that agent, frontend, API and tool
  source are not measured.
- **Stable, or out.** A check whose verdict differs between runs is fixed or
  removed.
- **Levels.** Required and recommended for harm the suite can see: a hang, a
  lost result, a rejected request, data loss, code run without consent.
  Optional for a feature that must work if present. Info only for an impactful
  difference, with a label from structural evidence.

## 1. Requests

What the agent sends, before anything goes wrong.

Frontend: headless.

### 1.1 Content

- [x] `request.valid` Every request is one the real API accepts. Every case
  feeds it, except for the rules a case provokes on purpose. The suite's
  provider keeps the real APIs' rules:
  tool call and result pairs, reasoning signatures, stored Responses items
  only when `store` is not false, text blocks, cache marks, image limits.
- [x] `request.model` The configured model ID reaches the provider exactly.
- [x] `request.input` The user's prompt reaches the provider unchanged.

### 1.2 Connection

- [x] `request.auth` Every request carries the configured API key.
- [x] `request.key-in-body` The API key is never in a request's body.
- [x] `request.base-path` Every request keeps the base URL's path, as
  gateways use.
- [x] `request.proxy` The agent reaches the provider through the proxy its
  environment names.
- [x] `request.tls` The agent reaches an HTTPS provider whose certificate a
  company CA signed, with the CA file the standard environment variables name.
- [x] `request.tls-untrusted` A certificate no trusted CA signed is refused,
  before any request.

## 2. Streaming

How the agent reads a streamed reply.

Frontend: headless.

### 2.1 Framing

- [x] `stream.framing` Valid SSE variations are parsed: comments, CRLF, no
  space after `data:`, bytes split inside characters, unknown event types
  and fields from newer API versions, gzip, and one event of 300 KB.
- [x] `stream.azure-filter` Azure's first chunk with no choices, only its
  content filter results, is parsed (Chat).
- [x] `stream.interleaved-calls` Parallel calls whose argument fragments
  alternate are put together per call (Chat and Responses).
- [x] `stream.call-chunks` Tool call chunks in the shapes compatible servers
  send: whole arguments, repeated or blank IDs and names, and calls that end
  with `finish_reason: "stop"`.

### 2.2 Broken and odd streams

- [x] `stream.ended-early` A reply that ends early, without its finish
  events, at the output limit, or with a stop reason that says it failed, is
  not taken as a finished turn: the agent asks again or exits non-zero.
- [x] `stream.empty-reply` What the agent does with a complete reply that
  holds no text and no call: asks again, or ends the turn, and with what exit
  status.
- [x] `stream.slow` A slow but steady stream, over a minute long, is read to
  its end: the proof is the next request, which carries the call that came
  last in the stream.

## 3. Errors and retries

What the agent does when the provider fails.

Frontend: headless.

### 3.1 Transient errors

- [x] `errors.retry` One transient provider error is retried, including a
  proxy's HTML error page and a reset connection.
- [x] `errors.retry-after` The agent waits as long as a `retry-after` or
  `retry-after-ms` header asks, or stops when the wait is an hour. A wait only
  in the error message is not measured.
- [x] `errors.backoff` The agent waits between attempts when errors do not
  stop: six tries with every wait under a second is a gap; an agent that
  stops trying does not hammer.

### 3.2 Errors that stay

- [x] `errors.final` The agent does not resend the same request on an error
  that will not go away, including a 404 for an unknown model and a 413 for a
  request too large.

## 4. Prompt cache

Whether each request extends the one before, so the provider can serve its
prefix from cache.

Frontend: headless.

### 4.1 Prefix

- [x] `cache.tool-loop` Each request in a tool loop extends the one before.
- [x] `cache.turns` A new turn extends the conversation before it, also
  after a restart: each turn is a new process that resumes the session.
- [-] A system prompt that holds the date: it changes only across days.

### 4.2 Marks and keys

- [x] `cache.marked` The request asks the provider to cache the prompt
  (Anthropic).
- [x] `cache.key` An OpenAI prompt cache key, if sent, stays the same through
  a session, also after a restart.

## 5. Reasoning

How the model's reasoning goes back to it. Anthropic signs it and Responses
encrypts it, and both reject a request that changes it.

Frontend: headless.

### 5.1 In a turn

- [x] `reasoning.tool-loop` Reasoning goes back unchanged while the tools run.
- [x] `reasoning.late-encrypted` Reasoning goes back unchanged when Responses
  sends the encrypted content only in the final response, as some Azure
  deployments do.
- [x] `reasoning.cut` A reply cut in its reasoning by a stream error leaves a
  history the API accepts, and the turn finishes.

### 5.2 Across turns

- [x] `reasoning.across-apis` Where earlier reasoning goes after a switch to another API.

## 6. Tool loop

How tool calls and their results travel between the model and any tool.

Frontend: headless.

### 6.1 Delivery

- [x] `tools.round-trip` A streamed tool call gets its result back, also after
  text of only new lines.
- [x] `tools.parallel` Several calls in one reply all get results, up to twenty,
  also when they finish out of order.
- [x] `tools.sequence` Results stay matched through a long tool loop.
- [x] `tools.empty-result` A tool result with nothing in it, which the API must
  still accept.
- [x] `tools.arguments` Tool arguments arrive exactly as streamed.
- [x] `tools.output-size` How much of a large tool result reaches the model.
- [x] `tools.repeat` What the agent does when the model sends the same call
  many times in a row: it runs each, or stops the loop.

### 6.2 Broken and odd calls

- [x] `tools.incomplete` A broken tool call never runs, also next to a valid
  one in the same reply.
- [x] `tools.unknown-name` A call to a tool that does not exist gets an error
  result, also when its name is too long for the API.
- [x] `tools.bad-arguments` What happens to a call whose arguments break the
  tool's schema, or are an empty string instead of `{}`.
- [x] `tools.reused-id` The model reuses a call ID from an earlier turn.
- [x] `tools.call-ids` Results come back under call IDs of any shape: long,
  or with punctuation.
- [x] `tools.unicode` Text the agent cuts or passes on stays valid Unicode:
  no request holds half of a surrogate pair.

### 6.3 Side effects and failures

- [x] `tools.effect-once` A finished tool call does not run again after a
  provider error.
- [x] `tools.killed` A tool killed by the system gives a result, and the agent
  goes on.

### 6.4 Sub-agents

- [x] `tools.subagent` A sub-agent's answer comes back to the main loop; the
  pass says what the sub-agent starts from.
- [x] `tools.subagent-error` A sub-agent whose requests fail does not stop the main
  loop, and the provider's error reaches it.

## 7. Shell

The agent's own shell tool, with commands that behave as real ones do.

Frontend: headless.

### 7.1 Output

- [x] `shell.output` A failing command's output and errors both come back.
- [x] `shell.exit-status` The model sees a failed command's exit status, as
  its number or as the result's error flag.
- [x] `shell.binary` Output that is not UTF-8 comes back.

### 7.2 Processes

- [x] `shell.stdin` A command that reads its input does not hang.
- [x] `shell.background` A command that leaves a background process returns at once.
- [x] `shell.timeout` A command that never ends gets a result, and its child
  is not left orphaned.

### 7.3 Large output and input

- [x] `shell.flood` A command that prints a million lines; the pass says how
  much reaches the model.
- [x] `shell.flood-status` A failing command with a flood of output still
  shows its last line in the result. Keeping the rest behind a path or handle
  is a gap: the suite cannot see it without reading the agent's words.
- [x] `shell.long-command` A command line of 200 KB, over the kernel's limit
  for one argument, runs whole or comes back as a result. The fixture logs
  input that arrives cut short.

### 7.4 Environment

- [x] `shell.cwd` Commands run in the workspace.
- [x] `shell.cd` Whether a `cd` carries over to the next command.

## 8. Files

The agent's own file tools, with files that real workspaces have.

Frontend: headless.

### 8.1 Reading

- [x] `files.read` Reading an unusual file leaves the agent working: binary,
  5 MB, one 1 MB line, Latin-1, missing, a directory, a FIFO, a symlink loop,
  a link to `/dev/zero`, or an awkward path. The pass says what the model
  sees of the file: whole, part, or none.
- [x] `files.read-changing` Reading a log that grows during the read gives
  a result and the task goes on. The pass says whether the model sees a state
  the file had.
- [-] A file rewritten in place during the read: which version the agent
  sees depends on where the writer is when it reads.

### 8.2 Writing and editing

- [x] `files.write` A write keeps the bytes it was given: CRLF line ends, a
  UTF-8 BOM, no final new line, tabs and Unicode, in new folders, or at a
  path with spaces, quotes and `$HOME` in it, and 272 KB in one call.
- [x] `files.edit` An edit changes exactly the text it names. Old text that
  occurs twice or not at all leaves the file unchanged, with a result. CRLF
  line ends, a byte order mark and tab indents are found and kept.
- [x] `files.edit-stale` An edit keeps a change made on disk after the read,
  and gets a result.
- [x] `files.write-guarded` What a write does to a read-only file, as the
  file on disk shows.

## 9. MCP

The protocol between the agent and a tool server. The suite's server takes a
profile per case, so it can use each part of the protocol.

Frontend: headless.

### 9.1 Lifecycle

- [x] `mcp.handshake` The agent opens the MCP session as the protocol requires.
- [x] `mcp.server-env` Whether a stdio server gets the provider API key in
  its environment.
- [x] `mcp.version` A server on an older protocol version still works.
- [x] `mcp.startup` A server that starts slowly or logs a lot still works.
- [x] `mcp.mute` A server that never answers does not block the agent.
- [x] `mcp.restart` A server that exits between calls: the agent restarts it or
  reports it.

### 9.2 Tools

- [x] `mcp.pagination` Tools on a later page of `tools/list` are found.
- [x] `mcp.many-tools` A server with 150 tools still works, in requests the API
  takes (Chat Completions takes at most 128 tools); the detail shows how many
  tools reach the model.
- [x] `mcp.list-changed` The agent reloads the tools when the server says
  they changed, unless its recipe says it lacks the feature. The model calls
  the tool again only once the agent has listed the tools again, or after
  10 s.
- [x] `mcp.unusable-tool` A server tool the model APIs cannot take, by its
  name or its schema, does not break requests. A schema that breaks the MCP
  spec may get the whole server refused, once the agent has listed it.
- [x] `mcp.schema` A tool with a rich JSON schema still works; the pass says
  how the schema reaches the model.
- [x] `mcp.same-names` Two MCP servers with the same tool names.
- [x] `mcp.native-names` An MCP tool with the name of one of the agent's own
  tools: no name is offered twice, and the agent's own tool still runs.
- [x] `mcp.instructions` Where the server's instructions reach the model.

### 9.3 Results

- [x] `mcp.protocol-error` A call the server answers with a protocol error gets
  an error result.
- [x] `mcp.structured` What reaches the model from a structured tool result.

### 9.4 Requests from the server

- [x] `mcp.ping` The agent answers the server's ping.
- [x] `mcp.roots` The agent tells the server its workspace.
- [x] `mcp.elicitation` The agent answers the server's question to the user.
- [x] `mcp.sampling` The agent serves the server's request for a completion.
- [x] `mcp.notifications` Log and progress notifications from the server do
  not break a call.

### 9.5 Transports

Most cases use one server over stdio. A recipe says how the agent connects a
server over each transport it supports.

- [x] `mcp.framing` Over stdio, a message that comes in many pieces, or a
  line on stdout that is not JSON, does not break the server.
- [x] `mcp.transport` The agent reaches an MCP server over HTTP: Streamable
  HTTP, and the older HTTP with SSE.
- [x] `mcp.session-expired` The agent opens a new MCP session when the server
  forgot the old one.
- [-] OAuth for remote servers: the login needs a person in a browser.

## 10. Media

Images and other content in tool results.

Frontend: headless.

### 10.1 Images

- [x] `media.image` An image from an MCP tool reaches the model as an image.
- [x] `media.read-image` The agent's own read tool sends an image file as an
  image, with the media type of its bytes, also when its name says otherwise.

### 10.2 Other MCP content

- [x] `media.content` Binary MCP content, such as a PDF or audio, never
  reaches the model as raw base64.

### 10.3 Limits

- [x] `media.limits` An image the API does not take does not break the
  session. Measured only when a normal image before it reached the model.
- [x] `media.many-images` A conversation with many images stays within the
  API's limits on a request with many images.

## 11. Context

What the agent does when the conversation outgrows the window.

Frontend: headless.

### 11.1 Compaction

- [x] `context.overflow` A conversation that outgrows the context window still
  finishes, also when the tool results are images, when each step reasons,
  and when the context error comes in the words of vLLM, llama.cpp or
  OpenRouter.
- [x] `context.resume-compacted` A continued session starts from its
  compacted context, not the whole history before it. Its first request
  carries the old session, the user's fact or the summary.
- [x] `context.keeps-fact` The user's instruction survives compaction.
- [x] `context.keeps-instructions` The workspace's instructions file survives compaction.
- [-] What the agent asks the model for a summary, and what it does with the
  reply: the suite would have to read the agent's own words to find the
  summary request.

### 11.2 Limits

- [x] `context.big-prompt` A user prompt larger than the context window.

### 11.3 Instructions

- [x] `context.instructions-nested` An instructions file in a subfolder.
- [x] `context.instructions-large` A large instructions file.

## 12. Sessions

A session across launches of the agent, through the recipe's setting that
continues it.

Frontend: headless.

- [x] `session.switch-effort` Continuing with another effort level keeps the history.
- [x] `session.switch-api` Continuing on another wire API converts the history.
- [x] `session.crash` A session resumes after the agent is killed in the
  middle of a turn. What it keeps of that turn depends on when the agent
  saves, so the case does not judge or label it.
- [x] `session.long` A long tool loop finishes.
- [-] A corrupt session file: where agents keep sessions is not part of the
  black box.
- [-] Whether the agent slows down or repeats content as a session grows: a
  ratio of times or sizes depends on the machine and on what the agent adds
  on purpose.

## 13. Settings

What the agent's own options change.

Frontend: headless.

### 13.1 The request

- [x] `settings.effort` The reasoning effort setting changes the request.
- [x] `settings.temperature` A temperature from the configuration reaches the request.
- [x] `settings.system-prompt` A system prompt from the configuration reaches every request.
- [x] `settings.headers` A custom header from the configuration reaches every request.

### 13.2 Modes and files

- [x] `settings.read-only` A read-only mode changes no file, through the write tool or the shell.
- [x] `settings.broken-file` A broken configuration file does not pass as a
  good one: the agent stops with a status that is not 0, or goes on with the
  task.

## 14. Workspace trust

What the agent runs of a workspace on its own, before the model asks for
anything. A cloned repository is untrusted input: the agent runs in its normal
permission mode, in a workspace the user has not trusted (the recipe's
`untrusted` setting undoes the trust its base files give).

Frontend: headless.

- [x] `workspace.repo-code` Opening a repository runs none of its code, such
  as a `core.fsmonitor` command in its `.git/config`. The suite first checks
  that git in the sandbox runs it.
- [x] `workspace.mcp-config` A server in the workspace's own MCP
  configuration starts only after the user trusts the workspace: no process of
  the server runs, even one that exits before it speaks. Control: with
  the recipe's `workspace-trust` setting, the server must start, or the case
  is not measured.

## 15. Command line

The contract of headless mode, which scripts and CI jobs rely on.

Frontend: headless.

### 15.1 Exit

- [x] `cli.exit-status` The exit status tells a script whether the task
  worked, also after a retry got past an error, or after a tool already ran.

### 15.2 Input

- [x] `cli.usage-error` An unknown option fails at once, with no request and
  a status that is not 0. Control: a normal launch first finishes its task
  and exits 0.
- [x] `cli.empty-prompt` An empty prompt, or one of only whitespace, fails at
  once, with no request and a status that is not 0. Control: as for
  `cli.usage-error`.
- [x] `cli.open-stdin` Whether, with the prompt in the arguments, the agent
  waits for a stdin that stays open and sends nothing. Control: as for
  `cli.usage-error`.

### 15.3 Output

- [x] `cli.json` The recipe's JSON output mode writes only valid JSON, and a
  failed run still exits with a status that is not 0.
- [x] `cli.answer` The final answer reaches stdout whole.
- [x] `cli.no-escapes` Output to a pipe holds no terminal escape codes.

## 16. User actions

What the user does to a session, in each frontend's own way: headless with
the recipe's options, in the TUI with its prompts.

Frontend: headless and TUI.

### 16.1 Continue

- [x] `session.continue` A continued session sends the earlier turn:
  headless, a new launch that continues it; in the TUI, the next prompt.
- [x] `reasoning.history` Where earlier reasoning goes in a later turn.

### 16.2 Attach

- [x] `media.prompt-image` Whether an image the prompt names as `@picture.png`
  reaches the model at once.

## 17. Interactive UI

What only a person at the keyboard of the TUI does.

Frontend: TUI.

- [x] `request.paste` A large paste reaches the model whole, in one message,
  with its indentation.
- [x] `errors.recover` The next prompt works after a failed turn: a request
  carries the second prompt, not only a late retry of the first.
- [x] `shell.terminal` A command cannot wait on the user's terminal: a yes or
  no question, or a pager.
- [x] `tools.terminal-escapes` Control sequences in a tool's output, such as
  OSC 52 that writes the clipboard, do not reach the user's terminal.

## Not measurable as a black box

- [-] What the model does with what it is given: the suite scripts the model.
- [-] Whether the agent obeys instructions in a tool result: the scripted model
  decides what to call, not the agent.
- [-] Internal state that does not reach the provider, the tool or the screen.
- [-] Work at the same time: calls from one reply that run at once, or
  several agents in one workspace or session. Which one goes first is a race,
  so the verdict would change from run to run.
- [-] What a cancel or a signal stops while work is under way: it depends on
  timing, and each TUI's cancel keys change between versions.
- [-] Hosts other than the provider that the agent contacts: telemetry and
  update checks change between versions and with the environment, not with
  how well the agent works.
- [-] Approval prompts: each agent words and draws them its own way, and a
  key that answers one may do something else in the next version.
