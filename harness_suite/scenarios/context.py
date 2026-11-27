"""The context window: growing conversations, compaction and its faults, big
prompts, instruction files."""

import re

from ..fixture import RULE_BEFORE
from ..messages import Error
from ..model import aside_tag
from ..results import INFO, RECOMMENDED, check, gap, label, passed, skipped, unsupported
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, Use, say, use

FACT = "SUITE_FACT_COLOR=violet-7319"
STEPS = 28
RESULT_BYTES = 8000  # Below every agent's tool output limit, so nothing is clipped.
IMAGE_STEPS = 40  # Of about 1300 tokens each.
RULES = f"# Project rules\n\n{RULE_BEFORE}: keep answers short.\n".encode()  # A workspace instructions file.
SHRINK = 0.98  # A task request below this share of the one before it is smaller.


def summaries(run, launch: int | None = None) -> list:
    """Side requests whose reply reached a later task request: the agent put
    the model's answer, a summary, into the conversation. With `launch`, only
    the side requests that launch sent."""
    tasks = run.tasks
    found = []
    for exchange in run.exchanges:
        if launch is not None and exchange.launch != launch:
            continue
        tag = exchange.kind == "aux" and exchange.reply and aside_tag(exchange.reply.text)
        if not tag:
            continue
        pattern = re.compile(re.escape(tag) + r"(?!\d)")
        if any(t.number > exchange.number and pattern.search(t.request.everything) for t in tasks):
            found.append(exchange)
    return found


def reductions(run, launch: int | None = None) -> list:
    """Task requests smaller than the task request before them: by more than
    2%, or by any amount after a rejected request or a summary. With
    `launch`, only within that launch."""
    found, before, reset = [], None, False
    used = {e.number for e in summaries(run, launch)}
    for exchange in run.exchanges:
        if launch is not None and exchange.launch != launch:
            continue
        if exchange.kind == "over-window" or exchange.number in used:
            reset = True
        if exchange.kind != "task":
            continue
        tokens = exchange.request.tokens
        if before is not None and (tokens < before * SHRINK or (reset and tokens < before)):
            found.append(exchange)
        before, reset = tokens, False
    return found


def never_full(run, window: int) -> str:
    largest = max((e.request.tokens for e in run.tasks), default=0)
    return (
        f"The conversation never filled the window: its largest request has about"
        f" {largest:,} of {window:,} tokens, as the agent left out or cut what the tools returned."
    )


class Growing(Scenario):
    """A conversation that grows past a 48k-token window, with text results,
    or in `images` with an image of about 1300 tokens in each result."""

    tools = True
    window = 48_000
    max_requests = 120
    timeout = 240.0

    def fits(self):
        # The images come from the MCP tool.
        return super().fits() and not (self.variant == "images" and self.source != "mcp")

    @property
    def enforced(self) -> int:
        """The window the server enforces."""
        return self.window

    def launches(self):
        return [Launch(prompt=PROMPT + "\n" + FACT + " (keep this fact).")]

    def files(self):
        return dict.fromkeys(self.recipe.instructions, RULES)

    def script(self):
        if self.variant == "images":
            return [*(use(Use("image", value="square")) for _ in range(IMAGE_STEPS)), DONE]
        thinking = "SUITE_THINKING: the next step of the plan." if self.variant == "thinking" else ""
        step = use(Use("text", size=RESULT_BYTES, style="lines"), thinking=thinking)
        return [*(step for _ in range(STEPS)), DONE]

    def shrank(self, run, launch: int | None = None) -> bool:
        return bool(reductions(run, launch) or summaries(run, launch))

    def unshrunk(self, run):
        return skipped(never_full(run, self.enforced) if run.finished_script else "The context never shrank.")


class Overflow(Growing):
    id = "context.overflow"
    title = "A conversation that outgrows the context window still finishes"
    description = (
        f"The model calls the tool {STEPS} times. Each result is 8 KB, so the"
        " conversation grows past the 48k-token window. The provider rejects any"
        " request over the window with the API's real context error. An agent"
        " that compacts or trims, before or after the error, finishes the task. In"
        f" `images`, the MCP tool returns {IMAGE_STEPS} images instead, each a small"
        " file that costs about 1300 tokens, as the APIs count an image by its pixels."
        " In `thinking`, each step also reasons: after compaction, the open tool"
        " turn must still start with its reasoning, unchanged, as Anthropic requires."
        " In `vllm`, `llama-cpp` and `openrouter`, on Chat, the server's real limit"
        " is half the advertised window, as with a local server started with a"
        " smaller context, so the error comes before the agent would compact on its"
        " own, in that server's words and shape. The agent must read it as a"
        " context error and reduce the context."
    )
    variants = ("text", "images", "thinking", "vllm", "llama-cpp", "openrouter")
    wordings = ("vllm", "llama-cpp", "openrouter")  # Chat servers with their own error.

    def fits(self):
        return super().fits() and (self.variant not in self.wordings or self.api == "chat")

    @property
    def enforced(self):
        return self.window // 2 if self.variant in self.wordings else self.window

    def overflow(self, request):
        window, tokens = self.enforced, request.tokens
        if tokens <= window:
            return None
        if self.variant == "vllm":
            # vLLM's OpenAI server: serving_engine.py, `_validate_input`.
            message = (
                f"This model's maximum context length is {window} tokens. However, your prompt contains"
                f" {tokens} input tokens. Please reduce the length of the input messages."
            )
            body = {"message": message, "type": "BadRequestError", "param": "input_tokens", "code": 400}
            return Error("context", body={"error": body})
        if self.variant == "llama-cpp":
            return Error(
                "context",
                body={
                    "error": {
                        "code": 400,
                        "message": "the request exceeds the available context size, try increasing it",
                        "type": "exceed_context_size_error",
                        "n_prompt_tokens": tokens,
                        "n_ctx": window,
                    }
                },
            )
        if self.variant == "openrouter":
            message = (
                f"This endpoint's maximum context length is {window} tokens. However, you requested about"
                f" {tokens} tokens ({tokens - 512} of text input, 512 of tool input). Please reduce the"
                ' length of either one, or use the "middle-out" transform to compress your prompt automatically.'
            )
            return Error(
                "context", body={"error": {"message": message, "code": 400, "metadata": {"provider_name": None}}}
            )
        return Error("context")

    @check(RECOMMENDED)
    def finishes(self, run):
        rejected = sum(e.kind == "over-window" for e in run.exchanges)
        steps = IMAGE_STEPS if self.variant == "images" else STEPS
        if not run.finished_script:
            return gap("stopped", f"Stopped after {len(run.served)} of {steps} calls, {rejected} rejections.")
        if rejected:
            return passed(detail=f"Reduced the context after the error; {rejected} rejected requests.")
        if self.variant in self.wordings:
            return skipped("No request went over the server's limit.")
        if self.shrank(run):
            return passed(detail="Reduced the context before the limit.")
        return skipped(never_full(run, self.enforced))

    @check(
        RECOMMENDED,
        id="context.keeps-fact",
        title="The user's instruction survives compaction",
        description=(
            "In the same growing conversation, the user's prompt holds a fact. Every"
            " request, also after the agent compacts or trims the context, should"
            " still show that fact to the model."
        ),
    )
    def keeps_fact(self, run):
        if any(FACT not in e.request.everything for e in run.tasks):
            return gap("lost", "A request after the context shrank no longer holds the user's fact.")
        if not self.shrank(run):
            return self.unshrunk(run)
        return passed()

    @check(
        RECOMMENDED,
        id="context.keeps-instructions",
        title="The instructions file survives compaction",
        description=(
            "In the same growing conversation, the workspace's instructions files, the"
            " ones the recipe names, hold a rule. Every request that holds it at the"
            " start should still hold it after the agent compacts or trims the context."
        ),
    )
    def keeps_instructions(self, run):
        if not self.recipe.instructions:
            return unsupported(detail="The recipe names no instructions file.")
        if not run.tasks or RULE_BEFORE not in run.tasks[0].request.everything:
            return skipped("The first request does not hold the rule.")
        if any(RULE_BEFORE not in e.request.everything for e in run.tasks):
            return gap("lost", "A request after the context shrank no longer holds the instructions file's rule.")
        if not self.shrank(run):
            return self.unshrunk(run)
        return passed()


class ResumeCompacted(Growing):
    id = "context.resume-compacted"
    title = "A continued session starts from its compacted context"
    description = (
        "The conversation grows past the window and the agent reduces it, as in"
        " `context.overflow`; then the user continues the session with a new"
        " prompt, headless in a new process. The first request of the new turn"
        " should start from the reduced context, not from the whole history before"
        " it, which no longer fits the window. It must carry the old session: the"
        " user's fact or the agent's summary, else the agent started a new one."
    )
    variants = ("text",)
    per_harness = True
    needs = ("continue",)

    def launches(self):
        return [*super().launches(), Launch(prompt=f"{PROMPT}\nSUITE_TURN_TWO", resume=True)]

    def script(self):
        return [*super().script(), say("SUITE_SECOND_DONE")]

    @check(RECOMMENDED)
    def resumed(self, run):
        first = [e for e in run.tasks if e.launch == 0]
        if not self.shrank(run, 0) and not any(e.kind == "over-window" for e in run.exchanges if e.launch == 0):
            return skipped(never_full(run, self.enforced) if first else "The first turn sent no request.")
        later = [e for e in run.exchanges if e.launch == 1 and e.kind in ("task", "over-window")]
        if not later:
            return gap("no second turn", "The continued launch sent no task request.")
        if later[0].kind == "over-window":
            before = f"; the last request before it had {first[-1].request.tokens:,}" if first else ""
            return gap(
                "whole history",
                f"The new turn's first request has about {later[0].request.tokens:,} tokens, over the"
                f" {self.enforced:,}-token window{before}.",
            )
        tags = [aside_tag(e.reply.text) for e in summaries(run, 0)]
        text = later[0].request.everything
        if FACT not in text and not any(tag in text for tag in tags):
            return gap("not resumed", "The new turn's first request holds neither the user's fact nor a summary.")
        if not run.finished_script:
            return gap("stopped", "The second turn did not finish.")
        return passed(detail=f"{later[0].request.tokens:,} tokens in the new turn's first request.")


class BigPrompt(Scenario):
    id = "context.big-prompt"
    title = "A user prompt larger than the context window"
    description = (
        "A first launch gets a small task, to show that the agent works. Then a new"
        " launch gets a prompt of 1 MiB, more than the 200k-token window. The label"
        " shows whether the agent refuses it before it sends anything, finishes the"
        " task (it trimmed or compacted the prompt), or stops."
    )

    def launches(self):
        return [Launch(), Launch(prompt=PROMPT + "\n" + ("SUITE_BULK " * 95_000), track=1)]

    def scripts(self):
        return [[DONE], [DONE]]

    @check(INFO)
    def handling(self, run):
        if not any(e.request for e in run.exchanges if e.launch == 0):
            return skipped("The small task sent no request.")
        sent = [e for e in run.exchanges if e.request and e.launch == 1]
        if not sent:
            return label("refused before sending")
        rejected = sum(e.kind == "over-window" for e in sent)
        detail = f"{len(sent)} {'request' if len(sent) == 1 else 'requests'}, {rejected} over the window."
        return label("finished" if 1 in run.finished else "stopped", detail)


NESTED_RULE, MODULE = "SUITE_RULE_NESTED", "pkg/suite_module.py"


def read_step(recipe, path: str) -> Use:
    """Read a workspace file with the agent's read tool, or else with `cat`."""
    return Use("read", value=path) if "read" in recipe.native else Use("text", command=f"cat {path}")


class InstructionsNested(Scenario):
    id = "context.instructions-nested"
    title = "An instructions file in a subfolder"
    description = (
        "Next to the workspace's instructions file, a subfolder `pkg/` has its own,"
        " with the same name and another rule, as in a monorepo. The model reads a"
        " file in `pkg/`. The label shows when the subfolder's rule reaches the"
        " model: in the first request, once the model works in the folder, or never."
    )
    tools = True
    sources = ("native",)

    def unsupported(self):
        if not self.recipe.instructions:
            return "The recipe names no instructions file"
        return None

    def files(self):
        name = self.recipe.instructions[0]
        return {
            name: f"# Project rules\n\n{RULE_BEFORE}: keep answers short.\n".encode(),
            f"pkg/{name}": f"# Package rules\n\n{NESTED_RULE}: use tabs in this package.\n".encode(),
            MODULE: b"VALUE = 1\n",
        }

    def script(self):
        return [use(read_step(self.recipe, MODULE)), DONE]

    @check(INFO)
    def loaded(self, run):
        if not run.requests:
            return skipped("The agent sent no task request.")
        if NESTED_RULE in run.requests[0].everything:
            return label("at start")
        later = next((n for n, r in enumerate(run.requests) if NESTED_RULE in r.everything), None)
        if later is not None:
            return label("when used", f"First in request {later + 1}.")
        return label("never" if run.finished_script else "never, stopped")


LARGE_RULES = 100_000  # Bytes, about 25k tokens.


class InstructionsLarge(Scenario):
    id = "context.instructions-large"
    title = "A large instructions file"
    description = (
        "The workspace's instructions file is 100 KB, about 25,000 tokens, with a"
        " marker at its start and its end, as a project that pastes its whole style"
        " guide does. The label shows what the first request holds: the whole file,"
        " its start only, or none of it."
    )

    def unsupported(self):
        if not self.recipe.instructions:
            return "The recipe names no instructions file"
        return None

    def files(self):
        line = "- Prefer clear names, small functions and tests for every change.\n"
        body = line * (LARGE_RULES // len(line))
        text = f"# Project rules\n\nSUITE_RULES_START\n{body}SUITE_RULES_END\n"
        return {self.recipe.instructions[0]: text.encode()}

    def script(self):
        return [DONE]

    @check(INFO)
    def kept(self, run):
        if not run.requests:
            return skipped("The agent sent no task request.")
        first = run.requests[0]
        start, end = "SUITE_RULES_START" in first.everything, "SUITE_RULES_END" in first.everything
        if start and end:
            return label("whole")
        if start:
            return label("start only", "The end marker is missing: the agent cut the file.")
        return label("not loaded")
