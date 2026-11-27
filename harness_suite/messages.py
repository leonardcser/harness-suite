"""The provider's view of a conversation, shared by every wire API.

Requests from all three APIs are parsed into the same `Request`, so checks
judge the conversation, not a wire format. Replies are what the scripted model
says; the API modules turn them into wire events.
"""

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class Media:
    """An image or document part. Only its bytes' hash is kept."""

    media_type: str
    sha256: str
    size: int
    format: str | None = None  # The image format its bytes show, as "png".
    width: int = 0  # In pixels; 0 when unknown.
    height: int = 0

    @property
    def image(self) -> bool:
        return self.media_type.startswith("image/")

    @property
    def tokens(self) -> int:
        """What the part costs: an image by its pixels, as Anthropic counts
        them, at most 1600; anything else, or an image of unknown size, 85."""
        if not self.image or not self.width:
            return 85
        return max(85, min(1600, self.width * self.height // 750))


@dataclass(frozen=True)
class Thinking:
    """Reasoning content. `signature` is the Anthropic signature or the
    Responses encrypted content; Chat reasoning has none."""

    text: str
    signature: str | None = None


@dataclass(frozen=True)
class Call:
    id: str
    name: str
    arguments: str  # The raw JSON text, which may be invalid on purpose.

    def args(self) -> dict | None:
        try:
            value = json.loads(self.arguments)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class Result:
    id: str
    parts: tuple
    error: bool = False

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.parts if isinstance(part, Text))

    @property
    def media(self) -> list[Media]:
        return [part for part in self.parts if isinstance(part, Media)]


@dataclass
class Message:
    role: str  # "user" or "assistant". Tool results are parts of user messages.
    parts: list

    def of(self, kind) -> list:
        return [part for part in self.parts if isinstance(part, kind)]

    @property
    def text(self) -> str:
        return "".join(part.text for part in self.of(Text))


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict


@dataclass
class Request:
    api: str
    raw: dict
    headers: dict
    model: str | None
    stream: bool
    system: str
    messages: list[Message]
    tools: list[Tool]
    parameters: dict  # Normalized sampling, reasoning and tier settings.
    previous_response_id: str | None = None
    # Responses: the input items as the model sees them, with the history that
    # `previous_response_id` and item references bring back from the store.
    items: list[dict] | None = None
    # Whether the real API returns reasoning for this request: Anthropic only
    # when it turns thinking on; Chat servers and Responses reasoning models
    # always do.
    reasons: bool = True

    @property
    def calls(self) -> list[Call]:
        return [c for m in self.messages if m.role == "assistant" for c in m.of(Call)]

    @property
    def images(self) -> list[Media]:
        """Every image the request holds, in messages and in tool results."""
        parts = [p for m in self.messages for p in m.parts]
        parts += [p for r in self.results for p in r.parts]
        return [p for p in parts if isinstance(p, Media) and p.image]

    @property
    def results(self) -> list[Result]:
        return [r for m in self.messages if m.role == "user" for r in m.of(Result)]

    def result(self, call_id: str) -> Result | None:
        return next((r for r in self.results if r.id == call_id), None)

    @property
    def user_texts(self) -> list[str]:
        """The text of each user message that has any (tool results do not)."""
        return [m.text for m in self.messages if m.role == "user" and m.text]

    @property
    def last(self) -> Message | None:
        """The newest turn. System messages among the turns are instructions,
        not turns: some harnesses put context there after the user's prompt."""
        turns = [m for m in self.messages if m.role not in ("system", "developer")]
        return turns[-1] if turns else None

    @property
    def everything(self) -> str:
        """All text the model can see, for presence checks: the system prompt,
        the tools as the request defines them, the messages and the results."""
        return json.dumps(
            [self.system, self.raw.get("tools") or [], [m.text for m in self.messages]]
            + [r.text for r in self.results],
            ensure_ascii=False,
        )

    def place(self, text: str) -> str | None:
        """Where the model sees `text` first: "system prompt", "tool
        description" (a tool or a group of tools), or "message N"; None if
        the request does not hold it."""
        if text in self.system:
            return "system prompt"
        if text in json.dumps(self.raw.get("tools") or [], ensure_ascii=False):
            return "tool description"
        for number, message in enumerate(self.messages, 1):
            if text in message.text or any(text in r.text for r in message.of(Result)):
                return f"message {number}"
        return None

    @property
    def visible(self) -> str:
        """The plain text the model sees: system, messages and tool results."""
        parts = [self.system]
        for message in self.messages:
            parts += [p.text for p in message.parts if isinstance(p, Text | Result)]
        return "\n".join(parts)

    @property
    def tokens(self) -> int:
        """A synthetic token count: four UTF-8 bytes per token. Text that is
        not valid Unicode still counts, for a request the provider refuses."""

        def size_of(text: str) -> int:
            return len(text.encode(errors="surrogatepass"))

        size = size_of(json.dumps(self.raw.get("tools", []), ensure_ascii=False))
        size += size_of(self.system)
        for message in self.messages:
            for part in message.parts:
                if isinstance(part, Text | Thinking):
                    size += size_of(part.text)
                elif isinstance(part, Call):
                    size += size_of(part.arguments) + len(part.name)
                elif isinstance(part, Result):
                    size += size_of(part.text) + 4 * sum(m.tokens for m in part.media)
                elif isinstance(part, Media):
                    size += 4 * part.tokens
        return size // 4 + 1


@dataclass(frozen=True)
class Error:
    """A provider error, sent as the HTTP response or inside the stream.

    `kind` is one of: rate, server, overloaded, auth, quota, context, invalid,
    model (an unknown model, 404), too-large (413), gateway: a proxy's HTML
    error page, not the API's JSON, or reset: the connection reset, with no
    response at all.
    """

    kind: str
    retry_after: str | None = None
    retry_after_ms: str | None = None
    message: str | None = None
    # The whole JSON body, for an OpenAI-compatible server's own error shape.
    body: dict | None = None


BEFORE_FINISH = -1  # An event position: after all content, before the finish events.


@dataclass(frozen=True)
class Stream:
    """How the reply is delivered. Event counts refer to the API's events."""

    cut_after: int | None = None  # Close the connection after N events.
    error_after: int | None = None  # A mid-stream error after N events, or BEFORE_FINISH.
    error: Error | None = None
    corrupt: int | None = None  # Truncate the JSON of event N.
    drop_terminal: bool = False  # Omit the finish events, then close normally.
    framing: str = "standard"  # standard, comments, crlf, no-space, gzip, long-line
    unknown: bool = False  # Add events that a newer API version could send.
    azure_filter: bool = False  # On Chat, Azure's content filter chunk first.
    chunk: int | None = None  # Write N bytes at a time.
    pace: float = 0.0  # Seconds between events, as a slow model streams.
    # How tool calls stream: "standard", one call after the other in fragments;
    # "interleaved", the fragments of parallel calls in turn (Chat, Responses);
    # "whole", each call's arguments with no fragments (Chat, Responses); on Chat also
    # "repeated-identity" and "blank-identity", the ID and name in every chunk.
    calls: str = "standard"
    # On Responses, the reasoning's encrypted content only in the final
    # response, not in its item's done event, as Azure sends it.
    late_reasoning: bool = False


@dataclass(frozen=True)
class Reply:
    text: str = ""
    thinking: str = ""
    # How the reasoning reaches the client: "shown" as text, "omitted" as a
    # signature with empty text (Anthropic), or "redacted" as opaque data.
    # Responses sends both hidden forms as encrypted content with no summary.
    thinking_form: str = "shown"
    calls: tuple[Call, ...] = ()
    # None: "tool" with calls, else "end". Or "length" or "refusal"; on Chat
    # also "error", as OpenRouter ends a failed stream; on Anthropic
    # "window", the context window full mid-reply.
    stop: str | None = None
    output_tokens: int = 32
    stream: Stream = field(default_factory=Stream)

    @property
    def stop_reason(self) -> str:
        return self.stop or ("tool" if self.calls else "end")


def signature(text: str) -> str:
    """The signature the provider gives a thinking block, to verify it later."""
    return "suite-sig-" + hashlib.sha256(text.encode()).hexdigest()[:32]


def opaque(text: str) -> str:
    """The data of hidden reasoning: a redacted block, or a signature for
    reasoning whose text the provider omits."""
    return "suite-opaque-" + hashlib.sha256(text.encode()).hexdigest()[:48]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
