"""Images, documents and other media in tool results."""

import base64

from ..apis.anthropic import MANY_IMAGES, MANY_SIDE
from ..fixture import BLOB, BLUE, PICTURES, RED, pdf, wav
from ..messages import sha256
from ..results import INFO, OPTIONAL, RECOMMENDED, check, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, Use, use

FORMATS = ("png", "jpeg", "gif", "webp")  # Formats every model API takes.
CONTROL = "normal.png"  # A small image every API takes, read before one it may not.


def picture(name: str) -> bytes:
    return PICTURES[name][0]()


def media_after(run) -> list:
    results = run.step_result(0)
    return [m for r in results for m in r.media] + [
        p
        for request in run.requests
        for m in request.messages
        if m.role == "user"
        for p in m.parts
        if getattr(p, "sha256", None)
    ]


def carrying(run, step: int):
    """The first request that holds every result of a step."""
    for served in (s for s in run.served if s.step == step and s.calls):
        found = next((r for r in run.requests if all(r.result(c.id) for c in served.calls)), None)
        if found:
            return found
    return None


def images_reached(run, expected: list[str], text: str):
    """Pass if every expected image reached the model as an image part. Images
    the agent re-encoded (resized, or another format) still count. A recipe
    that does not say the agent lacks images claims them, so text alone fails."""
    seen = media_after(run)
    if all(h in {m.sha256 for m in seen} for h in expected):
        return passed()
    images = {m.sha256: m.media_type for m in seen if m.image}
    if len(images) >= len(expected):
        return passed("re-encoded", f"Re-encoded as {', '.join(sorted(set(images.values())))}.")
    if images:
        return failed(f"{len(images)} of {len(expected)}", "Only some images reached the model.")
    return failed("text only", f"The result reached the model as {text[:80]!r}")


class Image(Scenario):
    id = "media.image"
    title = "An image from an MCP tool reaches the model as an image"
    description = (
        "The MCP fixture returns a small image as PNG, JPEG, GIF or WebP, the"
        " formats every model API takes, or two PNG images (`two-images`). An agent"
        " that passes images on must send each as a native image part. The detail notes images the agent re-encoded."
    )
    tools = True
    sources = ("mcp",)
    variants = (*FORMATS, "two-images")
    features = ("images",)

    def script(self):
        if self.variant == "two-images":
            return [use("two-images"), DONE]
        return [use(Use("image", value=self.variant)), DONE]

    @check(OPTIONAL)
    def reached(self, run):
        if not run.served:
            return skipped("The call was never served.")
        results = run.step_result(0)
        if not results:
            return failed("no result")
        expected = [RED, BLUE] if self.variant == "two-images" else [picture(self.variant)]
        return images_reached(run, [sha256(data) for data in expected], results[0].text)


class NativeRead(Scenario):
    id = "media.read-image"
    title = "The agent's own read tool sends an image file as an image"
    description = (
        "The model asks the agent's native file-read tool for a small PNG, JPEG,"
        " GIF or WebP file in the workspace, or for a JPEG named `.png`"
        " (`misnamed`). An agent that reads images must send it as an image part,"
        " in a request the API takes: Anthropic rejects an image whose media type"
        " does not match its bytes, which the OpenAI APIs accept."
    )
    tools = True
    sources = ("native",)
    variants = (*FORMATS, "misnamed")
    binds = ("read",)
    features = ("images",)

    @property
    def provokes(self):
        return ("image",) if self.variant == "misnamed" else ()

    @property
    def path(self) -> str:
        return "picture.png" if self.variant == "misnamed" else f"picture.{self.variant}"

    @property
    def format(self) -> str:
        return "jpeg" if self.variant == "misnamed" else self.variant

    def files(self):
        return {self.path: picture(self.format)}

    def script(self):
        return [use(Use("read", value=self.path)), DONE]

    @check(OPTIONAL)
    def reached(self, run):
        if not run.served:
            return skipped("The read call was never served.")
        results = run.step_result(0)
        if not results:
            return failed("no result")
        return images_reached(run, [sha256(picture(self.format))], results[0].text)


class PromptImage(Scenario):
    id = "media.prompt-image"
    title = "An image the user's prompt names"
    description = (
        "The user's prompt names a workspace image as `@picture.png`, as people"
        " point an agent at a screenshot. The label shows whether the first request"
        " already holds it as an image, without a tool call, in the TUI as headless."
    )

    def files(self):
        return {"picture.png": picture("png")}

    def launches(self):
        # Text after the mention: Enter must send the prompt, not pick from a
        # file list that the mention opens, which shows only after a delay.
        return [Launch(prompt=f"{PROMPT}\nLook at @picture.png now.")]

    @check(INFO)
    def attached(self, run):
        if not run.requests:
            return skipped("The agent sent no task request.")
        images = run.requests[0].images
        if sha256(picture("png")) in {m.sha256 for m in images}:
            return label("attached")
        return label("attached, re-encoded" if images else "not attached")


BINARY = {"pdf": pdf, "image-resource": lambda: BLUE, "blob": lambda: BLOB, "audio": wav}


class Content(Scenario):
    id = "media.content"
    title = "Binary MCP content never reaches the model as raw base64"
    description = (
        "After its text, the MCP tool's result holds binary content the protocol"
        " defines: an embedded PDF, PNG image or binary data (`blob`), or audio."
        " The agent may send it as a native part, describe it or leave it out,"
        " but not paste its base64 as text: that floods the context with tokens"
        " the model cannot read. A pass says what the model got."
    )
    tools = True
    sources = ("mcp",)
    variants = tuple(BINARY)
    tool_may_be_missing = True

    def script(self):
        return [use(self.variant), DONE]

    @check(RECOMMENDED)
    def not_raw(self, run):
        results = run.step_result(0)
        if not results:
            return skipped(f"No result came back. {run.exit_detail()}".strip())
        data = BINARY[self.variant]()
        encoded = base64.b64encode(data).decode()
        if any(encoded[:64] in request.visible for request in run.requests):
            return gap("raw base64", f"The model got the {len(encoded):,} characters of base64 as text.")
        if sha256(data) in {m.sha256 for m in media_after(run)}:
            return passed("native part")
        if self.variant == "image-resource" and any(m.image for m in media_after(run)):
            return passed("native part, re-encoded")
        return passed("left out or described", f"The result: {results[0].text[:120]!r}")


class Limits(Scenario):
    id = "media.limits"
    title = "An image the API does not take does not break the session"
    description = (
        "The model reads an image the API would refuse, with the agent's own read"
        " tool or from an MCP tool: a PNG of 9000 by 16 px (`wide`), over"
        " Anthropic's 8000 px a side; a PNG of 9 MB (`heavy`), 12 MB in base64, over its 10 MB; or a"
        " BMP (`bmp`), which no API takes. The OpenAI APIs take the first two. The"
        " agent should resize, convert or leave out what the API refuses, and"
        " finish. An image the API rejects stays in the history and can break"
        " every later request. A normal PNG comes first: only an agent that passes"
        " it on as an image is measured."
    )
    tools = True
    variants = ("wide", "heavy", "bmp")
    max_requests = 30
    provokes = ("image",)

    def unsupported(self):
        if self.source == "native" and "read" not in self.recipe.native:
            return "The recipe binds no native read tool"
        return None

    def name(self) -> str:
        return f"{self.variant}.{PICTURES[self.variant][1].removeprefix('image/')}"

    def files(self):
        if self.source != "native":
            return {}
        return {CONTROL: picture("png"), self.name(): picture(self.variant)}

    def script(self):
        if self.source == "native":
            return [use(Use("read", value=CONTROL)), use(Use("read", value=self.name())), DONE]
        return [use(Use("image", value="png")), use(Use("image", value=self.variant)), DONE]

    @check(RECOMMENDED)
    def handled(self, run):
        before = carrying(run, 0)
        if before is None:
            return skipped("The normal image got no result.")
        if not before.images:
            return skipped("The normal image did not reach the model as an image; media.image judges that.")
        if not run.was_served(1):
            return skipped("The call for the refused image was never served.")
        if not run.finished_script:
            return gap("stopped")
        after = carrying(run, 1)
        known = {m.sha256 for m in before.images}
        added = [m for m in after.images if m.sha256 not in known] if after else []
        if any(m.sha256 == sha256(picture(self.variant)) for m in added):
            return passed("sent as is")
        if added:
            return passed("re-encoded", ", ".join(sorted({f"{m.width}x{m.height} {m.media_type}" for m in added})))
        return passed("left out")


SHOTS = 24  # Over the 20 images after which Anthropic limits every image's sides.


class ManyImages(Scenario):
    id = "media.many-images"
    title = "A conversation with many images stays within the API's limits"
    description = (
        f"The model reads {SHOTS} images one after the other, as a loop over"
        " screenshots does, with the agent's own read tool or from an MCP tool."
        " Each is a PNG of 7000 by 16 px, which the API takes alone. But once a"
        f" request holds more than {MANY_IMAGES} images, Anthropic takes none over"
        f" {MANY_SIDE} px a side. The agent should resize the images, or leave out"
        " older ones, and finish."
    )
    tools = True
    apis = ("anthropic",)  # Only Anthropic limits the images of a request with many.
    max_requests = 80
    timeout = 240.0
    provokes = ("image",)

    def unsupported(self):
        if self.source == "native" and "read" not in self.recipe.native:
            return "The recipe binds no native read tool"
        return None

    def names(self) -> list[str]:
        return [f"shot-{n:02d}.png" for n in range(SHOTS)]

    def files(self):
        return dict.fromkeys(self.names(), picture("long")) if self.source == "native" else {}

    def script(self):
        if self.source == "native":
            return [*(use(Use("read", value=name)) for name in self.names()), DONE]
        return [*(use(Use("image", value="long")) for _ in range(SHOTS)), DONE]

    @check(RECOMMENDED)
    def handled(self, run):
        fullest = max(run.requests, key=lambda r: len(r.images), default=None)
        if fullest is None or not fullest.images:
            return skipped("No image reached the model as an image; media.image judges that.")
        if not run.finished_script:
            return gap("stopped", f"After {len(run.served)} of {SHOTS} reads.")
        if len(fullest.images) <= MANY_IMAGES:
            return passed("kept few", f"At most {len(fullest.images)} images in a request.")
        side = max(max(i.width, i.height) for i in fullest.images)
        return passed("resized", f"{len(fullest.images)} images in a request, none over {side} px a side.")
