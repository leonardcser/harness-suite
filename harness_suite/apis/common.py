"""Helpers shared by the wire APIs."""

import base64
import binascii
import json
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass

from ..messages import Media, Message, sha256

NAME_PATTERN = re.compile(r"[a-zA-Z0-9_-]{1,64}")


@dataclass(frozen=True)
class Rejection:
    """A rule of the real API that a request breaks. `rule` is its kind, as a
    case that provokes it names it: text, tool-name, tool-schema, tool-count,
    tool-id, tool-pairs, reasoning, image, cache, sampling, store or body."""

    rule: str
    text: str  # The API's own error message.


Rule = Callable[..., str | None]  # (request, store) -> the error message, or None.


def rejections(request, store: dict, rules: list[tuple[str, Rule]]) -> list[Rejection]:
    """Every rule of `rules` the request breaks, in the order the API checks them."""
    return [Rejection(kind, text) for kind, check in rules if (text := check(request, store))]


@dataclass(frozen=True)
class Frame:
    """One server-sent event. `data` is None for a comment."""

    data: str | None
    event: str | None = None
    terminal: bool = False  # A finish event, which `Stream.drop_terminal` omits.
    comment: str = ""


def frame(value: dict, named: bool, terminal: bool = False) -> Frame:
    return Frame(
        json.dumps(value, ensure_ascii=False),
        value["type"] if named else None,
        terminal,
    )


# The image formats every model API takes.
IMAGE_FORMATS = ("png", "jpeg", "gif", "webp")


def media(media_type: str, data: str) -> Media:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raw = data.encode()
    return Media(media_type, sha256(raw), len(raw), *picture(raw))


def picture(raw: bytes) -> tuple[str | None, int, int]:
    """The image format the bytes show, and the size in pixels, 0 if unknown."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return "png", *struct.unpack(">II", raw[16:24])
    if raw[:6] in (b"GIF87a", b"GIF89a") and len(raw) >= 10:
        return "gif", *struct.unpack("<HH", raw[6:10])
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "webp", *webp_size(raw)
    if raw.startswith(b"\xff\xd8"):
        return "jpeg", *jpeg_size(raw)
    return None, 0, 0


def webp_size(raw: bytes) -> tuple[int, int]:
    chunk = raw[12:16]
    if chunk == b"VP8X" and len(raw) >= 30:
        return 1 + int.from_bytes(raw[24:27], "little"), 1 + int.from_bytes(raw[27:30], "little")
    if chunk == b"VP8L" and len(raw) >= 25:
        bits = int.from_bytes(raw[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 " and len(raw) >= 30:
        return int.from_bytes(raw[26:28], "little") & 0x3FFF, int.from_bytes(raw[28:30], "little") & 0x3FFF
    return 0, 0


def jpeg_size(raw: bytes) -> tuple[int, int]:
    """From the first start-of-frame segment."""
    at = 2
    while at + 9 <= len(raw) and raw[at] == 0xFF:
        marker = raw[at + 1]
        if marker == 0xFF:
            at += 1
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", raw[at + 5 : at + 9])
            return width, height
        at += 2 + int.from_bytes(raw[at + 2 : at + 4], "big")
    return 0, 0


def unsupported_image(request) -> str | None:
    """OpenAI's error for an image in a format it does not take."""
    if any(image.format not in IMAGE_FORMATS for image in request.images):
        return (
            "You uploaded an unsupported image. Please make sure your image has of"
            " one the following formats: ['png', 'jpeg', 'gif', 'webp']."
        )
    return None


def data_url(url: str) -> Media | None:
    match = re.fullmatch(r"data:([^;,]+);base64,(.*)", url or "", re.S)
    return media(match[1], match[2]) if match else None


def text_of(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text_of(item) for item in value)
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return ""


def append(messages: list[Message], role: str, parts: list) -> None:
    """Merge consecutive messages of a role, as every API does."""
    if not parts:
        return
    if messages and messages[-1].role == role:
        messages[-1].parts.extend(parts)
    else:
        messages.append(Message(role, list(parts)))


def fragments(value: str, sizes=(7, 3, 11, 5)) -> list[str]:
    """Split a string into uneven pieces, as a real model's deltas are."""
    pieces, start, index = [], 0, 0
    while start < len(value):
        size = sizes[index % len(sizes)]
        pieces.append(value[start : start + size])
        start, index = start + size, index + 1
    return pieces


def text_pieces(text: str, stream) -> list[str]:
    """The deltas of a reply's text: all in one for a `long-line` stream."""
    return [text] if stream.framing == "long-line" else fragments(text)


def lone_surrogate(value, where: str = "body") -> str | None:
    """Where a string in a JSON value holds half of a UTF-16 surrogate pair,
    which no API takes: JSON text must be valid Unicode."""
    if isinstance(value, str):
        return where if any("\ud800" <= c <= "\udfff" for c in value) else None
    items = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else ()
    for key, item in items:
        if found := lone_surrogate(item, f"{where}.{key}"):
            return found
    return None


def not_object(parameters, name: str) -> str | None:
    """OpenAI's error for function parameters whose root is not an object.
    No parameters, or a schema with no type, such as `{}`, are fine."""
    if parameters is None or isinstance(parameters, dict) and parameters.get("type", "object") == "object":
        return None
    kind = parameters.get("type") if isinstance(parameters, dict) else type(parameters).__name__
    return (
        f"Invalid schema for function '{name}': schema must be a JSON Schema of"
        f" 'type: \"object\"', got 'type: \"{kind}\"'."
    )
