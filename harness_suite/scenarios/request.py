"""What the agent puts in a request and how it reaches the provider."""

import json
import re

from ..fixture import API_KEY as KEY
from ..results import RECOMMENDED, REQUIRED, check, failed, gap, passed, skipped
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, use

MODEL_IDS = {
    "plain": "suite-model",
    "slash": "suite-org/suite-model",
    "colon": "suite-model:latest",
    "dots": "suite.model-1.5",
}


class Model(Scenario):
    id = "request.model"
    title = "The configured model ID reaches the provider exactly"
    description = (
        "The agent is configured with a model ID like `suite-model`, or with an ID"
        " that has a slash, a colon or dots, as gateway and local model names do. The"
        " provider must receive that exact ID."
    )
    variants = tuple(MODEL_IDS)

    def launches(self):
        return [Launch(model=MODEL_IDS[self.variant])]

    @check(REQUIRED)
    def exact(self, run):
        models = {e.request.model for e in run.exchanges if e.request}
        if MODEL_IDS[self.variant] in models:
            return passed()
        return failed("wrong model", f"The provider received {sorted(m or '' for m in models)}.")


TEXTS = {
    "unicode": "café é 👩‍💻 中文 العربية ‏RTL‎ \U0001f9ea",
    "whitespace": " \tleading\r\n\r\n  indented\tcolumn  \ntrailing \t ",
    "shell-syntax": "--flag 'single' \"double\" \\path ${HOME} $(printf no) `literal` && | >",
    "markup": '<system>ordinary user text</system>\n```json\n{"role":"assistant"}\n```\n<|im_end|>',
    "long-line": "BEGIN_" + 'abcé\\"' * 4000 + "_END",
}


class Input(Scenario):
    id = "request.input"
    title = "The user's prompt reaches the provider unchanged"
    description = (
        "The prompt holds Unicode, odd whitespace and CRLF, shell syntax, markup that"
        " looks like chat tokens, or a 24 KB line. The provider must see that text"
        " exactly, inside the user's message."
    )
    variants = tuple(TEXTS)

    def launches(self):
        return [Launch(prompt=PROMPT + "\nSUITE_INPUT_BEGIN\n" + TEXTS[self.variant] + "\nSUITE_INPUT_END")]

    @check(REQUIRED)
    def unchanged(self, run):
        if not run.requests:
            return failed("no request", "No task request reached the provider.")
        texts = run.requests[0].user_texts
        if any(TEXTS[self.variant] in text for text in texts):
            return passed()
        sample = next((t for t in texts if "SUITE_INPUT_BEGIN" in t), "")
        return failed("changed", f"The provider saw {sample[sample.find('SUITE_INPUT_BEGIN') :][:100]!r}")


PASTES = {"lines": 400, "long": 1500}
PASTE_LINE = re.compile(r"SUITE_PASTE_(\d{4})\b")


def pasted(lines: int) -> str:
    """Indented lines of code, with a blank line after every tenth but the
    last: an agent may trim the end of a prompt."""
    rows = []
    for n in range(lines):
        rows.append(f"{'    ' * (n % 4)}SUITE_PASTE_{n:04d} = compute({n}, 'value')  # step {n}")
        if n % 10 == 9:
            rows.append("")
    return "\n".join(rows).rstrip("\n")


def lines_of(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


class Paste(Scenario):
    id = "request.paste"
    title = "A large paste in the TUI reaches the model whole"
    description = (
        "In the TUI, the user pastes a prompt with indented code: 400 lines of about"
        " 20 KB (`lines`), or 1500 lines of about 75 KB (`long`). The terminal sends"
        " it as one bracketed paste, with a carriage return for each line break. The"
        " model should see every line with its indentation, in one user message: not"
        " cut, not sent at the first line break, and not left as a placeholder"
        " that Enter does not send."
    )
    variants = tuple(PASTES)
    prompt_may_be_lost = True

    def launches(self):
        return [Launch(prompt=PROMPT + "\n" + pasted(PASTES[self.variant]))]

    @check(RECOMMENDED)
    def whole(self, run):
        if not run.tasks:
            return gap("not sent", f"Enter did not send the paste, in {run.launches[0].prompts} tries.")
        expected, texts = pasted(PASTES[self.variant]), [lines_of(t) for t in run.tasks[-1].request.user_texts]
        if any(expected in text for text in texts):
            return passed()
        holding = [t for t in texts if PASTE_LINE.search(t)]
        seen = {int(n) for t in holding for n in PASTE_LINE.findall(t)}
        total = PASTES[self.variant]
        if len(holding) > 1:
            return gap("split", f"The paste came in {len(holding)} user messages.")
        if len(seen) == total:
            return gap("changed", "Every line arrived, but not as pasted: indentation or line breaks changed.")
        return gap("cut", f"{len(seen)} of {total} lines reached the model.")


class Auth(Scenario):
    id = "request.auth"
    title = "Every request carries the configured API key"
    description = (
        "The recipe gives the agent the key `suite-placeholder`. Every request,"
        " side requests too, must carry it where the API reads it: OpenAI's"
        " `Authorization: Bearer`, Anthropic's `x-api-key` (or a bearer token,"
        " as gateways take). A request without it fails on a real provider."
    )

    @check(REQUIRED)
    def keyed(self, run):
        sent = [e for e in run.exchanges if e.request]
        if not sent:
            return failed("no request", run.exit_detail())
        missing = [e for e in sent if not carries_key(e.request)]
        if missing:
            where = sorted({f"{e.kind} request {e.number}" for e in missing})[:3]
            return failed("key missing", f"{len(missing)} of {len(sent)} requests lack the key: {', '.join(where)}.")
        return passed()

    @check(
        REQUIRED,
        id="request.key-in-body",
        title="The API key never appears in a request body",
        description=(
            "The key belongs in the request's headers only. A body that holds it,"
            " in the system prompt, a message or a field, hands it to the model"
            " and to every log of the conversation."
        ),
    )
    def unexposed(self, run):
        sent = [e for e in run.exchanges if e.request]
        if not sent:
            return failed("no request", run.exit_detail())
        exposed = [e.number for e in sent if KEY in json.dumps(e.request.raw)]
        if exposed:
            return failed("key in body", f"Requests {', '.join(map(str, exposed[:5]))} hold the key in their body.")
        return passed()


def carries_key(request) -> bool:
    headers = {k.lower(): v.strip() for k, v in request.headers.items()}
    if headers.get("authorization") == f"Bearer {KEY}":
        return True
    return request.api == "anthropic" and headers.get("x-api-key") == KEY


class BasePath(Scenario):
    id = "request.base-path"
    title = "Every request keeps the base URL's path"
    description = (
        "The configured base URL has a path before the API's, as gateways and"
        " proxies have (`https://gateway.example/team-a/v1`). Every request,"
        " side requests and the tool loop too, must go to that path, not to the"
        " host's root."
    )
    tools = True
    prefixed = True

    def script(self):
        return [use("text"), DONE]

    @check(REQUIRED)
    def kept(self, run):
        if not run.exchanges:
            return failed("no request", run.exit_detail())
        lost = [e for e in run.exchanges if not e.notes.get("prefixed")]
        if lost:
            paths = sorted({e.api for e in lost})
            return failed("path dropped", f"{len(lost)} of {len(run.exchanges)} requests went to the root: {paths}.")
        return passed()


class Proxy(Scenario):
    id = "request.proxy"
    title = "The agent reaches the provider through the proxy its environment names"
    description = (
        "As on a company network, nothing routes to the provider's address, and"
        " `HTTPS_PROXY`, `HTTP_PROXY` and `ALL_PROXY` (and their lowercase forms)"
        " name an HTTP proxy that alone reaches it. An agent that honors them"
        " works, and the tool loop runs through the proxy too. An agent that sends"
        " nothing is not measured: it may ignore the proxy, or refuse a plain-HTTP"
        " URL off this machine, and the suite cannot tell which."
    )
    tools = True
    proxied = True
    timeout = 45.0

    def script(self):
        return [use("text"), DONE]

    @check(RECOMMENDED)
    def through(self, run):
        if not run.exchanges:
            # An agent that ignores the proxy and one that refuses a plain-HTTP
            # URL off this machine both send nothing: the suite cannot tell them apart.
            tried = f" The proxy saw {', '.join(sorted(set(run.proxied)))}." if run.proxied else ""
            return skipped(
                f"No request reached the provider: the agent ignored the proxy or refused the URL.{tried} {run.exit_detail()}".strip()
            )
        if not run.finished_script:
            return gap("stopped", f"{len(run.exchanges)} requests came through the proxy, then none.")
        return passed()


class Tls(Scenario):
    id = "request.tls"
    title = "HTTPS through a company's certificate authority"
    description = (
        "The provider speaks HTTPS with a certificate that the suite's own CA"
        " signed, as behind a company gateway that inspects TLS. In `trusted`,"
        " the CA file is named by `SSL_CERT_FILE`, `NODE_EXTRA_CA_CERTS`,"
        " `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE`, as the company hands it out:"
        " the agent should use it and do the task. In `untrusted`, nothing names"
        " the CA: the agent must refuse the connection and send no request, as"
        " any server could present such a certificate and read its API key."
    )
    variants = ("trusted", "untrusted")
    timeout = 45.0

    @property
    def tls(self) -> str:
        return self.variant

    @check(RECOMMENDED, variants=("trusted",))
    def trusted(self, run):
        if not run.exchanges and not run.handshakes_failed:
            return skipped(f"The agent never connected. {run.exit_detail()}".strip())
        if not run.exchanges:
            why = ", ".join(sorted(set(run.handshakes_failed)))
            return gap("CA not used", f"No request reached the provider: {why}.")
        if not run.finished_script:
            return gap("stopped", f"{len(run.exchanges)} requests came through, then none.")
        return passed()

    @check(
        REQUIRED,
        id="request.tls-untrusted",
        title="A certificate no CA vouches for is refused",
        description=(
            "In the same scenario, `untrusted`: the agent must break off the"
            " handshake and send no request to a server whose certificate no CA it"
            " trusts. Sending one means it would hand its API key to anyone in the"
            " middle."
        ),
        variants=("untrusted",),
    )
    def refused(self, run):
        if run.exchanges:
            return failed("sent its key", f"{len(run.exchanges)} requests went to an untrusted server.")
        if not run.handshakes_failed:
            return skipped("The agent never connected.")
        return passed(detail=f"Handshake refused: {', '.join(sorted(set(run.handshakes_failed)))}.")
