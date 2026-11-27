"""What a check concludes, and how strict it is.

Each check has a level, written as a rule about the agent:

- required: the agent must do it. Pass or fail.
- recommended: the agent should do it. Pass or gap; a gap is a missing
  safeguard, not a failure.
- optional: the agent may do it. If it does, it must work: pass, fail, or
  unsupported, which is neutral.
- info: there is no right answer. A short label says what the agent does.

Any check can also be unsupported, when the agent or its recipe lacks what
the case needs; skipped (not measured), when the run did not reach what the
check judges; or hit a suite error.
"""

from dataclasses import dataclass, replace

REQUIRED, RECOMMENDED, OPTIONAL, INFO = "required", "recommended", "optional", "info"
LEVELS = (REQUIRED, RECOMMENDED, OPTIONAL, INFO)
ALLOWED = {
    REQUIRED: {"pass", "fail"},
    RECOMMENDED: {"pass", "gap"},
    OPTIONAL: {"pass", "fail", "unsupported"},
    INFO: {"label"},
}


@dataclass(frozen=True)
class Outcome:
    result: str  # pass, fail, gap, unsupported, label, skip, error
    summary: str  # Short: what the grid shows.
    detail: str = ""  # Why, in a sentence or two.


def passed(summary: str = "pass", detail: str = "") -> Outcome:
    return Outcome("pass", summary, detail)


def described(outcome: Outcome, how: Outcome) -> Outcome:
    """A pass that says what the agent did, with the summary and detail of
    `how`; any other outcome as it is."""
    return passed(how.summary, how.detail) if outcome.result == "pass" else outcome


def failed(summary: str, detail: str = "") -> Outcome:
    return Outcome("fail", summary, detail)


def gap(summary: str, detail: str = "") -> Outcome:
    return Outcome("gap", summary, detail)


def demoted(outcome: Outcome) -> Outcome:
    """A required check's finding as a recommended check gives it: a fail
    becomes a gap, and any other outcome stays as it is."""
    return replace(outcome, result="gap") if outcome.result == "fail" else outcome


def unsupported(summary: str = "not supported", detail: str = "") -> Outcome:
    return Outcome("unsupported", summary, detail)


def label(summary: str, detail: str = "") -> Outcome:
    return Outcome("label", summary, detail)


def skipped(reason: str) -> Outcome:
    return Outcome("skip", "not measured", reason)


def suite_error(summary: str, detail: str = "") -> Outcome:
    return Outcome("error", summary, detail)


@dataclass(frozen=True)
class CheckInfo:
    """One conclusion that a scenario's run supports."""

    id: str
    level: str
    title: str
    description: str
    variants: tuple[str, ...] | None = None  # None: every variant of the scenario.
    apis: tuple[str, ...] | None = None  # None: every API of the scenario.

    def applies(self, variant: str, api: str | None = None) -> bool:
        """Whether the check judges this variant, on this API if one is given."""
        return (self.variants is None or variant in self.variants) and (
            api is None or self.apis is None or api in self.apis
        )


def check(
    level: str,
    id: str = "",
    title: str = "",
    description: str = "",
    variants: tuple[str, ...] | None = None,
    apis: tuple[str, ...] | None = None,
):
    """Mark a scenario method `judge(self, run) -> Outcome` as a check. The id,
    title and description default to the scenario's."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level}")

    def mark(judge):
        judge.check = (level, id, title, description, variants, apis)
        return judge

    return mark


def conform(level: str, outcome: Outcome) -> Outcome:
    """Raise if a judge returns a result its level does not allow: that is a
    bug in the suite."""
    if outcome.result in ("skip", "unsupported", "error") or outcome.result in ALLOWED[level]:
        return outcome
    raise ValueError(f"a {level} check cannot conclude {outcome.result!r}")
