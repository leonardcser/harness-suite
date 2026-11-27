"""Run real scenarios against the fake agent, clean and with known defects."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness_suite import engine, limits, recipe, runner
from harness_suite.results import REQUIRED
from harness_suite.sandbox import Host
from harness_suite.scenarios import all_checks, all_scenarios

FAKE = Path(__file__).with_name("fake_harness.py")


def fake(*defects: str) -> recipe.Recipe:
    clean = recipe.load(str(FAKE.with_name("fake.toml")))
    data = copy.deepcopy(clean.data)
    data["command"] = [sys.executable, str(FAKE)]
    data["args"] += defects
    data["tui"]["args"] += defects
    return recipe.Recipe(clean.path, data)


def results(
    check_id: str,
    variant: str,
    *defects: str,
    frontend: str = "headless",
    source: str | None = None,
    api: str = "chat",
) -> dict[str, dict]:
    """Every row of the case that judges `check_id`, by check. A case with
    tools uses the shell tool where the scenario allows, unless `source` says."""
    scenario = next(s for info, s in all_checks() if info.id == check_id)
    if source is None:
        source = ("native" if "native" in scenario.sources else "mcp") if scenario.uses_tools(variant) else "none"
    with tempfile.TemporaryDirectory() as folder:
        case = scenario(variant, api, source, fake(*defects), frontend)
        rows = engine.case(case, lambda: Host([]), Path(folder))
    return {row["check"]: row for row in rows}


def outcome(check_id: str, variant: str, *defects: str, check: str | None = None, **options) -> dict:
    """The row of `check`, by default `check_id`, of the case that judges `check_id`."""
    return results(check_id, variant, *defects, **options)[check or check_id]


# The clean agent's summary of each of these checks, in every case of it, or
# in the cases that `where` names by API, tools and frontend.
SUMMARIES = [
    ("errors.final", "quota-later", "stopped after 2", {"api": ("chat", "responses")}),
    ("errors.final", "quota-later", "stopped at once", {"api": ("anthropic",)}),
    ("errors.final", "model", "stopped at once"),
    ("settings.temperature", "default", "pass"),
    ("mcp.instructions", "default", "left out"),
    ("mcp.structured", "default", "text"),
    ("mcp.list-changed", "default", "pass"),
    ("mcp.sampling", "default", "not supported"),
    *(("media.read-image", f, "pass") for f in ("png", "jpeg", "gif", "webp", "misnamed")),
    *(("media.image", f, "pass") for f in ("png", "jpeg", "gif", "webp", "two-images")),
    ("media.prompt-image", "default", "not attached"),
    *(("media.limits", v, "left out") for v in ("wide", "heavy", "bmp")),
    *(("media.content", v, "left out or described") for v in ("image-resource", "blob")),
    ("media.many-images", "default", "kept few"),
    *(("cli.json", v, "JSON lines") for v in ("text", "tools", "failure")),
    ("tools.bad-arguments", "empty", "rejected it"),
    ("tools.output-size", "lines", "16-64 KB, head+tail", {"tools": ("native",)}),
    ("tools.output-size", "lines", "sent whole, rejected", {"tools": ("mcp",)}),
    ("tools.subagent", "default", "own task, same model"),
    ("tools.subagent-error", "default", "told in the result"),
    ("context.instructions-nested", "default", "never"),
    ("context.instructions-large", "default", "whole"),
    ("session.crash", "default", "pass"),
    ("reasoning.history", "same-model", "kept", {"api": ("chat",)}),
    ("reasoning.history", "same-model", "kept, signed", {"api": ("responses", "anthropic")}),
    ("cache.tool-loop", "default", "stable"),
    ("cache.turns", "default", "stable"),
    ("cache.marked", "default", "marked"),
    ("cache.key", "default", "one per session"),
    ("shell.exit-status", "default", "status shown"),
    ("shell.cd", "default", "reset to workspace"),
    ("shell.flood", "default", "16-64 KB, head+tail"),
    ("shell.terminal", "prompt", "no terminal"),
    ("files.read", "large", "part"),
    ("files.read", "latin-1", "replaced"),
    *(("files.read", v, "no file content") for v in ("missing", "directory", "symlink-loop")),
    ("files.edit", "twice", "refused"),
    ("files.write-guarded", "default", "kept"),
    ("files.edit-stale", "default", "applied"),
    ("cli.open-stdin", "default", "ignores stdin"),
    ("tools.repeat", "default", "ran each"),
    ("stream.empty-reply", "empty", "exit status"),
    ("workspace.mcp-config", "default", "pass"),
]
# What the clean agent lacks: its recipe configures no effort setting.
NOT_MEASURED: set[str] = set()
UNSUPPORTED = {"mcp.sampling", "session.switch-effort", "settings.effort"}


class EveryCase(unittest.TestCase):
    """A harness needs only its recipe: every case of every scenario runs on
    the clean fake agent, on each API, headless and in its TUI, in one run."""

    @classmethod
    def setUpClass(cls):
        scenarios = list(dict.fromkeys(scenario for _, scenario in all_checks()))
        with tempfile.TemporaryDirectory() as folder:
            cls.rows = runner.run(Path(folder, "run"), [fake()], scenarios, host=True, jobs=limits.jobs())

    def test_nothing_fails(self):
        bad = [r for r in self.rows if r["result"] in ("error", "fail", "gap")]
        self.assertEqual(bad, [], "\n".join(f"{r['case']} {r['check']}: {r['summary']} {r['detail']}" for r in bad))

    def test_every_check_is_reported(self):
        self.assertEqual({r["check"] for r in self.rows}, {info.id for info, _ in all_checks()})

    def test_only_what_it_lacks_is_unmeasured(self):
        for result, wanted in (("skip", NOT_MEASURED), ("unsupported", UNSUPPORTED)):
            with self.subTest(result):
                rows = [r for r in self.rows if r["result"] == result]
                self.assertEqual({r["check"] for r in rows}, wanted)
                self.assertEqual([r["case"] for r in rows if not r["detail"]], [], "no reason given")

    def test_summaries(self):
        for check, variant, summary, *where in SUMMARIES:
            where = where[0] if where else {}
            with self.subTest(check=check, variant=variant, **{k: "/".join(v) for k, v in where.items()}):
                rows = [
                    r
                    for r in self.rows
                    if r["check"] == check and r["variant"] == variant and all(r[k] in v for k, v in where.items())
                ]
                self.assertTrue(rows, "no such case")
                found = {f"{r['api']}/{r['tools']}/{r['frontend']}": r["summary"] for r in rows}
                self.assertEqual(set(found.values()), {summary}, found)


class IdleAgent(unittest.TestCase):
    """An agent that does nothing passes nothing: each case first proves that
    what it judges happened. Only a refusal at once passes on `false`."""

    def test_passes_nothing(self):
        scenarios = list(dict.fromkeys(scenario for _, scenario in all_checks()))
        clean = fake()
        for command, allowed in ((["true"], set()), (["false"], {"cli.empty-prompt", "cli.usage-error"})):
            with self.subTest(command[0]), tempfile.TemporaryDirectory() as folder:
                data = copy.deepcopy(clean.data)
                data["command"] = command
                rows = runner.run(
                    Path(folder, "run"), [recipe.Recipe(clean.path, data)], scenarios, host=True, jobs=limits.jobs()
                )
                passing = [r for r in rows if r["result"] in ("pass", "label") and r["check"] not in allowed]
                self.assertEqual(passing, [], "\n".join(f"{r['case']} {r['check']}: {r['summary']}" for r in passing))


class Quick(unittest.TestCase):
    def test_each_required_variant_runs_once_with_the_agents_own_tools(self):
        cases = runner.quick_cases(runner.plan([fake()], all_scenarios(), frontends=["headless"]))
        found = [(info.id, c.variant) for c in cases for info, _ in c.checks() if info.level == REQUIRED]
        wanted = {
            (info.id, v)
            for info, s in all_checks()
            if info.level == REQUIRED and s.id
            for v in s.variants
            if info.applies(v)
        }
        self.assertEqual({key for key in found if key in wanted}, wanted)
        self.assertEqual(len(cases), len({(type(c), c.variant) for c in cases}))
        for case in cases:
            # The first API and the agent's own tools, unless the variant cannot run there.
            if case.api != "chat":
                self.assertFalse(type(case)(case.variant, "chat", case.source, case.recipe).fits(), case.id)
            if case.source == "mcp" and "native" in case.sources:
                self.assertFalse(type(case)(case.variant, case.api, "native", case.recipe).fits(), case.id)


class TuiAgent(unittest.TestCase):
    """The same fake agent, driven through its TUI in a terminal."""

    def test_prompt_lost_while_starting_or_shown_as_a_placeholder_is_sent(self):
        for defect in ("--slow-input", "--lose-first-prompt", "--collapse-paste"):
            with self.subTest(defect):
                self.assertEqual(outcome("session.continue", "same-model", defect, frontend="tui")["result"], "pass")

    def test_paste_cut_at_a_limit(self):
        row = outcome("request.paste", "long", "--paste-limit", frontend="tui")
        self.assertEqual((row["result"], row["summary"]), ("gap", "cut"), row)

    def test_command_reads_the_agents_terminal(self):
        row = outcome("shell.terminal", "prompt", "--share-terminal", frontend="tui")
        self.assertEqual(row["summary"], "terminal open", row)

    def test_paste_that_only_the_quit_command_sends(self):
        rows = results("request.paste", "lines", "--paste-needs-typing", frontend="tui")
        self.assertEqual((rows["request.paste"]["result"], rows["request.paste"]["summary"]), ("gap", "not sent"), rows)

    def test_dropped_reasoning_is_rejected(self):
        rows = results("reasoning.tool-loop", "shown", "--drop-reasoning", frontend="tui")
        self.assertEqual(rows["reasoning.tool-loop"]["summary"], "rejected by the API", rows)
        self.assertEqual(rows["request.valid"]["result"], "pass", rows)


class EveryApi(unittest.TestCase):
    """The checks that each API reads in its own way, on Responses and Anthropic."""

    def test_anthropic_defects(self):
        cases = [
            ("reasoning.tool-loop", "shown", "--drop-reasoning"),
            ("reasoning.tool-loop", "shown", "--strip-signatures"),
            ("reasoning.cut", "default", "--strip-signatures"),
        ]
        rows = results("reasoning.late-encrypted", "late-encrypted", "--early-reasoning-only", api="responses")
        self.assertEqual(rows["reasoning.late-encrypted"]["result"], "gap", rows)
        self.assertEqual(rows["reasoning.late-encrypted"]["summary"], "rejected by the API", rows)
        self.assertEqual(rows["request.valid"]["result"], "pass", rows)
        for check, variant, defect in cases:
            with self.subTest(defect):
                rows = results(check, variant, defect, api="anthropic")
                self.assertEqual(rows[check]["summary"], "rejected by the API", rows)
                self.assertEqual(rows["request.valid"]["result"], "pass", rows)

    def test_provoked_request_judged_by_its_own_check(self):
        cases = [
            ("tools.reused-id", "default", "--keep-ids"),
            ("tools.round-trip", "blank-text", "--keep-blank-text"),
            ("tools.unknown-name", "long-name", "--keep-long-names"),
        ]
        for check, variant, defect in cases:
            with self.subTest(defect):
                rows = results(check, variant, defect, api="anthropic")
                self.assertEqual(rows[check]["summary"], "rejected by the API", rows)
                self.assertEqual(rows["request.valid"]["result"], "pass", rows)

    def test_cache_marks(self):
        row = outcome("cache.tool-loop", "default", "--no-cache-marks", api="anthropic", check="cache.marked")
        self.assertEqual((row["result"], row["summary"]), ("gap", "no marks"), row)

    def test_repository_code(self):
        row = outcome("workspace.repo-code", "default", "--git-status")
        self.assertEqual((row["result"], row["summary"]), ("gap", "ran it"), row)

    def test_repository_mcp_config(self):
        row = outcome("workspace.mcp-config", "default", "--trust-workspace")
        self.assertEqual((row["result"], row["summary"]), ("gap", "started it"), row)

    def test_prompt_cache_key(self):
        row = outcome("cache.tool-loop", "default", "--cache-key-per-request", check="cache.key")
        self.assertEqual((row["result"], row["summary"]), ("gap", "changes"), row)


class DefectiveAgent(unittest.TestCase):
    def assertCaught(
        self, check: str, variant: str, defect: str, result: str = "fail", summary: str | None = None, **options
    ):
        row = outcome(check, variant, defect, **options)
        self.assertEqual(row["result"], result, row)
        if summary:
            self.assertEqual(row["summary"], summary, row)

    def test_banner_line_ends_the_reader(self):
        self.assertCaught("mcp.framing", "banner", "--mcp-strict-lines", result="gap", source="mcp")

    def test_mcp_client_without_capabilities(self):
        self.assertCaught("mcp.roots", "default", "--mcp-no-capabilities", result="unsupported", source="mcp")

    def test_mcp_tool_names_not_prefixed(self):
        self.assertCaught("mcp.native-names", "default", "--mcp-bare-names", summary="same name twice", source="mcp")

    def test_rich_schema_flattened(self):
        row = outcome("mcp.schema", "default", "--mcp-flat-schema", source="mcp")
        self.assertEqual(row["summary"], "simplified", row)

    def test_certificate_not_checked(self):
        self.assertCaught("request.tls-untrusted", "untrusted", "--insecure", summary="sent its key")

    def test_company_ca_ignored(self):
        self.assertCaught("request.tls", "trusted", "--ignore-ca-env", result="gap", summary="CA not used")

    def test_calls_dropped_without_their_finish_reason(self):
        self.assertCaught("stream.call-chunks", "finish-stop", "--calls-need-finish", result="gap", summary="call lost")

    def test_every_mcp_tool_offered_past_the_chat_limit(self):
        self.assertCaught(
            "mcp.many-tools", "default", "--all-tools", result="gap", summary="rejected by the API", source="mcp"
        )

    def test_line_reader_with_a_size_limit(self):
        self.assertCaught("stream.framing", "long-line", "--short-lines", result="fail")

    def test_lone_surrogate_passed_on(self):
        rows = results("tools.unicode", "lone-surrogate", "--keep-surrogates", source="mcp")
        self.assertEqual(rows["tools.unicode"]["summary"], "rejected by the API", rows)
        self.assertIn("surrogate", rows["tools.unicode"]["detail"], rows)
        self.assertEqual(rows["request.valid"]["result"], "pass", rows)

    def test_azure_filter_chunk_rejected(self):
        row = outcome("stream.azure-filter", "azure-filter", "--need-choices", check="stream.azure-filter", api="chat")
        self.assertEqual(row["result"], "gap", row)

    def test_terminal_escapes_passed_on(self):
        row = outcome("tools.terminal-escapes", "default", "--raw-results", frontend="tui")
        self.assertEqual((row["result"], row["summary"]), ("gap", "passed on"), row)

    def test_exit_status_hidden(self):
        self.assertCaught("shell.exit-status", "default", "--hide-status", result="gap", summary="status hidden")

    def test_failing_flood_cut_at_its_head(self):
        self.assertCaught("shell.flood-status", "failing", "--head-only", result="gap", summary="lost the last line")

    def test_broken_config_reported_as_success(self):
        self.assertCaught("settings.broken-file", "default", "--config-exit-zero", result="gap", summary="exit 0")

    def test_unknown_option_ignored(self):
        self.assertCaught("cli.usage-error", "default", "--lax-options", result="gap", summary="ran the task")

    def test_empty_prompt_sent(self):
        self.assertCaught(
            "cli.empty-prompt", "whitespace", "--allow-empty-prompt", result="gap", summary="sent a request"
        )

    def test_waiting_for_stdin_to_close(self):
        self.assertCaught("cli.open-stdin", "default", "--read-stdin", result="label", summary="waits for stdin")

    def test_reply_that_ended_early_taken_as_done(self):
        for variant in ("cut", "length"):
            with self.subTest(variant):
                self.assertCaught("stream.ended-early", variant, "--trust-early-end", result="gap", summary="exit 0")
        self.assertCaught("stream.empty-reply", "empty", "--trust-early-end", result="label", summary="exit 0")

    def test_dropped_result(self):
        rows = results("tools.round-trip", "text", "--drop-result")
        self.assertEqual(rows["tools.round-trip"]["result"], "fail", rows)
        self.assertEqual(rows["request.valid"]["summary"], "invalid request", rows)

    def test_double_execution(self):
        self.assertCaught("tools.round-trip", "text", "--run-twice", summary="ran twice")

    def test_clipped_small_result(self):
        self.assertCaught("tools.round-trip", "text", "--clip", summary="clipped")

    def test_crash_on_binary_output(self):
        self.assertCaught("shell.binary", "default", "--strict-decode", summary="no result")

    def test_wrong_model(self):
        self.assertCaught("request.model", "plain", "--wrong-model", summary="wrong model")

    def test_changed_input(self):
        self.assertCaught("request.input", "whitespace", "--mangle-input", summary="changed")

    def test_running_a_broken_call(self):
        self.assertCaught("tools.incomplete", "malformed-json", "--run-broken", summary="ran broken call")

    def test_dropped_reasoning_in_the_tool_loop(self):
        self.assertCaught("reasoning.tool-loop", "shown", "--drop-reasoning", summary="rejected by the API")

    def test_reasoning_in_a_user_message(self):
        row = outcome("reasoning.history", "same-model", "--reasoning-as-user")
        self.assertEqual(row["summary"], "in a user message", row)

    def test_no_retry(self):
        self.assertCaught("errors.retry", "server", "--no-retry", result="gap")

    def test_open_stdin(self):
        self.assertCaught("shell.stdin", "default", "--stdin-pipe", result="gap", summary="timed out")

    def test_child_left_running_after_a_timeout(self):
        self.assertCaught("shell.timeout", "default", "--kill-parent-only", result="gap", summary="child orphaned")

    def test_waiting_for_a_background_child(self):
        self.assertCaught("shell.background", "default", "--wait-eof", result="gap")

    def test_changing_prefix(self):
        self.assertCaught("cache.tool-loop", "default", "--volatile-prefix", result="gap", summary="prefix changed")

    def test_session_saved_mid_turn(self):
        rows = results("session.crash", "default", "--save-early")
        self.assertEqual(rows["session.crash"]["result"], "pass", rows)
        self.assertEqual(rows["request.valid"]["result"], "fail", rows)

    def test_mcp_defects(self):
        self.assertCaught("mcp.handshake", "default", "--mcp-list-early", summary="too early")
        self.assertCaught("mcp.pagination", "default", "--mcp-first-page", result="gap", summary="not found")
        self.assertCaught("mcp.session-expired", "default", "--mcp-keep-session", result="gap")
        for variant in ("bad-name", "not-object"):
            self.assertCaught(
                "mcp.unusable-tool", variant, "--mcp-keep-odd", result="gap", summary="rejected by the API"
            )
        self.assertCaught("mcp.ping", "default", "--mcp-mute", result="gap", summary="no answer")
        self.assertCaught("mcp.roots", "default", "--mcp-mute", summary="no answer")
        self.assertCaught("mcp.same-names", "default", "--mcp-flat-names", summary="wrong server")

    def test_pager_on_a_terminal(self):
        self.assertCaught("shell.terminal", "pager", "--pty", result="gap", summary="pager waited", frontend="tui")

    def test_write_without_folders(self):
        self.assertCaught("files.write", "new-folders", "--no-mkdir", result="gap", summary="not written")

    def test_unclipped_output(self):
        self.assertCaught("shell.flood", "default", "--no-clip", result="gap", summary="over the window")
        row = outcome("tools.output-size", "lines", "--no-clip")
        self.assertEqual(row["summary"], "sent whole, rejected", row)

    def test_reading_a_huge_file_whole(self):
        self.assertCaught("files.read", "large", "--read-whole", result="gap", summary="over the window")

    def test_log_line_in_json_output(self):
        self.assertCaught("cli.json", "text", "--log-to-stdout", result="gap", summary="not JSON")

    def test_many_images_kept(self):
        rows = results("media.many-images", "default", "--keep-all-images", source="mcp", api="anthropic")
        self.assertEqual(rows["media.many-images"]["summary"], "rejected by the API", rows)
        self.assertIn("many-image requests", rows["media.many-images"]["detail"], rows)
        self.assertEqual(rows["request.valid"]["result"], "pass", rows)

    def test_image_read_that_fails(self):
        self.assertCaught("media.read-image", "png", "--image-read-error", summary="text only")

    def test_binary_content_pasted_as_base64(self):
        for variant in ("blob", "audio"):
            with self.subTest(variant):
                self.assertCaught("media.content", variant, "--raw-content", result="gap", summary="raw base64")

    def test_image_the_api_refuses(self):
        rows = results("media.limits", "bmp", "--send-any-image", source="native")
        self.assertEqual(rows["media.limits"]["summary"], "rejected by the API", rows)
        self.assertEqual(rows["request.valid"]["result"], "pass", rows)

    def test_overflow_without_images_is_not_measured(self):
        rows = results("context.overflow", "images", "--no-images", source="mcp")
        self.assertEqual(rows["context.overflow"]["result"], "skip", rows)
        self.assertIn("never filled the window", rows["context.overflow"]["detail"])
        self.assertEqual(rows["context.keeps-fact"]["result"], "skip", rows)

    def test_instructions_lost_in_compaction(self):
        self.assertCaught("context.keeps-instructions", "text", "--forget-rules", result="gap", summary="lost")

    def test_no_compaction(self):
        self.assertCaught("context.overflow", "text", "--no-compact", result="gap", summary="stopped")

    def test_cli_defects(self):
        self.assertCaught("cli.exit-status", "failure", "--exit-zero", result="gap", summary="exit 0")
        self.assertCaught("cli.json", "failure", "--exit-zero", result="gap", summary="exit 0")

    def test_tool_missing_before_the_server_was_listed_is_not_measured(self):
        row = outcome("tools.round-trip", "text", "--mcp-late", source="mcp")
        self.assertEqual(row["result"], "skip", row)
        self.assertIn("before it had listed", row["detail"])

    @unittest.skipUnless(limits.available(), "needs a systemd user manager")
    def test_runaway_agent_dies_at_its_memory_cap(self):
        # Reading /dev/zero to its end never stops; the cap ends the agent alone.
        with mock.patch.object(limits, "AGENT_MEMORY", 256 * 1024**2):
            row = outcome("files.read", "device", "--read-special", "--read-whole")
        self.assertEqual((row["result"], row["summary"]), ("gap", "no result"), row)
        self.assertIn("memory cap of 0.25 GB", row["detail"])

    def test_subagent(self):
        rows = results("tools.subagent", "default", "--subagent-forks")
        self.assertEqual(rows["tools.subagent"]["result"], "pass", rows)
        self.assertEqual(rows["tools.subagent"]["summary"], "whole conversation, same model", rows)
        self.assertCaught("tools.subagent", "default", "--subagent-lost", summary="answer lost")
        row = outcome("tools.subagent", "default", "--subagent-background")
        self.assertEqual(row["result"], "pass", row)
        self.assertIn("in the background", row["detail"])

    def test_failing_subagent(self):
        self.assertCaught("tools.subagent-error", "default", "--subagent-crash", result="gap", summary="no result")
        self.assertCaught("tools.subagent-error", "default", "--subagent-lost", result="gap", summary="failure hidden")

    def test_system_prompt(self):
        self.assertCaught("settings.system-prompt", "default", "--system-as-user", summary="as user text")

    def test_read_only(self):
        for variant in ("write", "shell"):
            with self.subTest(variant):
                self.assertCaught("settings.read-only", variant, "--read-only-ignored", summary="file changed")

    def test_api_key(self):
        row = outcome("request.auth", "default", "--no-key")
        self.assertEqual(row["summary"], "key missing", row)

    def test_answer_on_stdout(self):
        self.assertCaught("cli.answer", "long", "--cut-output", summary="cut or changed")
        self.assertCaught("cli.no-escapes", "unicode", "--color", result="gap", summary="escape codes")

    def test_agent_that_ignores_the_proxy_is_not_measured(self):
        row = outcome("request.proxy", "default", "--no-proxy")
        self.assertEqual(row["result"], "skip", row)

    def test_dropped_base_path(self):
        self.assertCaught("request.base-path", "default", "--drop-path", summary="path dropped")

    def test_file_tool_defects(self):
        self.assertCaught("files.write", "crlf", "--text-mode-write", result="gap", summary="CRLF became LF")
        self.assertCaught("files.edit", "twice", "--edit-first", result="gap", summary="replaced one")
        self.assertCaught("files.edit", "crlf", "--edit-lf", result="gap", summary="CRLF became LF")
        self.assertCaught("files.edit", "crlf-lines", "--edit-lf", result="gap", summary="CRLF became LF")
        self.assertCaught("files.edit-stale", "default", "--edit-stale", result="gap", summary="change lost")


if __name__ == "__main__":
    unittest.main()
