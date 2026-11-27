"""Recipes turn settings into a launch."""

import json
import tempfile
import unittest
from pathlib import Path

from harness_suite import recipe

VARIABLES = {
    "home": "/home",
    "workspace": "/workspace",
    "prompt_file": "/home/prompt.txt",
    "provider_url": "http://127.0.0.1:1",
    "model": "suite-model",
    "model_b": "suite-model-b",
    "context_window": 1000,
    "max_output_tokens": 100,
    "python": "python3",
    "fixture": "/suite/fixture.py",
    "fixture_dir": "/workspace/.suite",
    "free_port": 40000,
}


class Settings(unittest.TestCase):
    def test_settings_merge_into_a_json_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "agent.toml")
            path.write_text(
                """
name = "agent"
command = ["agent"]
[apis.chat]
[[files]]
path = "${home}/config.json"
json = { model = "${model}", agent = { build = { tools = true } } }
[[settings.temperature.files]]
path = "${home}/config.json"
json = { agent = { build = { temperature = "${value}" } } }
"""
            )
            r = recipe.load(str(path))
            files = r.prepare(VARIABLES, "chat", "none", {"temperature": 0.3}, "hi", True).files
        self.assertEqual(len(files), 1)
        written = json.loads(files[0][1])
        self.assertEqual(written, {"model": "suite-model", "agent": {"build": {"tools": True, "temperature": 0.3}}})


class About(unittest.TestCase):
    def test_every_recipe_says_where_its_releases_are(self):
        for name in recipe.available():
            with self.subTest(name):
                about = recipe.load(name).about
                self.assertTrue(about["title"])
                self.assertRegex(about["repo"] or "", r"^https://github\.com/[\w.-]+/[\w.-]+$")


class Servers(unittest.TestCase):
    def test_each_server_gets_its_own_entry(self):
        """Every recipe writes each MCP server it connects with its own name
        and address, so two servers never land on one entry."""
        for name in recipe.available():
            r = recipe.load(name)
            for transport in recipe.TRANSPORTS:
                if r.transport(transport) is not True:
                    continue
                servers = [
                    {"server": server, "transport": transport, "server_url": f"http://127.0.0.1:{port}/mcp"}
                    for server, port in (("alpha", 1001), ("beta", 1002))
                ]
                with self.subTest(name=name, transport=transport):
                    prepared = r.prepare(VARIABLES, r.apis[0], "mcp", {}, "hi", True, servers=servers)
                    seen = json.dumps([prepared.argv, prepared.files, prepared.environment])
                    for server in servers:
                        self.assertIn(server["server"], seen)
                        if transport != "stdio":
                            self.assertIn(server["server_url"], seen)


if __name__ == "__main__":
    unittest.main()
