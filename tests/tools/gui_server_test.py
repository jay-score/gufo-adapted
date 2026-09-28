"""Local control routes enforce origin/token checks and validate launches."""

from pathlib import Path
import re
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
from launcher_config import DEFAULTS
from launcher_process import ProcessManager
from server import create_app


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "settings.json"
        self.manager = Mock(spec=ProcessManager)
        self.manager.snapshot.return_value = {"state": "stopped"}
        self.manager.start.return_value = {"state": "loading"}
        self.manager.stop.return_value = {"state": "stopped"}
        self.exiting = threading.Event()
        self.app = create_app(self.config, self.manager, self.exiting)
        self.client = self.app.test_client()
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        token = re.search(r'name="gufo-token" content="([^"]+)"', page.text)[1]
        self.headers = {"X-Gufo-Token": token}

    def test_control_and_file_routes_require_same_origin_and_runtime_token(self):
        for method, path in (("get", "/api/browse"), ("get", "/api/settings"),
                             ("post", "/api/start"), ("post", "/api/stop"), ("post", "/api/exit")):
            for headers in ({}, self.headers | {"Origin": "https://example.com"},
                            {"X-Gufo-Token": "stale token"}):
                with self.subTest(path=path, headers=headers):
                    response = getattr(self.client, method)(path, headers=headers)
                    self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/", headers={"Host": "attacker.example"}).status_code, 400)
        self.assertIn("frame-ancestors 'none'", self.client.get("/").headers["Content-Security-Policy"])
        self.manager.start.assert_not_called()
        self.manager.stop.assert_not_called()

    def test_save_restore_preview_and_browse(self):
        settings = DEFAULTS | {"context": 8192, "models_dir": str(self.root)}
        saved = self.client.post("/api/settings", json=settings, headers=self.headers)
        self.assertEqual(saved.status_code, 200)
        self.assertTrue(self.config.is_file())
        app = create_app(self.config, self.manager, self.exiting)
        client = app.test_client()
        token = re.search(r'name="gufo-token" content="([^"]+)"', client.get("/").text)[1]
        self.assertEqual(client.get("/api/settings", headers={"X-Gufo-Token": token}).json["settings"], settings)
        preview = self.client.post("/api/preview", json=settings, headers=self.headers)
        self.assertIn("--context 8192", preview.json["command"])
        (self.root / "model.gguf").touch()
        files = self.client.get("/api/browse", query_string={"path": str(self.root)}, headers=self.headers)
        self.assertEqual(files.json["entries"][0]["name"], "model.gguf")
        invalid = self.client.post("/api/settings", json=settings | {"temperature": None}, headers=self.headers)
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(self.client.post("/api/settings", data="x" * 40000,
                                         content_type="application/json", headers=self.headers).status_code, 413)

    def test_launch_uses_explicit_argv_and_does_not_overwrite_saved_settings(self):
        exe, model, mtp = [self.root / name for name in ("gufo.exe", "main & model.gguf", "mtp.gguf")]
        exe.touch()
        model.write_bytes(b"GGUF")
        mtp.write_bytes(b"GGUF")
        settings = DEFAULTS | {"executable": str(exe), "model": str(model),
                               "mtp": True, "mtp_model": str(mtp), "prompt_lookup": True}
        started = self.client.post("/api/start", json=settings, headers=self.headers)
        self.assertEqual(started.status_code, 200)
        command, port, name = self.manager.start.call_args.args
        self.assertEqual(command[command.index("--model") + 1], str(model))
        self.assertIn("--prompt-lookup", command)
        self.assertEqual((port, name), (8080, "gufo"))
        self.assertFalse(self.config.exists())
        invalid = self.client.post("/api/start", json=settings | {"port": 8090}, headers=self.headers)
        self.assertEqual(invalid.status_code, 400)
        model.unlink()
        invalid = self.client.post("/api/start", json=settings, headers=self.headers)
        self.assertEqual(invalid.status_code, 400)
        self.client.post("/api/stop", headers=self.headers)
        self.manager.stop.assert_called_once()

    def test_corrupt_settings_are_reported_and_preserved_until_explicit_save(self):
        self.config.write_text("broken")
        app = create_app(self.config, self.manager, self.exiting)
        client = app.test_client()
        token = re.search(r'name="gufo-token" content="([^"]+)"', client.get("/").text)[1]
        response = client.get("/api/settings", headers={"X-Gufo-Token": token})
        self.assertIn("Cannot read saved settings", response.json["warning"])
        self.assertEqual(self.config.read_text(), "broken")

    def test_exit_stops_owned_process_and_rejects_a_racing_start(self):
        response = self.client.post("/api/exit", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.manager.stop.assert_called_once()
        self.assertEqual(self.client.post("/api/start", json=DEFAULTS, headers=self.headers).status_code, 503)
        self.assertTrue(self.exiting.wait(1))

    def test_exit_failure_leaves_launcher_available_for_retry(self):
        self.manager.stop.return_value = {"state": "failed", "pid": 123,
                                          "error": "Gufo could not be stopped; try Stop again."}
        response = self.client.post("/api/exit", headers=self.headers)
        self.assertEqual(response.status_code, 503)
        self.assertIn("try Stop again", response.json["error"])
        self.assertFalse(self.exiting.is_set())
        self.assertEqual(self.client.get("/api/status", headers=self.headers).status_code, 200)
        self.manager.stop.return_value = {"state": "stopped"}
        self.assertEqual(self.client.post("/api/exit", headers=self.headers).status_code, 200)
        self.assertTrue(self.exiting.wait(1))


if __name__ == "__main__":
    unittest.main()
