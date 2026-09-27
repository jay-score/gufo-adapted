"""Exercise real child processes without loading a model or requiring a GPU."""

from pathlib import Path
import socket
import sys
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
from launcher_process import ProcessManager

# Windows venv python.exe is a redirector that spawns the real interpreter.
# Exercise the same single-process ownership as gufo.exe, without that wrapper.
PYTHON = getattr(sys, "_base_executable", sys.executable)


def unused_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


FAKE_SERVER = """
import json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
for i in range(500):
    print('log row', i, flush=True)
print('stderr is captured', file=sys.stderr, flush=True)
time.sleep(0.6)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({'status': 'ready', 'model': 'fixture'}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


class ProcessTest(unittest.TestCase):
    def setUp(self):
        self.manager = ProcessManager()
        self.addCleanup(self.manager.stop)

    def wait_for(self, state):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            snapshot = self.manager.snapshot()
            if snapshot["state"] == state:
                return snapshot
            time.sleep(0.05)
        self.fail(f"Expected {state}, got {self.manager.snapshot()}")

    def test_loading_ready_logs_duplicate_start_stop_and_relaunch(self):
        port = unused_port()
        argv = [PYTHON, "-u", "-c", FAKE_SERVER, str(port)]
        self.assertEqual(self.manager.start(argv, port, "fixture")["state"], "loading")
        with self.assertRaisesRegex(ValueError, "Stop the running"):
            self.manager.start(argv, port, "fixture")
        ready = self.wait_for("ready")
        self.assertEqual(ready["model_name"], "fixture")
        self.assertEqual(len(ready["logs"]), 300)
        self.assertIn("stderr is captured", ready["logs"])
        child = self.manager.process
        self.assertEqual(self.manager.stop()["state"], "stopped")
        self.assertIsNotNone(child.poll())
        self.manager.start(argv, port, "fixture")
        self.wait_for("ready")

    def test_start_failure_and_early_exit_are_visible(self):
        with self.assertRaisesRegex(ValueError, "Could not launch"):
            self.manager.start([str(Path.cwd() / "missing-gufo-executable")], unused_port(), "fixture")
        self.assertEqual(self.manager.snapshot()["state"], "failed")
        self.manager.start([PYTHON, "-c", "import sys; print('bad model', flush=True); sys.exit(7)"],
                           unused_port(), "fixture")
        failed = self.wait_for("failed")
        self.assertEqual(failed["exit_code"], 7)
        self.assertIn("code 7", failed["error"])

    def test_occupied_port_is_not_mistaken_for_our_server(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaisesRegex(ValueError, "already in use"):
                self.manager.start([PYTHON], listener.getsockname()[1], "fixture")
            self.assertIsNone(self.manager.process)

    def test_stop_while_model_is_loading(self):
        self.manager.start([PYTHON, "-c", "import time; time.sleep(30)"], unused_port(), "fixture")
        self.assertEqual(self.manager.stop()["state"], "stopped")
        self.assertIsNotNone(self.manager.process.poll())


if __name__ == "__main__":
    unittest.main()
