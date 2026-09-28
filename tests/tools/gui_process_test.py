"""Exercise real child processes without loading a model or requiring a GPU."""

from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
from launcher_process import ProcessManager, parse_metrics

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
    metrics_requests = 0
    def do_GET(self):
        if self.path == '/metrics':
            Handler.metrics_requests += 1
            # Exercise an unavailable scrape and automatic recovery.
            if Handler.metrics_requests == 2:
                self.send_response(503)
                self.end_headers()
                return
            speed = 25 if Handler.metrics_requests == 1 else 30
            body = ('# TYPE llamacpp:prompt_tokens_seconds gauge\\n'
                    'llamacpp:prompt_tokens_seconds 800\\n'
                    f'llamacpp:predicted_tokens_seconds {speed}\\n'
                    'llamacpp:prompt_tokens_total 100\\n'
                    'llamacpp:tokens_predicted_total 20\\n').encode()
        else:
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

    def test_metrics_poll_failure_recovery_and_restart(self):
        port = unused_port()
        argv = [PYTHON, "-u", "-c", FAKE_SERVER, str(port)]
        self.assertIsNone(self.manager.start(argv, port, "fixture")["metrics"])
        self.wait_for("ready")
        for expected_speed in (25, None, 30):
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                snapshot = self.manager.snapshot()
                metrics = snapshot["metrics"]
                if (metrics and metrics["generation_tps"] == expected_speed) or (metrics is None and expected_speed is None):
                    break
                time.sleep(0.05)
            else:
                self.fail(f"Expected speed {expected_speed}, got {metrics}")
            self.assertEqual(snapshot["state"], "ready")
            if metrics:
                self.assertEqual(metrics["prefill_tps"], 800)
                self.assertEqual(metrics["prompt_tokens_total"], 100)
                self.assertEqual(metrics["generated_tokens_total"], 20)
        self.assertIsNone(self.manager.stop()["metrics"])
        self.assertIsNone(self.manager.start(argv, port, "fixture")["metrics"])

    def test_metrics_parser_rejects_missing_and_nonfinite_values(self):
        text = ("# HELP ignored comment\n"
                "llamacpp:prompt_tokens_seconds 8e2\n"
                "llamacpp:predicted_tokens_seconds 25.5\n"
                "llamacpp:prompt_tokens_total 100\n"
                "llamacpp:tokens_predicted_total 0\n"
                "llamacpp:kv_cache_usage_ratio 0.0\n")
        self.assertEqual(parse_metrics(text), {"prefill_tps": 800, "generation_tps": 25.5,
                                             "prompt_tokens_total": 100, "generated_tokens_total": 0})
        for invalid in ("", "{}", text.replace("25.5", ""),
                        *(text.replace("25.5", value) for value in ("NaN", "+Inf", "-1", "bad"))):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_metrics(invalid)

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

    def test_failed_stop_keeps_child_owned_and_can_be_retried(self):
        self.manager.start([PYTHON, "-c", "import time; time.sleep(30)"], unused_port(), "fixture")
        child = self.manager.process
        target = self.manager.job if os.name == "nt" else child
        with patch.object(target, "terminate", side_effect=OSError("fixture termination failure")):
            failed = self.manager.stop()
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["pid"], child.pid)
        self.assertIn("try Stop again", failed["error"])
        self.assertEqual(self.manager.stop()["state"], "stopped")
        self.assertIsNotNone(child.poll())

    @unittest.skipUnless(os.name == "nt", "Windows Job Object lifetime")
    def test_failed_job_assignment_does_not_leave_a_child_running(self):
        from windows_job import WindowsJob
        children = []

        def fail_assignment(job, child):
            children.append(child)
            raise OSError("fixture job assignment failure")

        with patch.object(WindowsJob, "assign", fail_assignment):
            with self.assertRaisesRegex(ValueError, "Could not launch"):
                self.manager.start([PYTHON, "-c", "import time; time.sleep(30)"], unused_port(), "fixture")
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertIsNone(self.manager.job)
        self.assertEqual(self.manager.snapshot()["state"], "failed")

    @unittest.skipUnless(os.name == "nt", "Windows Job Object lifetime")
    def test_owned_tree_exits_on_stop_and_abrupt_launcher_exit(self):
        import ctypes
        from ctypes import wintypes
        from windows_job import kernel32

        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateProcess.restype = wintypes.BOOL

        def close_process(handle):
            if kernel32.WaitForSingleObject(handle, 0) == 258:
                kernel32.TerminateProcess(handle, 1)
                kernel32.WaitForSingleObject(handle, 5000)
            kernel32.CloseHandle(handle)

        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "tree.json"
            # Wait for assignment before creating descendants: simulate Gufo
            # starting helper processes after its model has begun loading.
            gate = Path(directory) / "assigned"
            child_code = (
                "import json, os, pathlib, subprocess, sys, time\n"
                f"while not pathlib.Path({str(gate)!r}).exists(): time.sleep(0.02)\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(), child.pid]))\n"
                "time.sleep(60)\n"
            )
            for abrupt in (False, True):
                with self.subTest(abrupt=abrupt):
                    marker.unlink(missing_ok=True)
                    gate.unlink(missing_ok=True)
                    helper = None
                    try:
                        if abrupt:
                            helper_code = (
                                f"import pathlib, sys, time; sys.path.insert(0, {str(Path(__file__).resolve().parents[2] / 'tools/gui')!r})\n"
                                "from launcher_process import ProcessManager\n"
                                "manager = ProcessManager()\n"
                                f"manager.start({[PYTHON, '-c', child_code]!r}, {unused_port()}, 'fixture')\n"
                                f"pathlib.Path({str(gate)!r}).touch()\n"
                                "time.sleep(60)\n"
                            )
                            helper = subprocess.Popen([PYTHON, "-c", helper_code],
                                                      creationflags=subprocess.CREATE_NO_WINDOW)
                        else:
                            self.manager.start([PYTHON, "-c", child_code], unused_port(), "fixture")
                            gate.touch()
                        deadline = time.monotonic() + 8
                        while not marker.exists() and time.monotonic() < deadline:
                            time.sleep(0.05)
                        pids = json.loads(marker.read_text())
                        handles = []
                        for pid in pids:
                            handle = kernel32.OpenProcess(0x100001, False, pid)  # SYNCHRONIZE | TERMINATE
                            self.assertTrue(handle, str(ctypes.WinError()))
                            self.addCleanup(close_process, handle)
                            handles.append(handle)
                        if abrupt:
                            helper.kill()  # No finally, signal handler or manager.stop().
                            helper.wait(timeout=5)
                        else:
                            self.assertEqual(self.manager.stop()["state"], "stopped")
                        for handle in handles:
                            self.assertEqual(kernel32.WaitForSingleObject(handle, 5000), 0,
                                             "Owned child survived launcher shutdown")
                    finally:
                        if helper is not None and helper.poll() is None:
                            helper.kill()
                            helper.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
