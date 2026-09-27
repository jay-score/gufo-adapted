"""Own one Gufo child, drain its output, and report model readiness."""

from collections import deque
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


class ProcessManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.state = "stopped"
        self.error = ""
        self.exit_code = None
        self.logs = deque(maxlen=300)
        self.command = []
        self.base_url = ""
        self.model_name = ""
        self.started_at = None

    def snapshot(self):
        with self.lock:
            return {"state": self.state, "error": self.error, "exit_code": self.exit_code,
                    "logs": list(self.logs), "command": list(self.command),
                    "base_url": self.base_url, "model_name": self.model_name,
                    "pid": self.process.pid if self.process and self.process.poll() is None else None,
                    "elapsed_seconds": int(time.monotonic() - self.started_at) if self.started_at else 0}

    def start(self, command, port, model_name):
        with self.lock:
            if self.state == "stopping" or (self.process and self.process.poll() is None):
                raise ValueError("Stop the running Gufo server before starting another.")
            # Match Gufo's address reuse so TIME_WAIT does not block a restart.
            # Windows can also allow this bind beside a reusable listener, so
            # check for an accepting server before launching our child.
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise ValueError(f"Port {port} is unavailable or already in use.") from exc
            try:
                connection = socket.create_connection(("127.0.0.1", port), timeout=0.25)
            except OSError:
                pass
            else:
                connection.close()
                raise ValueError(f"Port {port} is already in use. Choose another port.")
            self.logs.clear()
            self.command = list(command)
            self.base_url = f"http://127.0.0.1:{port}/v1"
            self.model_name = model_name
            self.error = ""
            self.exit_code = None
            self.started_at = time.monotonic()
            self.process = None  # Retire any observer from an earlier exited child.
            try:
                child = subprocess.Popen(
                    command, cwd=Path(command[0]).parent, shell=False,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            except OSError as exc:
                self.state = "failed"
                self.error = f"Could not launch Gufo: {exc}"
                raise ValueError(self.error) from exc
            self.process = child
            self.state = "loading"
            threading.Thread(target=self._read_output, args=(child,), daemon=True).start()
            threading.Thread(target=self._watch, args=(child, port, model_name), daemon=True).start()
        return self.snapshot()

    def _read_output(self, child):
        with child.stdout:
            while line := child.stdout.readline(4096):
                with self.lock:
                    if self.process is child:
                        self.logs.append(line.rstrip("\r\n"))

    def _watch(self, child, port, model_name):
        # Local probes must not inherit HTTP proxy settings from the user's shell.
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while child.poll() is None:
            with self.lock:
                if self.process is not child:
                    return
                loading = self.state == "loading"
            if loading:
                try:
                    with client.open(f"http://127.0.0.1:{port}/ready", timeout=0.5) as response:
                        body = json.loads(response.read(16384))
                    ready = isinstance(body, dict) and body.get("status") == "ready" and body.get("model") == model_name
                    with self.lock:
                        if ready and self.process is child and self.state == "loading" and child.poll() is None:
                            self.state = "ready"
                except (OSError, ValueError, urllib.error.URLError):
                    pass  # The model loads before Gufo starts listening.
            time.sleep(0.25)
        with self.lock:
            if self.process is child:
                self.exit_code = child.returncode
                if self.state == "stopping":
                    self.state = "stopped"
                elif self.state != "stopped":
                    self.state = "failed"
                    self.error = f"Gufo exited with code {child.returncode}. Check the process log below."

    def stop(self):
        with self.lock:
            child = self.process
            if child is None or child.poll() is not None:
                self.state = "stopped"
                self.error = ""
                return self.snapshot()
            self.state = "stopping"
        try:
            # On Windows this terminates only our owned child. Gufo currently has
            # no HTTP shutdown endpoint or handled CTRL_BREAK shutdown path.
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        except ProcessLookupError:
            pass
        finally:
            with self.lock:
                if self.process is child:
                    self.exit_code = child.poll()
                    self.state = "stopped" if child.poll() is not None else "failed"
                    self.error = "" if self.state == "stopped" else "Gufo could not be stopped; try Stop again."
        return self.snapshot()
