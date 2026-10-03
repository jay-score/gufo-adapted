"""Confirm the production fatal handler exits before lost-device cleanup."""

import subprocess
import sys


def main():
    control = subprocess.run(
        [sys.argv[1], "--device-loss-no-handler-child"],
        capture_output=True,
        timeout=5,
    )
    assert control.returncode == 76, control.returncode
    result = subprocess.run(
        [sys.argv[1], "--device-loss-child"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 75, (result.returncode, result.stdout, result.stderr)
    assert "event=device_lost remedy=restart exit_status=75" in result.stderr
    assert "reason=injected scheduler runner failure" in result.stderr
    print("PASS: device loss exits 75 before invalidation or model teardown")


if __name__ == "__main__":
    main()
