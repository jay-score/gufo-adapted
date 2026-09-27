"""Windows launcher regression checks; no model, ROCm, or GPU required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    if os.name != "nt":
        print("SKIP: Windows launcher checks require Windows")
        return
    source = Path(__file__).resolve().parents[2] / "tools/windows/run-flash-next.ps1"
    compiler = Path(os.environ["WINDIR"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    shells = [shutil.which(name) for name in ("powershell.exe", "pwsh.exe")]
    assert shells[0], "Windows PowerShell is required"
    switches = {
        "-DraftVocab": ["--mtp-draft-vocab", "latin"],
        "-Survival": ["--mtp-policy", "survival"],
        "-Lookup": ["--prompt-lookup"],
    }
    count = 0
    with tempfile.TemporaryDirectory(prefix="gufo-launcher-") as temporary:
        root = Path(temporary)
        launcher = root / "tools/windows/run-flash-next.ps1"
        launcher.parent.mkdir(parents=True)
        shutil.copy2(source, launcher)
        binary = root / "build/release/gufo.exe"
        binary.parent.mkdir(parents=True)
        stub = root / "stub.cs"
        stub.write_text('''using System;
class Stub {
    static int Main(string[] args) {
        foreach (string arg in args) Console.WriteLine("[" + arg + "]");
        return int.Parse(Environment.GetEnvironmentVariable("EXIT_STUB_STATUS"));
    }
}
''')
        subprocess.run([str(compiler), "/nologo", f"/out:{binary}", str(stub)], check=True)
        model = root / "model"
        for relative in (
            "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
            "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
        ):
            path = model / relative
            path.parent.mkdir(parents=True)
            path.touch()

        def run(shell, mode, draft, flags=(), exit_code=0):
            environment = os.environ.copy()
            environment["EXIT_STUB_STATUS"] = str(exit_code)
            return subprocess.run(
                [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher),
                 "-Snapshot", str(model), "-Mode", mode, "-Draft", draft, *flags],
                env=environment, capture_output=True, text=True, timeout=30,
            )

        for shell in filter(None, shells):
            for mode in ("serve", "bench"):
                for draft in ("mtp", "mtp3", "off"):
                    for flags in ([], *([flag] for flag in switches), list(switches)):
                        result = run(shell, mode, draft, flags)
                        args = [line[1:-1] for line in result.stdout.splitlines()
                                if line.startswith("[") and line.endswith("]")]
                        rejected = flags and (mode != "serve" or draft == "off")
                        if rejected:
                            assert result.returncode != 0 and not args, result.stdout
                            assert "require -Mode serve and -Draft mtp or mtp3" in result.stderr, result.stderr
                        else:
                            assert result.returncode == 0 and args[0] == mode, result.stderr
                            for flag, expected in switches.items():
                                if flag in flags:
                                    index = args.index(expected[0])
                                    assert args[index:index + len(expected)] == expected, args
                                else:
                                    assert expected[0] not in args, args
                        count += 1
                for code in (7, 42):
                    result = run(shell, mode, "mtp", exit_code=code)
                    assert result.returncode == code, result.stderr
                    count += 1
    print(f"PASS: {count} Windows launcher checks")


if __name__ == "__main__":
    main()
