"""Settings survive restarts, and GUI choices generate a safe native argv."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
from command import build_command
from files import browse_directory, check_gguf
from presets import load_document, new_document, resolved_settings, save_document
from settings import DEFAULTS, validate_settings


class ConfigTest(unittest.TestCase):
    def test_round_trip_and_failed_write_preserves_previous_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings/launcher.json"
            self.assertEqual(resolved_settings(load_document(path)), DEFAULTS)
            settings = DEFAULTS | {"model": str(Path(directory) / "模型 & test.gguf"), "speculative": "mtp"}
            save_document(path, new_document(settings))
            self.assertEqual(resolved_settings(load_document(path)), settings)
            before = path.read_bytes()
            with patch("presets.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    save_document(path, new_document(settings | {"context": 8192}))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            path.write_text('{"version": 99, "settings": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                resolved_settings(load_document(path))
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolved_settings(load_document(path))

    def test_rejects_invalid_values_before_save_or_launch(self):
        for change in ({"port": True}, {"port": 0}, {"port": 65536}, {"context": -1},
                       {"temperature": float("nan")}, {"temperature": 2.1}, {"top_p": 1.1}, {"top_p": 0},
                       {"top_k": 2.5}, {"seed": 2**60}, {"max_tokens": 0}, {"repeat_penalty": 0},
                       {"think": "yes"}, {"mtp": "false"}, {"speculative": "unknown"},
                       {"dflash_model": "relative.gguf"}, {"model": "relative.gguf"},
                       {"served_model_name": "line\nbreak"}, {"extra_args": "--anything"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_settings(change)

    def test_shards_sidecars_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="gufo files ") as directory:
            root = Path(directory)
            model = root / "模型-00001-of-00003.gguf"
            model.write_bytes(b"GGUF")  # Metadata-only first shards are valid.
            exe = root / "gufo.exe"
            exe.touch()
            settings = DEFAULTS | {"model": str(model), "executable": str(exe)}
            with self.assertRaisesRegex(ValueError, "missing shard"):
                build_command(settings, check_files=True)
            for index in (2, 3):
                (root / f"模型-{index:05d}-of-00003.gguf").write_bytes(b"GGUF")
            command = build_command(settings, check_files=True)
            self.assertEqual(command[0], str(exe))
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertNotIn("--mmproj", command)
            self.assertNotIn("--mtp-model", command)
            with self.assertRaisesRegex(ValueError, "first shard"):
                check_gguf(str(root / "模型-00002-of-00003.gguf"), "Model", sharded=True)
            settings.update(speculative="mtp", prompt_lookup=True, mtp_policy="survival", mtp_draft_vocab="latin")
            with self.assertRaisesRegex(ValueError, "MTP sidecar"):
                build_command(settings, check_files=True)
            sidecar = root / "mtp & sidecar.gguf"
            sidecar.write_bytes(b"GGUF")
            settings.update(mtp_model=str(sidecar), mmproj=str(sidecar))
            command = build_command(settings, check_files=True)
            self.assertIn("--prompt-lookup", command)
            self.assertEqual(command[command.index("--mtp-model") + 1], str(sidecar))
            self.assertEqual(command[command.index("--mmproj") + 1], str(sidecar))
            sidecar.write_text("not a model")
            with self.assertRaisesRegex(ValueError, "GGUF header"):
                build_command(settings, check_files=True)

    def test_legacy_mtp_settings_migrate_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.json"
            for enabled, mode in ((True, "mtp"), (False, "off")):
                legacy = {"mtp": enabled, "context": 200000, "draft_tokens": 3,
                          "mtp_model": str(Path(directory) / "mtp.gguf")}
                path.write_text(json.dumps({"version": 1, "settings": legacy}), encoding="utf-8")
                before = path.read_bytes()
                settings = resolved_settings(load_document(path))
                self.assertEqual(settings["speculative"], mode)
                self.assertNotIn("mtp", settings)
                self.assertEqual(settings["mtp_model"], legacy["mtp_model"])
                self.assertEqual(settings["context"], 200000)
                self.assertEqual(settings["draft_tokens"], 3)
                self.assertEqual(path.read_bytes(), before)
                save_document(path, new_document(settings))
                self.assertEqual(resolved_settings(load_document(path)), settings)

    def test_dflash_and_off_do_not_inherit_mtp_flags(self):
        with tempfile.TemporaryDirectory(prefix="gufo draft ") as directory:
            root = Path(directory)
            exe, model, draft = (root / name for name in ("gufo.exe", "main.gguf", "DFlash2.gguf"))
            exe.touch()
            model.write_bytes(b"GGUF")
            settings = DEFAULTS | {
                "executable": str(exe), "model": str(model), "speculative": "dflash2",
                "dflash_model": str(draft), "mtp_model": str(root / "missing-mtp.gguf"),
                "prompt_lookup": True, "mtp_policy": "survival", "mtp_draft_vocab": "latin",
            }
            with self.assertRaisesRegex(ValueError, "DFlash2 draft"):
                build_command(settings, check_files=True)
            draft.write_bytes(b"GGUF")
            command = build_command(settings, check_files=True)
            self.assertEqual(command[command.index("--speculative") + 1], "dflash2")
            self.assertEqual(command[command.index("--dflash-model") + 1], str(draft))
            for flag in ("--mtp-model", "--draft-tokens", "--mtp-policy", "--mtp-draft-vocab", "--prompt-lookup"):
                self.assertNotIn(flag, command)
            path = root / "launcher.json"
            save_document(path, new_document(settings))
            self.assertEqual(resolved_settings(load_document(path)), settings)
            draft.unlink()
            command = build_command(settings | {"speculative": "off"}, check_files=True)
            self.assertEqual(command[command.index("--speculative") + 1], "off")
            self.assertNotIn("--dflash-model", command)
            self.assertNotIn("--mtp-model", command)

    def test_folder_picker_groups_shards_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "subfolder").mkdir()
            for name in ("model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "mtp.gguf", "gufo.exe", "notes.txt"):
                (root / name).touch()
            result = browse_directory(directory)
            self.assertEqual([entry["name"] for entry in result["entries"]],
                             ["subfolder", "model-00001-of-00002.gguf", "mtp.gguf"])
            self.assertEqual(result["entries"][1]["shards"], 2)
            self.assertEqual([entry["name"] for entry in browse_directory(directory, "executable")["entries"]],
                             ["subfolder", "gufo.exe"])
            with self.assertRaises(ValueError):
                browse_directory("relative")


if __name__ == "__main__":
    unittest.main()
