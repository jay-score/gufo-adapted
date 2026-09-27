"""Settings survive restarts, and GUI choices generate a safe native argv."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
import launcher_config as config


class ConfigTest(unittest.TestCase):
    def test_round_trip_and_failed_write_preserves_previous_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings/launcher.json"
            self.assertEqual(config.load_settings(path), config.DEFAULTS)
            settings = config.DEFAULTS | {"model": str(Path(directory) / "模型 & test.gguf"), "mtp": True}
            config.save_settings(path, settings)
            self.assertEqual(config.load_settings(path), settings)
            before = path.read_bytes()
            with patch.object(config.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    config.save_settings(path, settings | {"context": 8192})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            path.write_text('{"version": 99, "settings": {}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                config.load_settings(path)
            path.write_text("{bad json", encoding="utf-8")
            with self.assertRaises(ValueError):
                config.load_settings(path)

    def test_rejects_invalid_values_before_save_or_launch(self):
        for change in ({"port": True}, {"port": 0}, {"port": 65536}, {"context": -1},
                       {"temperature": float("nan")}, {"temperature": 2.1}, {"top_p": 1.1}, {"top_p": 0},
                       {"top_k": 2.5}, {"seed": 2**60}, {"max_tokens": 0}, {"repeat_penalty": 0},
                       {"think": "yes"}, {"mtp": "false"}, {"model": "relative.gguf"},
                       {"served_model_name": "line\nbreak"}, {"extra_args": "--anything"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.validate_settings(change)

    def test_shards_sidecars_and_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="gufo files ") as directory:
            root = Path(directory)
            model = root / "模型-00001-of-00003.gguf"
            model.write_bytes(b"GGUF")  # Metadata-only first shards are valid.
            exe = root / "gufo.exe"
            exe.touch()
            settings = config.DEFAULTS | {"model": str(model), "executable": str(exe)}
            with self.assertRaisesRegex(ValueError, "missing shard"):
                config.build_command(settings, check_files=True)
            for index in (2, 3):
                (root / f"模型-{index:05d}-of-00003.gguf").write_bytes(b"GGUF")
            command = config.build_command(settings, check_files=True)
            self.assertEqual(command[0], str(exe))
            self.assertEqual(command[command.index("--model") + 1], str(model))
            self.assertNotIn("--mmproj", command)
            self.assertNotIn("--mtp-model", command)
            with self.assertRaisesRegex(ValueError, "first shard"):
                config.check_gguf(str(root / "模型-00002-of-00003.gguf"), "Model", sharded=True)
            settings.update(mtp=True, prompt_lookup=True, mtp_policy="survival", mtp_draft_vocab="latin")
            with self.assertRaisesRegex(ValueError, "MTP sidecar"):
                config.build_command(settings, check_files=True)
            sidecar = root / "mtp & sidecar.gguf"
            sidecar.write_bytes(b"GGUF")
            settings.update(mtp_model=str(sidecar), mmproj=str(sidecar))
            command = config.build_command(settings, check_files=True)
            self.assertIn("--prompt-lookup", command)
            self.assertEqual(command[command.index("--mtp-model") + 1], str(sidecar))
            self.assertEqual(command[command.index("--mmproj") + 1], str(sidecar))
            sidecar.write_text("not a model")
            with self.assertRaisesRegex(ValueError, "GGUF header"):
                config.build_command(settings, check_files=True)

    def test_folder_picker_groups_shards_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "subfolder").mkdir()
            for name in ("model-00001-of-00002.gguf", "model-00002-of-00002.gguf", "mtp.gguf", "gufo.exe", "notes.txt"):
                (root / name).touch()
            result = config.browse_directory(directory)
            self.assertEqual([entry["name"] for entry in result["entries"]],
                             ["subfolder", "model-00001-of-00002.gguf", "mtp.gguf"])
            self.assertEqual(result["entries"][1]["shards"], 2)
            self.assertEqual([entry["name"] for entry in config.browse_directory(directory, "executable")["entries"]],
                             ["subfolder", "gufo.exe"])
            with self.assertRaises(ValueError):
                config.browse_directory("relative")


if __name__ == "__main__":
    unittest.main()
