"""Preset lifecycle, migration, and model constraints without loading weights."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/gui"))
from command import build_command
from presets import (change_preset, load_document, new_document, resolved_settings,
                     save_document, validate_document)
from settings import DEFAULTS


class PresetsTest(unittest.TestCase):
    def change(self, document, **request):
        return change_preset(document, {"revision": document["revision"], **request})

    def test_migration_preserves_values_and_original_until_save(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.json"
            legacy = {"mtp": True, "context": 200000, "temperature": 0.7,
                      "mtp_model": str(Path(directory) / "sidecar.gguf")}
            original = json.dumps({"version": 1, "settings": legacy}).encode()
            path.write_bytes(original)
            document = load_document(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(path.with_name("launcher.json.bak").exists())
            values = resolved_settings(document)
            self.assertEqual((values["speculative"], values["context"], values["temperature"]), ("mtp", 200000, 0.7))
            self.assertEqual(values["mtp_model"], legacy["mtp_model"])
            self.assertEqual(document["presets"][0]["name"], "Default")
            self.assertEqual(values["model_family"], "generic")
            save_document(path, document)
            self.assertEqual(load_document(path), document)
            self.assertEqual(path.with_name("launcher.json.bak").read_bytes(), original)

    def test_lifecycle_keeps_ids_and_shared_settings_separate(self):
        original = new_document()
        values = DEFAULTS | {"context": 65536, "port": 8088, "model_family": "qwen38_27b",
                             "speculative": "dflash2", "temperature": 0.6}
        created = self.change(original, action="create", name="27B · 64K", settings=values)
        preset_id = created["selected_preset"]
        self.assertNotEqual(preset_id, "default")
        self.assertEqual(len(original["presets"]), 1)
        self.assertEqual(resolved_settings(created), values)
        self.assertEqual(resolved_settings(created, "default")["port"], 8088)
        self.assertEqual(resolved_settings(created, "default")["context"], 32768)
        self.assertNotIn("port", created["presets"][1]["settings"])
        renamed = self.change(created, action="rename", id=preset_id, name="Everyday")
        self.assertEqual(renamed["presets"][1]["id"], preset_id)
        self.assertEqual(resolved_settings(renamed), values)
        saved = self.change(renamed, action="save", id=preset_id, settings=values | {"context": 8192})
        self.assertEqual(resolved_settings(saved)["context"], 8192)
        deleted = self.change(saved, action="delete", id=preset_id)
        self.assertEqual(deleted["selected_preset"], "default")
        self.assertEqual(deleted["revision"], 4)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.change(deleted, action="delete", id="default")

    def test_rejects_stale_unknown_and_duplicate_operations(self):
        document = new_document()
        for request in ({"action": "save", "id": "default", "settings": DEFAULTS, "revision": 9},
                        {"action": "delete", "id": "missing", "revision": 0},
                        {"action": "create", "name": "default", "settings": DEFAULTS, "revision": 0},
                        {"action": "rename", "id": "default", "name": "\nOops", "revision": 0},
                        {"action": "rename", "id": "default", "name": " ", "revision": 0},
                        {"action": "save", "id": "default", "settings": DEFAULTS, "revision": True}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                change_preset(document, request)
        self.assertEqual(document, new_document())

    def test_invalid_documents_and_failed_writes_preserve_saved_state(self):
        document = new_document()
        invalid = deepcopy(document)
        invalid["selected_preset"] = "missing"
        with self.assertRaises(ValueError):
            validate_document(invalid)
        invalid = deepcopy(document)
        invalid["presets"][0]["settings"]["port"] = 8000
        with self.assertRaises(ValueError):
            validate_document(invalid)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.json"
            save_document(path, document)
            before = path.read_bytes()
            changed = self.change(document, action="rename", id="default", name="Changed")
            with patch("presets.os.replace", side_effect=OSError("disk error")), self.assertRaises(OSError):
                save_document(path, changed)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_family_constraints_and_inactive_flags(self):
        for family, mode in (("qwen38_flash_next", "dflash2"), ("qwen38_27b", "mtp")):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "does not support"):
                build_command(DEFAULTS | {"model_family": family, "speculative": mode})
        values = DEFAULTS | {"model_family": "qwen38_27b", "speculative": "dflash2", "prompt_lookup": True,
                             "mtp_policy": "survival", "reasoning_effort": "low", "think": "off"}
        command = build_command(values)
        for flag in ("--model-family", "--reasoning-effort", "--mtp-policy", "--prompt-lookup", "--mtp-model"):
            self.assertNotIn(flag, command)
        self.assertIn("--dflash-model", command)

    def test_corrupt_version_two_is_backed_up_before_explicit_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.json"
            original = b'{"version": 2, "presets": "broken"}'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                load_document(path)
            save_document(path, new_document())
            self.assertEqual(path.with_name("launcher.json.bak").read_bytes(), original)
            path.write_bytes(b"another broken document")
            save_document(path, new_document())
            self.assertEqual(path.with_name("launcher.json.bak").read_bytes(), original)

    def test_invalid_legacy_payload_is_not_silently_replaced_with_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "launcher.json"
            for payload in (None, [], "", False):
                path.write_text(json.dumps({"version": 1, "settings": payload}), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_document(path)


if __name__ == "__main__":
    unittest.main()
