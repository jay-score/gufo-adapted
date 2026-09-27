"""Hash comparison must catch changes outside the old top-64 summary."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

script = Path(__file__).resolve().parents[2] / "tools/bench/logit-eval.py"
spec = importlib.util.spec_from_file_location("logit_eval", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LogitEvalTest(unittest.TestCase):
    def test_full_row_and_invalid_inputs(self):
        values = [float(i) for i in range(128)]

        def dump(row):
            digest = hashlib.sha256(struct.pack("<128f", *row)).hexdigest()
            return f"GFLE-SHA256 1 128 1\n1 127 {digest}\n"

        with tempfile.TemporaryDirectory() as temporary:
            left, right = (Path(temporary) / name for name in ("left", "right"))
            left.write_text(dump(values))
            right.write_text(dump(values))
            self.assertEqual(module.compare(left, right), ([], 1))
            values[0] = -1.0  # Same top 64, changed tail logit.
            right.write_text(dump(values))
            self.assertEqual(module.compare(left, right), ([1], 1))
            valid = dump(values)
            for invalid in (
                "", "GFLE-SHA256 1 128 1\n", valid.replace(" 127 ", " 128 "),
                valid.replace("\n1 ", "\n2 "), valid.replace(" 1 128", " 2 128"),
                valid + valid.splitlines()[1] + "\n",
            ):
                right.write_text(invalid)
                with self.assertRaises(ValueError):
                    module.compare(left, right)
            right.write_text(valid.replace(" 127 ", " 126 "))
            with self.assertRaises(ValueError):
                module.compare(left, right)


if __name__ == "__main__":
    unittest.main()
