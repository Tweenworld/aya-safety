"""Release gate must fail closed: leaks, secrets, local paths, missing inputs and empty gates all raise.

Run: python -m unittest discover -s tests -v
"""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import build_release as br  # noqa: E402

SIG = "the shopkeeper should confirm the KES 4,500 payment before releasing"  # synthetic shingle


class TestReleaseGate(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def write(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def test_clean_dir_passes(self):
        self.write("ok.md", "Aggregate table only: 35/66, 22/66, 21/66.")
        self.assertEqual(br.verify_release_dir(self.dir, {SIG})["files_scanned"], 1)

    def test_verbatim_leak_caught(self):
        self.write("leak.md", f"Example output: {SIG} immediately.")
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_json_escaped_leak_caught(self):
        # Completions inside .jsonl carry escaped newlines; the shingle spans one here.
        escaped = SIG.replace("confirm the", "confirm\\nthe")
        self.write("leak.jsonl", '{"x": "' + escaped + '"}\n')
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_leak_in_unexpected_extension_caught(self):
        self.write("weights.bin", f"\x00\x01{SIG}\x02")
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_secret_caught(self):
        self.write("cfg.txt", "HF_TOKEN=hf_" + "a" * 34)
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_local_path_caught(self):
        # Assembled at runtime so this source file does not itself trip the pre-push path check.
        sep = "\\\\"
        self.write("prov.json", '{"ref": "C:' + sep + "Users" + sep + "someone" + sep + 'data.jsonl"}')
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_empty_signature_set_refuses(self):
        self.write("ok.md", "fine")
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, set())

    def test_empty_dir_refuses(self):
        with self.assertRaises(RuntimeError):
            br.verify_release_dir(self.dir, {SIG})

    def test_missing_input_refuses(self):
        with self.assertRaises(FileNotFoundError):
            br.copy_required([("artifacts/does_not_exist.pt", "model/x.pt")])


if __name__ == "__main__":
    unittest.main()
