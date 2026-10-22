"""End-to-end CLI checks for resumption, provenance and explicit test access."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


class CliProtocolTests(unittest.TestCase):
    """Use tiny random encoders only to test orchestration, never model efficacy."""

    @classmethod
    def setUpClass(cls):
        """Locate command entry points independently of the current directory."""
        cls.root = Path(__file__).resolve().parents[1]

    def command(self, arguments, expect_success=True):
        """Capture command errors so failed protocol assertions remain informative."""
        result = subprocess.run([sys.executable] + arguments, cwd=str(self.root), capture_output=True, text=True)
        if expect_success and result.returncode:
            self.fail(result.stdout + result.stderr)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def test_epoch_boundary_resume_is_exact(self):
        """Interrupted+resumed training must reproduce uninterrupted adapter/head tensors."""
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            config = json.loads((self.root / "configs/cpu_smoke.json").read_text())
            config.update({"train_samples": 8, "val_samples": 4, "threads": 1})
            config_path = temporary / "config.json"
            config_path.write_text(json.dumps(config))
            continuous, resumed = temporary / "continuous", temporary / "resumed"
            base = ["scripts/train.py", "--config", str(config_path), "--allow-random-backbone"]
            self.command(base + ["--output", str(continuous)])
            self.command(base + ["--output", str(resumed), "--epochs-this-invocation", "1"])
            self.command(base + ["--output", str(resumed), "--resume"])
            first = torch.load(continuous / "last.pt", map_location="cpu", weights_only=True)
            second = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
            for component in ("adapter", "head"):
                for name in first["model"][component]:
                    torch.testing.assert_close(first["model"][component][name], second["model"][component][name], atol=0, rtol=0)
            self.assertFalse((resumed / "test_accessed.json").exists())
            self.command(["scripts/evaluate.py", "--run", str(resumed), "--split", "test"], expect_success=False)
            self.assertFalse((resumed / "test_accessed.json").exists())
            self.command(["scripts/evaluate.py", "--run", str(resumed), "--split", "validation"])
            self.command(["scripts/evaluate_crowding.py", "--run", str(resumed), "--samples", "4",
                          "--output", str(temporary / "interventions.json")])
            interventions = json.loads((temporary / "interventions.json").read_text())
            self.assertEqual(interventions["role"], "development diagnostic")
            self.assertEqual(len(interventions["observations"]), 9)
            summary = ["scripts/summarize.py", "--runs", str(continuous), str(resumed),
                       "--output", str(temporary / "summary.json")]
            self.command(summary)
            manifest_path = continuous / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["config"]["max_train_samples"] = 4
            manifest_path.write_text(json.dumps(manifest))
            self.command(summary, expect_success=False)
            self.command(["scripts/evaluate.py", "--run", str(resumed), "--split", "test", "--allow-test"])
            self.assertTrue((resumed / "test_accessed.json").exists())
            self.command(base + ["--output", str(resumed), "--resume"], expect_success=False)

    def test_random_backbone_requires_explicit_smoke_override(self):
        """A failed pretrained load can never quietly become a random-backbone run."""
        with tempfile.TemporaryDirectory() as temporary:
            result = self.command(["scripts/train.py", "--config", "configs/cpu_smoke.json", "--output", temporary], expect_success=False)
            self.assertIn("--allow-random-backbone", result.stderr)
