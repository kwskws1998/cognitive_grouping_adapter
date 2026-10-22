"""Verify setup cannot install packages outside the active conda environment."""

import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("project_setup_env", ROOT / "scripts/setup_env.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class CondaSetupTests(unittest.TestCase):
    """Mock package installation while testing environment identity and commands."""

    def test_missing_conda_refuses_before_installing(self):
        """An unactivated environment must never trigger pip installation."""
        with patch.dict(os.environ, {}, clear=True), patch.object(SETUP.subprocess, "run") as run, \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                SETUP.main([])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_non_conda_prefix_is_rejected(self):
        """An ordinary Python environment cannot be relabeled using CONDA_PREFIX."""
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"CONDA_PREFIX": temporary}), patch.object(sys, "prefix", temporary):
                with self.assertRaisesRegex(RuntimeError, "does not identify a conda"):
                    SETUP.active_conda_python()

    def test_different_interpreter_prefix_is_rejected(self):
        """A nested environment must not receive packages intended for conda."""
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "conda-meta").mkdir()
            with patch.dict(os.environ, {"CONDA_PREFIX": temporary}), patch.object(sys, "prefix", temporary + "/nested"):
                with self.assertRaisesRegex(RuntimeError, "does not belong"):
                    SETUP.active_conda_python()

    def test_install_uses_active_python_without_creating_an_environment(self):
        """All package commands must target the validated active interpreter."""
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            python = str(prefix / "bin/python")
            with patch.dict(os.environ, {"CONDA_PREFIX": temporary}), patch.object(sys, "prefix", temporary), \
                    patch.object(sys, "executable", python), patch.object(SETUP.subprocess, "run") as run, \
                    patch.object(SETUP.subprocess, "check_output", return_value="torch==2.8.0\n") as freeze, \
                    patch.object(SETUP.Path, "write_text") as write:
                SETUP.main(["--torch-index-url", "https://download.pytorch.org/whl/cu128"])
                self.assertEqual(run.call_count, 3)
                for call in run.call_args_list:
                    self.assertEqual(call.args[0][:4], [python, "-m", "pip", "install"])
                    self.assertTrue(call.kwargs["check"])
                torch_command = run.call_args_list[1].args[0]
                self.assertEqual(torch_command[-2:], ["--index-url", "https://download.pytorch.org/whl/cu128"])
                freeze.assert_called_once_with([python, "-m", "pip", "freeze"], text=True)
                write.assert_called_once_with("torch==2.8.0\n", encoding="utf-8")
            self.assertEqual(sorted(path.name for path in prefix.iterdir()), ["conda-meta"])


if __name__ == "__main__":
    unittest.main()
