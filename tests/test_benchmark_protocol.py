"""Independent checks of real-image partitions, official recipes and final-test roles."""

import copy
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset

from cog_grouping.data import make_datasets, make_transforms, ParaXRandomResizedCrop
from cog_grouping.engine import make_optimizer, parax_learning_rate, train_epoch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _common import read_config
from run_benchmarks import build_plan
from freeze_benchmarks import freeze
from cog_grouping.provenance import source_fingerprint
import train as training_cli
import evaluate as evaluation_cli
import run_benchmarks as benchmark_cli
import summarize as summary_cli


class FixtureDataset(Dataset):
    """Expose dataset-specific labels while recording forbidden test construction."""

    calls = []

    def __init__(self, root, train=True, split=None, transform=None, download=False):
        """Provide two balanced classes without downloading benchmark images."""
        self.transform = transform
        self.targets = [0] * 10 + [1] * 10
        self.labels = np.asarray(self.targets)
        self._labels = self.targets
        self.classes = ["a", "b"]
        self.calls.append({"train": train, "split": split})

    def __len__(self):
        """Return the fixture's training-pool size."""
        return len(self.targets)

    def __getitem__(self, index):
        """Render distinct index-dependent images for loader integration tests."""
        image = Image.fromarray(np.full((32, 32, 3), index, dtype=np.uint8))
        return self.transform(image) if self.transform else image, self.targets[index]


class BenchmarkDataTests(unittest.TestCase):
    """Verify training/test boundaries for all automatically downloaded benchmarks."""

    def setUp(self):
        """Reset dataset call tracking for each isolated test."""
        FixtureDataset.calls = []
        self.normalization = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    def test_real_benchmark_development_never_constructs_test(self):
        """CIFAR, SVHN and Food use held-out training indices and no test object."""
        for name, constructor in (("cifar100", "CIFAR100"), ("svhn", "SVHN"), ("food101", "Food101")):
            with self.subTest(dataset=name), patch("cog_grouping.data.datasets." + constructor, FixtureDataset):
                FixtureDataset.calls = []
                train, validation, test, classes, roles = make_datasets(
                    {"dataset": name, "validation_fraction": 0.2, "image_size": 64}, self.normalization)
                self.assertEqual((len(train), len(validation)), (16, 4))
                self.assertIsNone(test)
                self.assertFalse(set(roles["train_indices"]) & set(roles["validation_indices"]))
                self.assertTrue(all(call["train"] and call["split"] in (None, "train") for call in FixtureDataset.calls))
                self.assertEqual(classes, 10 if name == "svhn" else 2)

    def test_confirmation_uses_full_train_and_fixed_role(self):
        """Full-pool confirmation must not create a validation or test dataset."""
        with patch("cog_grouping.data.datasets.CIFAR100", FixtureDataset):
            train, validation, test, _, roles = make_datasets(
                {"dataset": "cifar100", "phase": "confirmation", "validation_fraction": 0, "image_size": 64},
                self.normalization)
        self.assertEqual(len(train), 20)
        self.assertIsNone(validation)
        self.assertIsNone(test)
        self.assertEqual(roles["validation_indices"], [])
        self.assertTrue(all(call["train"] for call in FixtureDataset.calls))

    def test_benchmark_counts_reject_debug_data(self):
        """A tiny fixture cannot be reported as a complete CIFAR-100 benchmark."""
        with patch("cog_grouping.data.datasets.CIFAR100", FixtureDataset):
            with self.assertRaisesRegex(ValueError, "unexpected sample/class counts"):
                make_datasets({"dataset": "cifar100", "benchmark_protocol": "parax_matched_v1"}, self.normalization)

    def test_official_eval_resize_preserves_aspect_then_crops(self):
        """Evaluation must match resize-short-side 256 followed by a center crop."""
        from torchvision import transforms
        image = Image.fromarray(np.arange(320 * 200 * 3, dtype=np.uint32).reshape(200, 320, 3).astype(np.uint8))
        ours = make_transforms(224, **self.normalization, protocol="parax")
        expected = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                       transforms.CenterCrop(224), transforms.ToTensor(),
                                       transforms.Normalize(**self.normalization)])
        torch.testing.assert_close(ours(image), expected(image), atol=0, rtol=0)

    def test_crop_matches_unmodified_official_source(self):
        """The copied crop must reproduce official random coordinates under fixed seeds."""
        path = ROOT / "third_party/ParaX/image_classification/util/crop.py"
        if not path.exists():
            self.skipTest("official reference sources have not been fetched")
        spec = importlib.util.spec_from_file_location("reference_crop", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        image = Image.new("RGB", (320, 200))
        for seed in range(5):
            torch.manual_seed(seed)
            expected = module.RandomResizedCrop.get_params(image, (0.08, 1.0), (3 / 4, 4 / 3))
            torch.manual_seed(seed)
            actual = ParaXRandomResizedCrop.get_params(image, (0.08, 1.0), (3 / 4, 4 / 3))
            self.assertEqual(actual, expected)

    def test_imagenet_r_requires_split_evidence(self):
        """The code must never invent a new ImageNet-R split under a reference name."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "split_provenance"):
                make_datasets({"dataset": "imagenet-r", "data_root": directory}, self.normalization)


class BenchmarkRecipeTests(unittest.TestCase):
    """Check optimizer, update scheduling and resolved benchmark plans."""

    def test_recipe_and_paper_lr_are_distinguished(self):
        """The paper's absolute 0.1 and launcher's batch-scaled 0.4 must stay separate."""
        for dataset in ("cifar100", "svhn", "food101", "imagenet-r"):
            matched = read_config(ROOT / "configs/benchmark" / (dataset + ".json"))
            reference = read_config(ROOT / "configs/benchmark/reference" / (dataset + ".json"))
            self.assertEqual((matched["epochs"], matched["warmup_epochs"]), (100, 20))
            self.assertEqual(matched["batch_size"] * matched["accumulation_steps"], 1024)
            self.assertEqual(matched["learning_rate"], 0.1)
            self.assertEqual(reference["learning_rate"], 0.1 * 1024 / 256)
            self.assertEqual(reference["parax"]["kernel_sizes"], [3, 5, 7])
            self.assertTrue(reference["parax"]["enable_conv"])
            self.assertEqual(reference["parax"]["rank"], 128)
            self.assertEqual(reference["parax"]["readout"], "official_cls_bn")
            self.assertEqual(reference["precision"], "fp32")

    def test_schedule_matches_scalar_oracle(self):
        """Fractional warmup and cosine values must match the published code formula."""
        self.assertEqual(parax_learning_rate(0.1, 0, 100, 20), 0)
        self.assertAlmostEqual(parax_learning_rate(0.1, 10.5, 100, 20), 0.0525)
        self.assertAlmostEqual(parax_learning_rate(0.1, 20, 100, 20), 0.1)
        self.assertAlmostEqual(parax_learning_rate(0.1, 60, 100, 20), 0.05)
        self.assertAlmostEqual(parax_learning_rate(0.1, 100, 100, 20), 0)

    def test_sgd_momentum_and_tail_accumulation_agree(self):
        """Momentum updates with a short final group must match real larger batches."""
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(9)
        first = nn.Linear(3, 2)
        second = copy.deepcopy(first)
        data = TensorDataset(torch.randn(7, 3), torch.arange(7) % 2)
        config = {"optimizer": "sgd", "learning_rate": 0.1, "momentum": 0.9, "weight_decay": 0.0}
        first_opt, second_opt = make_optimizer(first, config), make_optimizer(second, config)
        torch.set_num_threads(1)
        calls = []
        def schedule(progress):
            """Record scheduling boundaries while holding the numerical test LR fixed."""
            calls.append(progress)
            return 0.1
        train_epoch(first, DataLoader(data, batch_size=2), first_opt, torch.device("cpu"),
                    accumulation_steps=2, max_grad_norm=None, schedule=schedule)
        train_epoch(second, DataLoader(data, batch_size=4), second_opt, torch.device("cpu"), max_grad_norm=None)
        for one, two in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(one, two, atol=2e-7, rtol=2e-6)
        self.assertEqual(calls, [0.0, 0.5])

    def test_plan_retains_all_methods_and_phase_roles(self):
        """Plans must preserve the same configuration for every matched method."""
        plan = build_plan(["cifar100"], ["grouping", "official_parax"], [42, 43], "matched", "confirmation", None, "runs/test_plan")
        self.assertEqual(len(plan), 4)
        for entry in plan:
            self.assertEqual(entry["config"]["validation_fraction"], 0)
            self.assertEqual(entry["config"]["phase"], "confirmation")
        with self.assertRaisesRegex(ValueError, "reserved"):
            build_plan(["cifar100"], ["grouping"], [42], "reference", "development", None, "runs/test_plan")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_plan(["cifar100"], ["grouping"], [42, 42], "matched", "development", None, "runs/test_plan")


class FrozenSelectionTests(unittest.TestCase):
    """Selection must use validation only and give methods equal search budgets."""

    def create_candidates(self, directory):
        """Create independent development records with method-specific validation optima."""
        source = source_fingerprint(ROOT)
        for mode in ("grouping", "mlp"):
            for lr in (0.03, 0.1):
                run = Path(directory) / (mode + str(lr))
                run.mkdir()
                config = read_config(ROOT / "configs/benchmark/cifar100.json")
                config.update({"mode": mode, "learning_rate": lr})
                accuracy = 0.8 if (mode == "grouping") == (lr == 0.1) else 0.7
                (run / "manifest.json").write_text(json.dumps({"config": config, "source": source}))
                (run / "status.json").write_text(json.dumps({"complete": True}))
                (run / "history.json").write_text(json.dumps([{"epoch": 99, "validation": {"accuracy": accuracy, "nll": 0.4}}]))

    def test_each_method_can_select_its_own_lr_using_validation(self):
        """The selection procedure must preserve different validation-optimal settings."""
        with tempfile.TemporaryDirectory() as temporary:
            self.create_candidates(temporary)
            result = freeze(temporary, Path(temporary) / "frozen.json")
            selected = {row["mode"]: row["selected_learning_rate"] for row in result["choices"]}
            self.assertEqual(selected, {"grouping": 0.1, "mlp": 0.03})
            self.assertFalse(result["final_test_accessed"])
            buffer = io.StringIO()
            arguments = ["run_benchmarks.py", "--phase", "confirmation", "--frozen-plan",
                         str(Path(temporary) / "frozen.json"), "--seeds", "42", "43", "--dry-run"]
            with patch.object(sys, "argv", arguments), patch.object(sys, "stdout", buffer):
                benchmark_cli.main()
            planned = [json.loads(line) for line in buffer.getvalue().splitlines()]
            self.assertEqual(len(planned), 4)
            for row in planned:
                self.assertEqual(row["config"]["learning_rate"], selected[row["config"]["mode"]])
                self.assertEqual(row["config"]["validation_fraction"], 0)
                self.assertTrue(row["config"]["frozen_plan_sha256"])

    def test_each_lr_requires_the_same_seed_set(self):
        """Seed changes must not be confounded with the learning-rate search."""
        with tempfile.TemporaryDirectory() as temporary:
            self.create_candidates(temporary)
            for mode in ("grouping", "mlp"):
                path = Path(temporary) / (mode + "0.03") / "manifest.json"
                manifest = json.loads(path.read_text())
                manifest["config"]["seed"] = 43
                path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "same development seed set"):
                freeze(temporary, Path(temporary) / "frozen.json")

    def test_different_lrs_can_only_share_the_same_frozen_confirmation(self):
        """Summary compatibility must preserve the frozen choice provenance."""
        with tempfile.TemporaryDirectory() as temporary:
            runs = []
            for mode, lr in (("grouping", 0.1), ("mlp", 0.03)):
                run = Path(temporary) / mode
                run.mkdir()
                runs.append(str(run))
                config = read_config(ROOT / "configs/benchmark/cifar100.json")
                config.update({"mode": mode, "phase": "confirmation", "validation_fraction": 0,
                               "learning_rate": lr, "frozen_plan_sha256": "same-frozen-plan"})
                manifest = {"config": config, "backbone_sha256": "fixture", "split_sha256": "fixture",
                            "normalization": {}, "source": {"sha256": "fixture"},
                            "parameters": {"adapter_trainable": 12, "head_trainable": 4}}
                (run / "manifest.json").write_text(json.dumps(manifest))
                (run / "status.json").write_text(json.dumps({"complete": True}))
                (run / "history.json").write_text(json.dumps([{"train": {"optimizer_steps": 1, "seconds": 1,
                                                                           "peak_cuda_allocated_bytes": None}}]))
                (run / "test_evaluation.json").write_text(json.dumps({"accuracy": 0.7, "nll": 0.4, "samples": 20}))
            arguments = ["summarize.py", "--runs"] + runs + ["--final-test", "--output", str(Path(temporary) / "summary.json")]
            with patch.object(sys, "argv", arguments), patch.object(sys, "stdout", io.StringIO()):
                summary_cli.main()
            path = Path(runs[1]) / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["config"]["frozen_plan_sha256"] = "different-plan"
            path.write_text(json.dumps(manifest))
            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    summary_cli.main()

    def test_unequal_search_budgets_are_rejected(self):
        """Missing trials for one method must not become an apparently fair search."""
        with tempfile.TemporaryDirectory() as temporary:
            self.create_candidates(temporary)
            (Path(temporary) / "mlp0.03/manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "unequal"):
                freeze(temporary, Path(temporary) / "frozen.json")

    def test_final_test_access_disqualifies_development_selection(self):
        """A development result with prior final-test access cannot select a method."""
        with tempfile.TemporaryDirectory() as temporary:
            self.create_candidates(temporary)
            (Path(temporary) / "grouping0.1/test_accessed.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "no final-test access"):
                freeze(temporary, Path(temporary) / "frozen.json")


class ConfirmationTrainingTests(unittest.TestCase):
    """Exercise the no-validation checkpoint path and its resume/test boundaries."""

    def test_full_pool_resume_selects_fixed_final_epoch(self):
        """Confirmation must resume with no validation metrics or test construction."""
        from test_model_engine_data import ToyFrozenEncoder
        def backbone(*args, **kwargs):
            """Supply a tiny frozen encoder while retaining the actual training engine."""
            value = ToyFrozenEncoder()
            value.data_config = {"mean": (0.0,) * 3, "std": (1.0,) * 3}
            value.weights_info = {"fixture": True}
            return value
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            config = {"backbone": "resnet18", "dataset": "cifar100", "pretrained": True,
                      "phase": "confirmation", "validation_fraction": 0, "image_size": 32,
                      "epochs": 2, "batch_size": 4, "bottleneck": 4, "grouping": {"radius": 1},
                      "device": "cpu", "precision": "fp32", "workers": 0, "threads": 1,
                      "optimizer": "sgd", "learning_rate": 0.01, "schedule": "parax_iteration"}
            config_path = path / "config.json"
            config_path.write_text(json.dumps(config))
            run = path / "run"
            arguments = ["train.py", "--config", str(config_path), "--output", str(run)]
            FixtureDataset.calls = []
            with patch.object(training_cli, "make_backbone", backbone), \
                    patch("cog_grouping.data.datasets.CIFAR100", FixtureDataset):
                with patch.object(sys, "argv", arguments + ["--epochs-this-invocation", "1"]):
                    training_cli.main()
                self.assertFalse(json.loads((run / "status.json").read_text())["complete"])
                with patch.object(sys, "argv", arguments + ["--resume"]):
                    training_cli.main()
            status = json.loads((run / "status.json").read_text())
            saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            self.assertTrue(status["complete"])
            self.assertIsNone(status["best_validation_accuracy"])
            self.assertEqual(saved["epoch"], 1)
            self.assertTrue(all(row["validation"] is None for row in saved["history"]))
            self.assertTrue(all(call["train"] for call in FixtureDataset.calls))
            self.assertFalse((run / "test_accessed.json").exists())

    def test_incomplete_run_cannot_open_final_test(self):
        """Final-test gating must happen before loading a checkpoint or dataset."""
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "status.json").write_text(json.dumps({"complete": False}))
            arguments = ["evaluate.py", "--run", str(run), "--split", "test", "--allow-test"]
            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(ValueError, "completed training"):
                    evaluation_cli.main()
            self.assertFalse((run / "test_accessed.json").exists())

if __name__ == "__main__":
    unittest.main()
