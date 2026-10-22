"""Frozen-encoder, accumulation, data-role and safe-checkpoint tests."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cog_grouping.backbones import BackboneOutput, FrozenBackbone
from cog_grouping.data import CrowdingDataset, make_datasets, stratified_indices
from cog_grouping.engine import evaluate, precision_context, train_epoch
from cog_grouping.grouping import GroupingConfig
from cog_grouping.model import FrozenVisionClassifier, build_classifier
from cog_grouping.provenance import atomic_json, atomic_torch_save, state_fingerprint


class ToyFrozenEncoder(FrozenBackbone):
    """A frozen normalization/dropout fixture with no downloaded dependencies."""

    def __init__(self):
        """Expose the same spatial interface as the real encoder wrappers."""
        super().__init__()
        self.channels = 6
        self.encoder = nn.Sequential(nn.Conv2d(3, 6, 3, padding=1), nn.BatchNorm2d(6), nn.Dropout(0.5))
        self.freeze()

    @torch.no_grad()
    def forward(self, images):
        """Return normal, non-inference tensors to the trainable adapter."""
        return BackboneOutput(self.encoder(images))


class ModelAndEngineTests(unittest.TestCase):
    """Verify optimization invariants, including the short final accumulation group."""

    def setUp(self):
        """Use a repeatable small CPU fixture."""
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_backbone_stays_frozen_and_in_eval(self):
        """Learning must update adapter/head but not encoder weights or BN buffers."""
        backbone = ToyFrozenEncoder()
        before = state_fingerprint(backbone)
        model = FrozenVisionClassifier(backbone, 3, bottleneck=4, grouping_config=GroupingConfig(radius=1))
        initial = model.adapter.up.weight.detach().clone()
        data = TensorDataset(torch.randn(7, 3, 5, 5), torch.arange(7) % 3)
        loader = DataLoader(data, batch_size=2)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        train_epoch(model, loader, optimizer, torch.device("cpu"), accumulation_steps=2)
        self.assertFalse(backbone.training)
        self.assertFalse(backbone.encoder[1].training)
        self.assertEqual(state_fingerprint(backbone), before)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))
        self.assertFalse(torch.equal(initial, model.adapter.up.weight))

    def test_accumulation_matches_sample_mean_with_tail(self):
        """Microbatches 2+2 and 2+1 must match actual batches 4 and 3."""
        first = nn.Linear(5, 3)
        second = copy.deepcopy(first)
        data = TensorDataset(torch.randn(7, 5), torch.arange(7) % 3)
        first_loader, second_loader = DataLoader(data, batch_size=2), DataLoader(data, batch_size=4)
        first_opt, second_opt = torch.optim.SGD(first.parameters(), 0.1), torch.optim.SGD(second.parameters(), 0.1)
        a = train_epoch(first, first_loader, first_opt, torch.device("cpu"), accumulation_steps=2, max_grad_norm=None)
        b = train_epoch(second, second_loader, second_opt, torch.device("cpu"), accumulation_steps=1, max_grad_norm=None)
        self.assertEqual(a["optimizer_steps"], 2)
        self.assertEqual(a["samples"], 7)
        for one, two in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(one, two, atol=2e-7, rtol=2e-6)

    def test_metrics_are_weighted_by_samples(self):
        """An incomplete last batch must not receive the weight of a full batch."""
        class Identity(nn.Module):
            """Interpret supplied tensors as logits."""
            def forward(self, values):
                """Return the fixture unchanged."""
                return values
        logits = torch.tensor([[10., 0.], [10., 0.], [0., 10.]])
        labels = torch.zeros(3, dtype=torch.long)
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        result = evaluate(Identity(), loader, torch.device("cpu"))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["nll"], float(nn.functional.cross_entropy(logits, labels)), places=5)

    def test_safe_checkpoint_roundtrip(self):
        """A checkpoint must survive atomic writing and weights_only loading."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            atomic_torch_save(path, {"tensor": torch.arange(5), "metadata": {"epoch": 2}})
            saved = torch.load(path, weights_only=True)
            torch.testing.assert_close(saved["tensor"], torch.arange(5))
            self.assertEqual(saved["metadata"]["epoch"], 2)

    def test_adapter_checkpoint_restores_predictions(self):
        """Strict adapter/head loading must exactly restore inference."""
        model = FrozenVisionClassifier(ToyFrozenEncoder(), 3, bottleneck=4, grouping_config=GroupingConfig(radius=1))
        state = copy.deepcopy(model.trainable_state_dict())
        images = torch.randn(2, 3, 4, 4)
        model.eval()
        expected = model(images).detach()
        with torch.no_grad():
            model.head.weight.add_(1)
        model.load_trainable_state_dict(state)
        torch.testing.assert_close(model(images), expected, atol=0, rtol=0)

    def test_cpu_rejects_cuda_precision(self):
        """Precision is an explicit experimental condition."""
        with self.assertRaises(ValueError):
            precision_context(torch.device("cpu"), "bf16")

    def test_local_modes_have_identical_head_initialization(self):
        """Different adapter constructors must not change the task-head starting point."""
        heads = []
        for mode in ("linear", "mlp", "boundary", "grouping"):
            model = build_classifier(ToyFrozenEncoder(), 3, {"mode": mode, "seed": 42, "bottleneck": 4}, ".")
            heads.append(model.head.weight.detach().clone())
        for head in heads[1:]:
            torch.testing.assert_close(heads[0], head, atol=0, rtol=0)


class DataTests(unittest.TestCase):
    """Check split integrity and deterministic, balanced synthetic trials."""

    def test_training_pool_split_is_disjoint_complete_and_stratified(self):
        """No training-pool item can belong to both optimization and validation."""
        labels = [0] * 10 + [1] * 10
        train, validation = stratified_indices(labels, 0.2, 42)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(sorted(train + validation), list(range(20)))
        self.assertEqual([labels[index] for index in validation].count(0), 2)
        self.assertEqual((train, validation), stratified_indices(labels, 0.2, 42))

    def test_crowding_is_deterministic_and_balanced(self):
        """Re-rendering the same trial must not redraw nuisance variables."""
        data = CrowdingDataset(size=10, image_size=64, seed=3)
        first, label = data[4]
        second, second_label = data[4]
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(label, second_label)
        self.assertEqual(sum(data.targets), 5)

    def test_default_construction_does_not_open_test(self):
        """Training exposes no final-test dataset or examples."""
        config = {"dataset": "crowding", "image_size": 64, "train_samples": 10, "val_samples": 4}
        _, _, test, _, split = make_datasets(config, {"mean": (0.,) * 3, "std": (1.,) * 3})
        self.assertIsNone(test)
        self.assertIsNone(split["effective_sizes"]["test"])

    def test_density_interventions_preserve_background_noise(self):
        """A clutter intervention must not also redraw the trial's background noise."""
        for index in range(4):
            sparse = CrowdingDataset(size=4, image_size=64, seed=3, density="sparse")[index][0]
            dense = CrowdingDataset(size=4, image_size=64, seed=3, density="dense")[index][0]
            torch.testing.assert_close(sparse[:, :5, :5], dense[:, :5, :5], atol=0, rtol=0)
            self.assertFalse(torch.equal(sparse, dense))

    def test_role_seeds_change_stimuli(self):
        """Matching trial indices across roles must not create the same image."""
        train = CrowdingDataset(size=4, image_size=64, seed=3)
        validation = CrowdingDataset(size=4, image_size=64, seed=100000003)
        self.assertFalse(torch.equal(train[0][0], validation[0][0]))

    def test_invalid_small_or_unbalanced_data_is_rejected(self):
        """Invalid settings must not masquerade as balanced diagnostics."""
        with self.assertRaises(ValueError):
            CrowdingDataset(size=3)
        with self.assertRaises(ValueError):
            stratified_indices([0, 1], 0.2, 42)
