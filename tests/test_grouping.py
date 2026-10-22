"""Numerical and mechanism tests, including boundary behavior and gradients."""

import unittest
import math

import torch

from cog_grouping.grouping import DirectionalGrouping, GroupingConfig, shifted
from cog_grouping.adapters import EvidenceAdapter


class GroupingTests(unittest.TestCase):
    """Check experimentally relevant invariants rather than classification claims."""

    @classmethod
    def setUpClass(cls):
        """Keep the small numerical tests efficient on CPU."""
        torch.set_num_threads(1)

    def setUp(self):
        """Use a deterministic independent random fixture for every test."""
        torch.manual_seed(57)
        self.operator = DirectionalGrouping(GroupingConfig(radius=2))

    def test_shapes_conservation_bounds(self):
        """All outputs must remain finite, bounded and on the input grid."""
        amplitude = torch.rand(2, 3, 5, 7)
        result = self.operator(amplitude, torch.randn(2, 4, 5, 7))
        self.assertEqual(result.widths.shape, (2, 4, 5, 7))
        for value in (result.unique, result.shared, result.context, result.private_fraction):
            self.assertEqual(value.shape, amplitude.shape)
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(result.unique + result.shared, amplitude)
        self.assertTrue((result.private_fraction >= 0).all() and (result.private_fraction <= 1).all())

    def test_no_wraparound(self):
        """A right-edge signal must not appear at the left edge after a shift."""
        image = torch.zeros(1, 1, 3, 5)
        image[..., -1] = 9
        value, valid = shifted(image, 0, -1)
        self.assertEqual(float(value[..., 0].abs().sum()), 0)
        self.assertFalse(bool(valid[..., 0].any()))

    def test_constant_context_is_not_diluted_at_edges(self):
        """Invalid padded sources must not enter the context normalization."""
        result = self.operator(torch.ones(1, 2, 3, 5), torch.ones(1, 4, 3, 5))
        torch.testing.assert_close(result.context, torch.ones_like(result.context))

    def test_empty_activity(self):
        """No activity yields no unique/shared/context evidence and no NaNs."""
        result = self.operator(torch.zeros(1, 2, 4, 4), torch.randn(1, 3, 4, 4))
        for value in (result.unique, result.shared, result.context, result.private_fraction):
            self.assertEqual(float(value.abs().sum()), 0)

    def test_singleton_is_fully_private(self):
        """A single valid cell is never grouped with padding."""
        result = self.operator(torch.ones(1, 2, 1, 1), torch.randn(1, 3, 1, 1))
        torch.testing.assert_close(result.private_fraction, torch.ones_like(result.private_fraction))
        torch.testing.assert_close(result.context, torch.ones_like(result.context))

    def test_isolated_center_is_fully_private(self):
        """An active item with no active neighboring sources has fraction one."""
        amplitude = torch.zeros(1, 1, 7, 7)
        amplitude[..., 3, 3] = 1
        result = self.operator(amplitude, torch.randn(1, 3, 7, 7))
        self.assertEqual(float(result.private_fraction[..., 3, 3]), 1)

    def test_neighbor_similarity_changes_spread(self):
        """Same and orthogonal descriptors must cause different local spread."""
        similar = torch.zeros(1, 2, 5, 5)
        similar[:, 0] = 1
        dissimilar = similar.clone()
        dissimilar[..., 2, 3] = torch.tensor([0., 1.]).view(1, 2)
        width_same = self.operator.neighbor_widths(similar)[0, 1, 2, 2]
        width_different = self.operator.neighbor_widths(dissimilar)[0, 1, 2, 2]
        self.assertGreater(float(width_same), float(width_different))

    def test_nearest_neighbor_limits_direct_width_dependence(self):
        """A distant descriptor cannot directly alter a source's local spread width."""
        descriptors = torch.randn(1, 3, 7, 7)
        changed = descriptors.clone()
        changed[..., 3, 6] *= -1
        torch.testing.assert_close(self.operator.neighbor_widths(descriptors)[..., 3, 3],
                                   self.operator.neighbor_widths(changed)[..., 3, 3])

    def test_quarter_turn_equivariance(self):
        """Horizontal and vertical axes must not acquire accidental implementation bias."""
        amplitude = torch.rand(1, 2, 5, 7)
        descriptors = torch.randn(1, 3, 5, 7)
        first = self.operator(amplitude, descriptors)
        second = self.operator(torch.rot90(amplitude, 1, (-2, -1)), torch.rot90(descriptors, 1, (-2, -1)))
        torch.testing.assert_close(second.unique, torch.rot90(first.unique, 1, (-2, -1)), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(second.context, torch.rot90(first.context, 1, (-2, -1)), atol=2e-6, rtol=2e-5)

    def test_amplitude_scale_invariance(self):
        """A common positive amplitude gain must not change the private fraction."""
        amplitude = torch.rand(1, 2, 5, 5) + 0.1
        descriptors = torch.randn(1, 3, 5, 5)
        first, second = self.operator(amplitude, descriptors), self.operator(amplitude * 3, descriptors)
        torch.testing.assert_close(first.private_fraction, second.private_fraction, atol=2e-6, rtol=2e-5)

    def test_numerical_gradient(self):
        """Check autograd against finite differences away from switching ties."""
        amplitude = (torch.rand(1, 1, 3, 3, dtype=torch.float64) + 0.1).requires_grad_()
        descriptors = torch.randn(1, 2, 3, 3, dtype=torch.float64, requires_grad=True)
        operator = DirectionalGrouping(GroupingConfig(radius=1))
        function = lambda a, d: operator(a, d).unique + operator(a, d).context * 0.1
        self.assertTrue(torch.autograd.gradcheck(function, (amplitude, descriptors), eps=1e-6, atol=1e-4, rtol=1e-3))

    def test_three_item_profile_matches_closed_form_quadrature(self):
        """Verify the center saddle and area ratio against an independent scalar formula."""
        width = 0.2
        operator = DirectionalGrouping(GroupingConfig(radius=1, min_width=width, max_width=width))
        amplitude = torch.ones(1, 1, 1, 3, dtype=torch.float64)
        descriptors = torch.ones(1, 2, 1, 3, dtype=torch.float64)
        value_at_center = 1 + 2 * math.exp(-1 / width)
        value_at_saddle = 2 * math.exp(-0.5 / width) + math.exp(-1.5 / width)
        value_at_neighbor = 1 + math.exp(-1 / width) + math.exp(-2 / width)
        numerator = 0.25 * (value_at_center - value_at_saddle)
        denominator = 0.25 * value_at_center + 0.5 * value_at_saddle + 0.25 * value_at_neighbor
        actual = float(operator(amplitude, descriptors).private_fraction[0, 0, 0, 1])
        self.assertAlmostEqual(actual, numerator / denominator, places=10)

    def test_reject_invalid_values_and_shape(self):
        """Invalid inputs must fail before producing plausible-looking statistics."""
        for bad in (-torch.ones(1, 1, 3, 3), torch.full((1, 1, 3, 3), float("nan"))):
            with self.assertRaises(ValueError):
                self.operator(bad, torch.ones(1, 2, 3, 3))
        with self.assertRaises(ValueError):
            self.operator(torch.ones(1, 2, 3, 3), torch.ones(2, 2, 3, 3))
        with self.assertRaises(ValueError):
            GroupingConfig(radius=0)

    def test_all_adapter_modes_are_identity_at_initialization(self):
        """Matched controls start from exactly the same unmodified encoder features."""
        image = torch.randn(2, 8, 5, 7)
        counts = []
        for mode in EvidenceAdapter.modes:
            adapter = EvidenceAdapter(8, 4, mode, GroupingConfig(radius=1))
            torch.testing.assert_close(adapter(image), image, rtol=0, atol=0)
            counts.append(sum(parameter.numel() for parameter in adapter.parameters()))
        self.assertEqual(len(set(counts)), 1)

    def test_control_modes_propagate_finite_gradients(self):
        """Every control must support optimization after the identity initialization."""
        for mode in EvidenceAdapter.modes:
            adapter = EvidenceAdapter(4, 3, mode, GroupingConfig(radius=1))
            optimizer = torch.optim.SGD(adapter.parameters(), lr=0.02)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                adapter(torch.randn(2, 4, 4, 4)).square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None))
                optimizer.step()
