import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from evaluation import masked_psnr, masked_ssim  # noqa: E402


class _ScalarMetric:
    def __init__(self):
        self.reset_count = 0
        self.prediction = None
        self.target = None

    def reset(self):
        self.reset_count += 1

    def __call__(self, prediction, target):
        self.prediction = prediction.clone()
        self.target = target.clone()
        return prediction.new_tensor(0.75)


class EvaluationMetricsTest(unittest.TestCase):
    def test_no_sky_psnr_uses_legacy_black_padded_full_frame(self):
        target = torch.zeros((1, 2, 2, 3))
        prediction = target.clone()
        prediction[:, 0, 0] = 1.0
        full_mask = torch.ones((1, 2, 2), dtype=torch.bool)
        no_sky_mask = full_mask.clone()
        no_sky_mask[:, 0, 0] = False

        torch.testing.assert_close(
            masked_psnr(prediction, target, full_mask),
            torch.tensor(6.0206),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertTrue(torch.isinf(masked_psnr(prediction, target, no_sky_mask)))

    def test_masked_ssim_black_pads_ignored_pixels_and_resets_state(self):
        metric = _ScalarMetric()
        target = torch.ones((1, 2, 2, 3))
        prediction = target.clone()
        valid = torch.ones((1, 2, 2), dtype=torch.bool)
        valid[:, 0, 0] = False

        value = masked_ssim(metric, prediction, target, valid)

        torch.testing.assert_close(value, torch.tensor(0.75))
        self.assertEqual(metric.reset_count, 2)
        self.assertTrue(
            torch.equal(
                metric.prediction[:, :, 0, 0],
                torch.zeros_like(metric.prediction[:, :, 0, 0]),
            )
        )
        self.assertTrue(
            torch.equal(
                metric.target[:, :, 0, 0],
                torch.zeros_like(metric.target[:, :, 0, 0]),
            )
        )


if __name__ == "__main__":
    unittest.main()
