import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from utils import (  # noqa: E402
    CameraCalibrationOptModule,
    CameraOptModule,
    CameraRefinementSchedule,
    CameraRigPoseModule,
    so3_exp_map,
)


class CameraRefinementTest(unittest.TestCase):
    def test_so3_identity_and_reference_pose_are_fixed(self):
        omega = torch.zeros((2, 3), requires_grad=True)
        rotation = so3_exp_map(omega)
        torch.testing.assert_close(rotation, torch.eye(3).repeat(2, 1, 1))

        module = CameraOptModule(2, reference_index=0)
        module.zero_init()
        camtoworlds = torch.eye(4).repeat(2, 1, 1)
        adjusted = module(camtoworlds, torch.tensor([0, 1]))
        torch.testing.assert_close(adjusted[0], camtoworlds[0])
        (adjusted[..., :3, 3].sum() + adjusted[..., :3, :3].sum()).backward()
        self.assertEqual(float(module.trans.weight.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(module.rot.weight.grad[0].abs().sum()), 0.0)

    def test_schedule_progressively_releases_parameters(self):
        schedule = CameraRefinementSchedule(500, 3000, 8000, 15000, -1, 20000)
        self.assertEqual(schedule.at(0, True, True).name, "warmup")
        self.assertTrue(schedule.at(500, True, True).pose)
        self.assertTrue(schedule.at(3000, True, True).focal)
        self.assertFalse(schedule.at(3000, True, True).principal)
        self.assertTrue(schedule.at(15000, True, True).radial_low)
        self.assertTrue(schedule.at(20000, True, True).frozen)

    def test_calibration_gradient_controls_freeze_high_order(self):
        module = CameraCalibrationOptModule(1, shared_focal=True)
        K = torch.tensor([[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]])
        radial = torch.zeros((1, 4))
        optimized_K, optimized_radial = module(K, radial, torch.tensor([0]))
        (optimized_K.sum() + optimized_radial.sum()).backward()
        module.apply_gradient_controls(
            focal_active=True,
            principal_active=True,
            radial_low_active=True,
            radial_high_active=False,
        )
        self.assertGreater(float(module.radial_deltas.weight.grad[0, :2].abs().sum()), 0.0)
        self.assertEqual(float(module.radial_deltas.weight.grad[0, 2:].abs().sum()), 0.0)

    def test_calibration_monotonicity_loss_is_zero_for_identity_mapping(self):
        module = CameraCalibrationOptModule(1)
        coeffs = torch.zeros((1, 4))
        self.assertEqual(float(module.monotonicity_loss(coeffs, 1.4)), 0.0)

    def test_rig_pose_preserves_camera_extrinsic(self):
        base = torch.eye(4).repeat(2, 1, 1)
        base[1, 0, 3] = 1.0
        frame_ids = torch.tensor([0, 0])
        camera_ids = torch.tensor([10, 20])
        module = CameraRigPoseModule(base, frame_ids, camera_ids)
        adjusted = module(base, frame_ids, camera_ids)
        torch.testing.assert_close(adjusted, base)


if __name__ == "__main__":
    unittest.main()
