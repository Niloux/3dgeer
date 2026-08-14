import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from simple_trainer import Config, Runner, _truncate_shN_for_ply, parse_config
from utils import CameraCalibrationOptModule
from gsplat import export_splats


class SimpleTrainerConfigTest(unittest.TestCase):
    def test_step_scaling_includes_pose_optimization_start(self):
        cfg = Config(max_steps=30_000, pose_opt_start_step=1_000)

        cfg.adjust_steps(0.5)

        self.assertEqual(cfg.max_steps, 15_000)
        self.assertEqual(cfg.pose_opt_start_step, 500)

    def test_camera_calibration_module_shares_physical_camera_parameters(self):
        module = CameraCalibrationOptModule(2)
        with torch.no_grad():
            module.focal_log_scales.weight[0] = torch.tensor([0.1, -0.1])
            module.principal_offsets.weight[0] = torch.tensor([0.01, -0.02])
            module.radial_deltas.weight[0] = torch.tensor(
                [0.001, -0.002, 0.003, -0.004]
            )
        K = torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]]
        ).repeat(2, 1, 1)
        radial = torch.zeros((2, 4))

        optimized_K, optimized_radial = module(
            K, radial, torch.tensor([0, 0])
        )

        self.assertTrue(torch.equal(optimized_K[0], optimized_K[1]))
        self.assertTrue(torch.equal(optimized_radial[0], optimized_radial[1]))
        detached_K = optimized_K.detach()
        self.assertGreater(float(detached_K[..., 0, 0].min()), 0.0)
        self.assertGreater(float(detached_K[..., 1, 1].min()), 0.0)
        self.assertAlmostEqual(float(detached_K[0, 0, 2]), 51.0)
        self.assertAlmostEqual(float(detached_K[0, 1, 2]), 57.6, places=5)
        torch.testing.assert_close(
            optimized_radial[0],
            torch.tensor([0.001, -0.002, 0.003, -0.004]),
        )

    def test_eval_without_test_split_writes_only_training_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = object.__new__(Runner)
            runner.cfg = SimpleNamespace(use_bilateral_grid=False)
            runner.world_rank = 0
            runner.valset = []
            runner.train_evalset = [object()]
            runner.splats = {"means": [object()]}
            runner.sky_splats = None
            runner.stats_dir = directory
            runner.writer = SimpleNamespace(
                add_scalar=lambda *args: None,
                flush=lambda: None,
            )
            calls = []

            def fake_eval(dataset, step, stage, apply_train_adjustment):
                calls.append((dataset, step, stage, apply_train_adjustment))
                return {
                    "psnr": 30.0,
                    "ssim": 0.9,
                    "lpips": 0.1,
                    "ellipse_time": 0.01,
                    "num_images": 1,
                }

            runner._eval_dataset = fake_eval

            runner.eval(step=9)

            self.assertEqual(calls, [(runner.train_evalset, 9, "train", True)])
            stats = json.loads((Path(directory) / "val_step0009.json").read_text())
            self.assertNotIn("psnr", stats)
            self.assertNotIn("num_val_images", stats)
            self.assertEqual(stats["train_psnr"], 30.0)
            self.assertEqual(stats["num_train_images"], 1)

    def test_yaml_values_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                """\
preset: default
save_ply: true
eval_steps: [10, 20]
use_test_split: false
init_type: lidar
init_use_knn_pca: true
calib_opt: true
calib_opt_radial_lr: 2.0e-6
pose_opt: true
pose_opt_start_step: 1000
tb_loss_window: 50
sky_enabled: true
sky_mask_dir: semantic_masks/sky
strategy:
  max_gaussians: 100
""",
                encoding="utf-8",
            )

            cfg = parse_config(
                ["--config", str(path), "--no-save-ply", "--max-steps", "30"]
            )

            self.assertFalse(cfg.save_ply)
            self.assertEqual(cfg.eval_steps, [10, 20])
            self.assertFalse(cfg.use_test_split)
            self.assertEqual(cfg.max_steps, 30)
            self.assertEqual(cfg.strategy.max_gaussians, 100)
            self.assertTrue(cfg.init_use_knn_pca)
            self.assertTrue(cfg.calib_opt)
            self.assertEqual(cfg.calib_opt_radial_lr, 2.0e-6)
            self.assertTrue(cfg.pose_opt)
            self.assertEqual(cfg.pose_opt_start_step, 1000)
            self.assertEqual(cfg.tb_loss_window, 50)
            self.assertTrue(cfg.sky_enabled)
            self.assertEqual(cfg.sky_mask_dir, "semantic_masks/sky")

    def test_ply_sh_degree_is_independent_from_training_sh_degree(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                """\
preset: default
sh_degree: 3
ply_sh_degree: 0
""",
                encoding="utf-8",
            )

            cfg = parse_config(["--config", str(path)])

            self.assertEqual(cfg.sh_degree, 3)
            self.assertEqual(cfg.ply_sh_degree, 0)

    def test_dc_only_ply_export_omits_higher_order_sh(self):
        shN = torch.randn(1, 15, 3)
        shN_dc_only = _truncate_shN_for_ply(shN, 0)

        data = export_splats(
            means=torch.zeros(1, 3),
            scales=torch.zeros(1, 3),
            quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            opacities=torch.zeros(1),
            sh0=torch.zeros(1, 1, 3),
            shN=shN_dc_only,
            format="ply",
        )
        header = data.split(b"end_header\n", 1)[0]

        self.assertEqual(tuple(shN_dc_only.shape), (1, 0, 3))
        self.assertIn(b"property float f_dc_2", header)
        self.assertNotIn(b"f_rest_", header)


if __name__ == "__main__":
    unittest.main()
