import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from simple_trainer import Runner, parse_config


class SimpleTrainerConfigTest(unittest.TestCase):
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
            self.assertTrue(cfg.sky_enabled)
            self.assertEqual(cfg.sky_mask_dir, "semantic_masks/sky")


if __name__ == "__main__":
    unittest.main()
