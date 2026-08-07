import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from simple_trainer import parse_config


class SimpleTrainerConfigTest(unittest.TestCase):
    def test_yaml_values_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.yaml"
            path.write_text(
                """\
preset: default
save_ply: true
eval_steps: [10, 20]
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
            self.assertEqual(cfg.max_steps, 30)
            self.assertEqual(cfg.strategy.max_gaussians, 100)


if __name__ == "__main__":
    unittest.main()
