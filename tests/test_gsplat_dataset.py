import json
import struct
import tempfile
import unittest
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from examples.datasets.gsplat import Dataset, Parser


class GsplatDatasetTest(unittest.TestCase):
    def test_manifest_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            (root / "pointcloud").mkdir()
            imageio.imwrite(
                root / "images/frame.png", np.full((4, 4, 3), 64, np.uint8)
            )
            imageio.imwrite(
                root / "masks/frame.png", np.eye(4, dtype=np.uint8) * 255
            )

            with (root / "pointcloud/points3D.ply").open("wb") as f:
                f.write(
                    b"ply\nformat binary_little_endian 1.0\n"
                    b"element vertex 2\nproperty float x\nproperty float y\n"
                    b"property float z\nproperty uchar red\nproperty uchar green\n"
                    b"property uchar blue\nend_header\n"
                )
                f.write(
                    struct.pack(
                        "<fffBBBfffBBB", 0, 0, 0, 1, 2, 3, 1, 2, 3, 4, 5, 6
                    )
                )

            camera = {
                "model": "fisheye",
                "width": 4,
                "height": 4,
                "K": [[2, 0, 2], [0, 2, 2], [0, 0, 1]],
                "radial_coeffs": [0.1, 0.2, 0.3, 0.4],
            }
            frame = {
                "image": "images/frame.png",
                "mask": "masks/frame.png",
                "camera": "left",
                "world_to_camera": np.eye(4).tolist(),
            }
            manifest = {
                "format": "gsdata-fisheye-v6",
                "pointcloud": "pointcloud/points3D.ply",
                "cameras": {"left": camera},
                "frames": [
                    {**frame, "split": "train"},
                    {**frame, "split": "validation"},
                ],
            }
            (root / "dataset.json").write_text(json.dumps(manifest))

            parser = Parser(str(root), factor=2)
            sample = Dataset(parser, "train")[0]
            self.assertEqual(parser.points.shape, (2, 3))
            self.assertEqual(parser.points_rgb.tolist(), [[1, 2, 3], [4, 5, 6]])
            self.assertEqual(sample["image"].shape, (2, 2, 3))
            self.assertEqual(sample["mask"].dtype, torch.bool)
            self.assertEqual(sample["K"][0, 0], 1)
            np.testing.assert_allclose(
                sample["radial_coeffs"].numpy(), camera["radial_coeffs"]
            )


if __name__ == "__main__":
    unittest.main()
