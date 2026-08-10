import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.datasets.colmap import (
    _combine_supervision_masks,
    _fisheye_fov_mask,
    _resolve_semantic_mask_path,
)


class ColmapDatasetTest(unittest.TestCase):
    def test_fisheye_fov_mask_stays_in_front_hemisphere(self):
        K = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, 2.5], [0.0, 0.0, 1.0]])
        mask = _fisheye_fov_mask(K, np.zeros(4), 5, 5, 178.0)

        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])
        with self.assertRaises(ValueError):
            _fisheye_fov_mask(K, np.zeros(4), 5, 5, 180.0)

    def test_sky_restores_supervision_only_inside_camera_validity(self):
        camera = np.array([[True, True], [False, True]])
        training = np.array([[True, False], [False, True]])
        sky = np.array([[False, True], [True, False]])

        valid, valid_sky = _combine_supervision_masks(camera, training, sky)

        np.testing.assert_array_equal(valid, np.array([[True, True], [False, True]]))
        np.testing.assert_array_equal(
            valid_sky, np.array([[False, True], [False, False]])
        )

    def test_flat_sky_mask_resolves_for_nested_colmap_name(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "frame_00001-L.png"
            expected.write_bytes(b"mask")
            actual = _resolve_semantic_mask_path(
                directory, "L/frame_00001-L.JPG"
            )
            self.assertEqual(actual, str(expected))


if __name__ == "__main__":
    unittest.main()
