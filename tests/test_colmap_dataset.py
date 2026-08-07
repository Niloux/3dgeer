import unittest

import numpy as np

from examples.datasets.colmap import _fisheye_fov_mask


class ColmapDatasetTest(unittest.TestCase):
    def test_fisheye_fov_mask_stays_in_front_hemisphere(self):
        K = np.array([[1.0, 0.0, 2.5], [0.0, 1.0, 2.5], [0.0, 0.0, 1.0]])
        mask = _fisheye_fov_mask(K, np.zeros(4), 5, 5, 178.0)

        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[0, 0])
        with self.assertRaises(ValueError):
            _fisheye_fov_mask(K, np.zeros(4), 5, 5, 180.0)


if __name__ == "__main__":
    unittest.main()
