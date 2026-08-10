import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from gaussian_models import (  # noqa: E402
    composite_sky,
    create_sky_splats_with_optimizers,
    initialize_surface_priors_knn_pca,
    sky_hemisphere,
)
from gsplat.utils import normalized_quat_to_rotmat  # noqa: E402


class GaussianModelsTest(unittest.TestCase):
    def test_knn_pca_aligns_planar_normals_and_keeps_scales_positive(self):
        x, y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 10),
            torch.linspace(-1.0, 1.0, 10),
            indexing="ij",
        )
        points = torch.stack((x.flatten(), y.flatten(), torch.zeros(100)), dim=1)

        rotations, scales, accepted_count = initialize_surface_priors_knn_pca(
            points,
            k=16,
            local_scale_factor=1.0,
            normal_scale_factor=0.25,
            planarity_threshold=0.3,
            curvature_threshold=0.1,
        )

        accepted = scales[:, 2] < scales[:, :2].amin(dim=1)
        rotation_matrices = normalized_quat_to_rotmat(rotations)
        self.assertEqual(accepted_count, int(accepted.sum()))
        self.assertGreater(accepted_count, 0)
        self.assertTrue(torch.all(rotation_matrices[accepted, 2, 2].abs() > 0.999))
        self.assertTrue(torch.all(scales > 0.0))

    def test_knn_pca_duplicate_fallback_avoids_cuda_eigh_and_log_zero(self):
        with mock.patch.object(
            torch.linalg,
            "eigh",
            side_effect=torch._C._LinAlgError("CUSOLVER_STATUS_INVALID_VALUE"),
        ):
            _, scales, accepted_count = initialize_surface_priors_knn_pca(
                torch.zeros((4, 3)),
                k=4,
                local_scale_factor=1.0,
                normal_scale_factor=0.25,
                planarity_threshold=0.3,
                curvature_threshold=0.1,
            )

        self.assertEqual(accepted_count, 0)
        self.assertTrue(torch.isfinite(torch.log(scales)).all())

    def test_gaussian_sky_has_fixed_geometry_and_trainable_sh(self):
        xyz, scales = sky_hemisphere(32, 10.0, 42)
        repeated_xyz, repeated_scales = sky_hemisphere(32, 10.0, 42)
        np.testing.assert_allclose(xyz, repeated_xyz)
        np.testing.assert_allclose(scales, repeated_scales)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 10.0, rtol=1e-6)
        self.assertTrue(np.all(xyz[:, 2] > 0.0))
        self.assertTrue(np.all(scales > 0.0))

        splats, optimizers = create_sky_splats_with_optimizers(
            count=32,
            radius=10.0,
            initial_opacity=0.7,
            sh_degree=0,
            sh0_lr=0.01,
            shN_lr=0.0,
            seed=42,
            device="cpu",
            world_rank=0,
            world_size=1,
        )
        splats["sh0"].mean().backward()
        self.assertIsNotNone(splats["sh0"].grad)
        for name in ("means", "scales", "quats", "opacities"):
            self.assertFalse(splats[name].requires_grad)
            self.assertIsNone(splats[name].grad)

        before = splats["sh0"].detach().clone()
        optimizers["sh0"].step()
        self.assertFalse(torch.equal(before, splats["sh0"]))

    def test_sky_is_composited_behind_foreground(self):
        foreground = torch.full((1, 2, 2, 3), 0.2)
        alpha = torch.full((1, 2, 2, 1), 0.25)
        sky = torch.full_like(foreground, 0.8)
        expected = torch.full_like(foreground, 0.8)
        torch.testing.assert_close(composite_sky(foreground, alpha, sky), expected)


if __name__ == "__main__":
    unittest.main()
