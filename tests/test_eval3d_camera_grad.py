"""Gradient checks for the eval3d world-space rasterizer."""

import unittest

import torch

from gsplat.rendering import rasterization


def _viewmat_from_delta(delta: torch.Tensor) -> torch.Tensor:
    """Build a global-shutter world-to-camera pose from tx and yaw."""
    tx, yaw = delta.unbind()
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rotation = torch.stack(
        (
            cos_yaw,
            zero,
            sin_yaw,
            zero,
            one,
            zero,
            -sin_yaw,
            zero,
            cos_yaw,
        )
    ).reshape(3, 3)
    translation = torch.stack((tx, zero, zero)).unsqueeze(-1)
    upper = torch.cat((rotation, translation), dim=-1)
    bottom = torch.stack((zero, zero, zero, one)).unsqueeze(0)
    return torch.cat((upper, bottom), dim=0).unsqueeze(0)


def _eval3d_objective_from_viewmats(
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None = None,
    radial_coeffs: torch.Tensor | None = None,
    wide_angle: bool = False,
) -> torch.Tensor:
    device = viewmats.device
    dtype = viewmats.dtype
    if wide_angle:
        means = torch.tensor(
            [
                [-1.85, 0.30, 2.00],
                [1.90, -0.35, 2.10],
                [0.25, 1.45, 1.90],
                [-0.30, -1.50, 2.00],
            ],
            device=device,
            dtype=dtype,
        )
        scales = torch.tensor(
            [
                [0.34, 0.28, 0.30],
                [0.32, 0.36, 0.28],
                [0.30, 0.34, 0.26],
                [0.36, 0.30, 0.28],
            ],
            device=device,
            dtype=dtype,
        )
        opacities = torch.tensor([0.82, 0.74, 0.78, 0.70], device=device, dtype=dtype)
        colors = torch.tensor(
            [
                [0.9, 0.2, 0.1],
                [0.1, 0.35, 0.85],
                [0.2, 0.85, 0.25],
                [0.75, 0.15, 0.7],
            ],
            device=device,
            dtype=dtype,
        )
    else:
        means = torch.tensor(
            [[-0.30, 0.12, 3.0], [0.42, -0.18, 4.0]], device=device, dtype=dtype
        )
        scales = torch.tensor(
            [[0.24, 0.18, 0.20], [0.22, 0.28, 0.18]], device=device, dtype=dtype
        )
        opacities = torch.tensor([0.82, 0.74], device=device, dtype=dtype)
        colors = torch.tensor(
            [[0.9, 0.2, 0.1], [0.1, 0.35, 0.85]], device=device, dtype=dtype
        )
    quats = torch.zeros((len(means), 4), device=device, dtype=dtype)
    quats[:, 0] = 1.0
    width, height = 32, 24
    if Ks is None:
        Ks = torch.tensor(
            [[[19.0, 0.0, 16.0], [0.0, 18.0, 12.0], [0.0, 0.0, 1.0]]],
            device=device,
            dtype=dtype,
        )
    if radial_coeffs is None:
        radial_coeffs = torch.tensor(
            [[-0.025, 0.004, -0.0005, 0.00005]], device=device, dtype=dtype
        )

    renders, alphas, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        packed=False,
        camera_model="fisheye",
        radial_coeffs=radial_coeffs,
        with_geer=True,
        with_eval3d=True,
    )
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    ys = torch.linspace(-0.7, 0.9, height, device=device, dtype=dtype)
    weights = (ys[:, None] + 1.3 * xs[None, :]).unsqueeze(0).unsqueeze(-1)
    return (renders * weights).mean() + 0.25 * (alphas * weights).mean()


def _eval3d_objective(delta: torch.Tensor) -> torch.Tensor:
    return _eval3d_objective_from_viewmats(_viewmat_from_delta(delta))


class Eval3DCameraGradientTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA device")
    def test_global_shutter_fisheye_pose_vjp_matches_finite_difference(self):
        delta = torch.tensor([0.04, 0.025], device="cuda", requires_grad=True)
        objective = _eval3d_objective(delta)
        (analytic,) = torch.autograd.grad(objective, delta)

        eps = 1e-3
        numeric = []
        with torch.no_grad():
            for axis in range(delta.numel()):
                step = torch.zeros_like(delta)
                step[axis] = eps
                numeric.append(
                    (
                        _eval3d_objective(delta + step)
                        - _eval3d_objective(delta - step)
                    )
                    / (2.0 * eps)
                )
        numeric = torch.stack(numeric)

        self.assertTrue(torch.isfinite(analytic).all())
        self.assertGreater(analytic.abs().min().item(), 1e-5)
        torch.testing.assert_close(analytic, numeric, rtol=3e-2, atol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA device")
    def test_camera_opt_module_receives_eval3d_pose_gradient(self):
        from examples.utils import CameraOptModule

        pose_adjust = CameraOptModule(1).cuda()
        pose_adjust.zero_init()
        camtoworlds = torch.eye(4, device="cuda").unsqueeze(0)
        image_ids = torch.zeros(1, device="cuda", dtype=torch.long)

        adjusted_camtoworlds = pose_adjust(camtoworlds, image_ids)
        objective = _eval3d_objective_from_viewmats(
            torch.linalg.inv(adjusted_camtoworlds)
        )
        objective.backward()

        pose_grad = pose_adjust.embeds.weight.grad
        self.assertIsNotNone(pose_grad)
        self.assertTrue(torch.isfinite(pose_grad).all())
        self.assertGreater(pose_grad.norm().item(), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA device")
    def test_global_shutter_fisheye_calibration_vjp_matches_finite_difference(self):
        viewmats = torch.eye(4, device="cuda").unsqueeze(0)
        Ks = torch.tensor(
            [[[19.0, 0.0, 16.0], [0.0, 18.0, 12.0], [0.0, 0.0, 1.0]]],
            device="cuda",
            requires_grad=True,
        )
        radial_coeffs = torch.tensor(
            [[-0.025, 0.004, -0.0005, 0.00005]],
            device="cuda",
            requires_grad=True,
        )
        objective = _eval3d_objective_from_viewmats(
            viewmats, Ks, radial_coeffs, wide_angle=True
        )
        analytic_Ks, analytic_radial = torch.autograd.grad(
            objective, (Ks, radial_coeffs)
        )
        analytic_intrinsics = analytic_Ks[0, (0, 1, 0, 1), (0, 1, 2, 2)]

        numeric_intrinsics = []
        intrinsic_indices = ((0, 0), (1, 1), (0, 2), (1, 2))
        with torch.no_grad():
            for row, col in intrinsic_indices:
                step = torch.zeros_like(Ks)
                step[0, row, col] = 2e-2
                numeric_intrinsics.append(
                    (
                        _eval3d_objective_from_viewmats(
                            viewmats, Ks + step, radial_coeffs, wide_angle=True
                        )
                        - _eval3d_objective_from_viewmats(
                            viewmats, Ks - step, radial_coeffs, wide_angle=True
                        )
                    )
                    / (4e-2)
                )

            numeric_radial = []
            radial_eps = (1e-3, 2e-3, 5e-3, 1e-2)
            for axis, eps in enumerate(radial_eps):
                step = torch.zeros_like(radial_coeffs)
                step[0, axis] = eps
                numeric_radial.append(
                    (
                        _eval3d_objective_from_viewmats(
                            viewmats, Ks, radial_coeffs + step, wide_angle=True
                        )
                        - _eval3d_objective_from_viewmats(
                            viewmats, Ks, radial_coeffs - step, wide_angle=True
                        )
                    )
                    / (2.0 * eps)
                )

        numeric_intrinsics = torch.stack(numeric_intrinsics)
        numeric_radial = torch.stack(numeric_radial)
        self.assertTrue(torch.isfinite(analytic_intrinsics).all())
        self.assertTrue(torch.isfinite(analytic_radial).all())
        torch.testing.assert_close(
            analytic_intrinsics, numeric_intrinsics, rtol=5e-2, atol=2e-4
        )
        torch.testing.assert_close(
            analytic_radial.flatten(), numeric_radial, rtol=7e-2, atol=3e-4
        )

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA device")
    def test_standard_rasterizer_viewmat_gradient_still_works(self):
        viewmats = torch.eye(4, device="cuda").unsqueeze(0).requires_grad_()
        means = torch.tensor([[0.0, 0.0, 3.0]], device="cuda")
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
        scales = torch.full((1, 3), 0.1, device="cuda")
        opacities = torch.full((1,), 0.5, device="cuda")
        colors = torch.full((1, 3), 0.5, device="cuda")
        Ks = torch.tensor(
            [[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]],
            device="cuda",
        )
        renders, alphas, _ = rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            16,
            16,
            packed=False,
            camera_model="fisheye",
        )
        (renders.sum() + alphas.sum()).backward()

        self.assertIsNotNone(viewmats.grad)
        self.assertTrue(torch.isfinite(viewmats.grad).all())
        self.assertGreater(viewmats.grad.norm().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
