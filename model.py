from __future__ import annotations
import math, ctypes
from pathlib import Path
from typing import Dict, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import spline_filter

LIBRARY = Path(__file__).resolve().parent/'cubic_pull.so'

cache = {}


def pull_resample_dwi(source, u, inverse):
    # Pull x+u in world mm; interpolation is in intensity space.
    key = (source.data_ptr(), tuple(source.shape), tuple(source.stride()))
    if key not in cache:
        cache[key] = (source, FixedCubicSampler(source, inverse, threads=torch.get_num_threads()))
    return cache[key][1](u)
def _as_shape3(shape: Sequence[int]):
    result = tuple(int(v) for v in shape)
    if len(result) != 3 or any(v < 2 for v in result):
        raise ValueError(f"Expected a three-dimensional shape >= 2, got {shape}")
    return result

def cubic_bspline_basis_matrix(
    n_voxels: int,
    voxel_axis_mm: float,
    control_point_spacing_mm: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
):

    if voxel_axis_mm <= 0 or control_point_spacing_mm <= 0:
        raise ValueError("Voxel size and control-point spacing must be positive")

    physical_extent_mm = (n_voxels - 1) * voxel_axis_mm
    n_control_points = int(math.ceil(physical_extent_mm / control_point_spacing_mm)) + 4

    voxel_positions_mm = torch.arange(n_voxels, device=device, dtype=dtype) * voxel_axis_mm
    lattice_position = voxel_positions_mm / control_point_spacing_mm + 1.0
    lattice_floor = torch.floor(lattice_position).to(torch.long)
    fraction = lattice_position - lattice_floor.to(dtype)

    one_minus = 1.0 - fraction
    weights = torch.stack(
        [
            one_minus.pow(3) / 6.0,
            (3.0 * fraction.pow(3) - 6.0 * fraction.pow(2) + 4.0) / 6.0,
            (-3.0 * fraction.pow(3) + 3.0 * fraction.pow(2) + 3.0 * fraction + 1.0) / 6.0,
            fraction.pow(3) / 6.0,
        ],
        dim=1,
    )
    indices = lattice_floor[:, None] + torch.tensor(
        [-1, 0, 1, 2], device=device, dtype=torch.long
    )[None, :]
    if int(indices.min()) < 0 or int(indices.max()) >= n_control_points:
        raise RuntimeError("Internal cubic B-spline lattice padding error")

    basis = torch.zeros((n_voxels, n_control_points), device=device, dtype=dtype)
    basis.scatter_(1, indices, weights)
    return basis, n_control_points

class CubicBSplineFFD(torch.nn.Module):

    def __init__(
        self,
        n_directions: int,
        spatial_shape: Sequence[int],
        affine: np.ndarray,
        control_point_spacing_mm: float = 15.0,
        max_displacement_mm: float = 3.0,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_directions = int(n_directions)
        self.spatial_shape = _as_shape3(spatial_shape)
        self.control_point_spacing_mm = float(control_point_spacing_mm)
        self.max_displacement_mm = float(max_displacement_mm)
        if self.n_directions < 1:
            raise ValueError("At least one diffusion direction is required")
        if self.max_displacement_mm <= 0:
            raise ValueError("Maximum displacement must be positive")

        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("affine must be a finite 4x4 matrix")
        spatial_affine = affine[:3, :3]
        if abs(np.linalg.det(spatial_affine)) < 1e-10:
            raise ValueError("NIfTI spatial affine is singular")

        voxel_axis_mm = np.linalg.norm(spatial_affine, axis=0)

        torch_device = torch.device(device)
        bx, cpx = cubic_bspline_basis_matrix(
            self.spatial_shape[0], voxel_axis_mm[0], self.control_point_spacing_mm,
            device=torch_device, dtype=dtype,
        )
        by, cpy = cubic_bspline_basis_matrix(
            self.spatial_shape[1], voxel_axis_mm[1], self.control_point_spacing_mm,
            device=torch_device, dtype=dtype,
        )
        bz, cpz = cubic_bspline_basis_matrix(
            self.spatial_shape[2], voxel_axis_mm[2], self.control_point_spacing_mm,
            device=torch_device, dtype=dtype,
        )
        self.control_grid_shape = (cpx, cpy, cpz)
        self.register_buffer("basis_x", bx)
        self.register_buffer("basis_y", by)
        self.register_buffer("basis_z", bz)
        self.register_buffer(
            "voxel_axis_mm", torch.as_tensor(voxel_axis_mm, device=torch_device, dtype=dtype)
        )
        self.register_buffer(
            "spatial_affine", torch.as_tensor(spatial_affine, device=torch_device, dtype=dtype)
        )
        self.register_buffer(
            "inverse_spatial_affine",
            torch.as_tensor(np.linalg.inv(spatial_affine), device=torch_device, dtype=dtype),
        )

        self.raw_coefficients = torch.nn.Parameter(
            torch.zeros(
                (self.n_directions, 3, cpx, cpy, cpz),
                device=torch_device,
                dtype=dtype,
            )
        )

    def dense_raw_field(self):

        along_x = torch.einsum("dcijk,xi->dcxjk", self.raw_coefficients, self.basis_x)
        along_xy = torch.einsum("dcxjk,yj->dcxyk", along_x, self.basis_y)
        return torch.einsum("dcxyk,zk->dcxyz", along_xy, self.basis_z)

    def forward(self):

        raw = self.dense_raw_field()
        max_disp = torch.as_tensor(
            self.max_displacement_mm, dtype=raw.dtype, device=raw.device
        )
        denominator = torch.sqrt(max_disp.square() + raw.square().sum(dim=1, keepdim=True))
        return max_disp * raw / denominator

    def initialize_from_dense_displacement(self, displacement_world_mm: torch.Tensor):

        expected = (self.n_directions, 3, *self.spatial_shape)
        if tuple(displacement_world_mm.shape) != expected:
            raise ValueError(
                f"Initial dense displacement has shape {tuple(displacement_world_mm.shape)}; "
                f"expected {expected}"
            )
        target = displacement_world_mm.to(
            device=self.raw_coefficients.device, dtype=self.raw_coefficients.dtype
        )
        if not torch.isfinite(target).all():
            raise ValueError("Initial dense displacement contains NaN or Inf")

        max_disp = torch.as_tensor(
            self.max_displacement_mm, device=target.device, dtype=target.dtype
        )
        target_norm = torch.linalg.vector_norm(target, dim=1, keepdim=True)
        if torch.any(target_norm >= max_disp):
            raise ValueError(
                "Initial displacement reaches/exceeds the destination maximum; "
                "use an equal or larger --max-displacement-mm"
            )

        raw_target = max_disp * target / torch.sqrt(
            (max_disp.square() - target_norm.square()).clamp_min(
                torch.finfo(target.dtype).eps
            )
        )

        pinv_x = torch.linalg.pinv(self.basis_x)
        pinv_y = torch.linalg.pinv(self.basis_y)
        pinv_z = torch.linalg.pinv(self.basis_z)
        along_x = torch.einsum("ix,dcxyz->dciyz", pinv_x, raw_target)
        along_xy = torch.einsum("jy,dciyz->dcijz", pinv_y, along_x)
        fitted_coefficients = torch.einsum("kz,dcijz->dcijk", pinv_z, along_xy)
        with torch.no_grad():
            self.raw_coefficients.copy_(fitted_coefficients)
            reproduced = self.forward()
            error = reproduced - target
            return {
                "dense_rmse_mm": float(torch.sqrt(error.square().mean()).item()),
                "dense_max_abs_error_mm": float(error.abs().max().item()),
                "source_max_displacement_mm": float(target_norm.max().item()),
                "initialized_max_displacement_mm": float(
                    torch.linalg.vector_norm(reproduced, dim=1).max().item()
                ),
            }

def jacobian_determinant(displacement_world_mm: torch.Tensor, inverse_spatial_affine: torch.Tensor):

    if displacement_world_mm.ndim != 5 or displacement_world_mm.shape[1] != 3:
        raise ValueError("Expected displacement field [D,3,X,Y,Z]")
    du_di = torch.stack(
        torch.gradient(displacement_world_mm, dim=(2, 3, 4), edge_order=1), dim=-1
    )
    du_di = du_di.permute(0, 2, 3, 4, 1, 5)
    inverse_spatial_affine = inverse_spatial_affine.to(
        device=du_di.device, dtype=du_di.dtype
    )
    du_dx_world = torch.matmul(du_di, inverse_spatial_affine)
    identity = torch.eye(3, device=du_di.device, dtype=du_di.dtype)
    return torch.linalg.det(du_dx_world + identity)

class FixedCubicSampler:
    def __init__(self,source,inverse,threads=2):
        assert source.device.type=='cpu' and inverse.device.type=='cpu'
        if source.requires_grad:raise ValueError('This sampler requires fixed source intensities')
        self.source=np.ascontiguousarray(source.detach().numpy(),dtype=np.float64)
        self.inverse=np.ascontiguousarray(inverse.detach().numpy(),dtype=np.float64)
        self.shape=source.shape;self.pad=12;self.threads=threads
        self.coefficients=np.stack([spline_filter(np.pad(s,self.pad,mode='edge'),order=3,mode='nearest') for s in self.source])
        self.lib=ctypes.CDLL(str(LIBRARY));self.call=self.lib.cubic_pull
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.call.argtypes=[ptr,ptr,ptr,ptr]+[ctypes.c_int]*6+[ctypes.c_bool,ptr,ptr,np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')]
        self.call.restype=None

    def evaluate(self,u,need_gradient):
        assert u.device.type=='cpu' and u.shape==(self.shape[0],3,*self.shape[1:])
        displacement=np.ascontiguousarray(u.detach().numpy(),dtype=np.float64)
        if not np.isfinite(displacement).all():raise FloatingPointError('Nonfinite displacement')
        out=np.empty(self.shape,dtype=np.float64)
        gradient=np.empty(displacement.shape if need_gradient else (1,),dtype=np.float64)
        valid=np.empty(self.shape,dtype=np.uint8)
        self.call(self.coefficients,self.source,displacement,self.inverse,*self.shape,self.pad,self.threads,need_gradient,out,gradient,valid)
        return torch.from_numpy(out).to(u.dtype),torch.from_numpy(gradient).to(u.dtype),torch.from_numpy(valid.astype(bool))

    def __call__(self,u):return CubicPull.apply(u,self)

class CubicPull(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,sampler):
        out,gradient,valid=sampler.evaluate(u,ctx.needs_input_grad[0])
        ctx.save_for_backward(gradient);ctx.mark_non_differentiable(valid)
        return out,valid

    @staticmethod
    def backward(ctx,upstream,_):
        gradient,=ctx.saved_tensors
        return upstream[:,None]*gradient,None

class JointBiasGeometryDSWModel(torch.nn.Module):

    COMPONENT_NAMES = ("fa", "v1", "bias_coeff", "disp", "bend")

    def __init__(
        self,
        b0_images: np.ndarray,
        diffusion_images: np.ndarray,
        fit_mask: np.ndarray,
        bvecs: np.ndarray,
        bval: float | np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        v1_mu: np.ndarray,
        kappa: np.ndarray,
        affine: np.ndarray,
        *,
        control_point_spacing_mm: float = 30.0,
        max_displacement_mm: float = 20.0,
        bias_basis: str = "bspline",
        bias_control_point_spacing_mm: float | None = None,
        lambda_fa: float = .05,
        lambda_v1: float = .2,
        lambda_bias_coeff: float = .2,
        lambda_disp: float = .0025,
        lambda_bend: float = .625,
        bias_regularization_domain: str = "field",
        normalize_atlas_priors: bool = False,
        force_sn_lower_than_s0: bool = True,
        deform_b0: bool = True,
        shared_b0_deformation: bool = True,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        device = torch.device(device)
        b0_images = np.asarray(b0_images)
        diffusion_images = np.asarray(diffusion_images)
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if b0_images.ndim != 4 or b0_images.shape[-1] < 1:
            raise ValueError("b0_images must have shape [X,Y,Z,Nlow] with Nlow >= 1")
        if diffusion_images.ndim != 4 or diffusion_images.shape[-1] < 6:
            raise ValueError("diffusion_images must have shape [X,Y,Z,Ndwi] with Ndwi >= 6")
        spatial_shape = diffusion_images.shape[:3]
        if b0_images.shape[:3] != spatial_shape or fit_mask.shape != spatial_shape:
            raise ValueError("b0 images, DWIs, and fit mask must share one grid")
        if not fit_mask.any():
            raise ValueError("Joint fit mask is empty")
        self.spatial_shape = tuple(int(v) for v in spatial_shape)
        self.n_directions = diffusion_images.shape[-1]
        self.n_b0 = b0_images.shape[-1]
        bvalues = np.broadcast_to(np.asarray(bval,dtype=float),(self.n_directions,)).copy()
        if not np.isfinite(bvalues).all() or (bvalues<=0).any():
            raise ValueError("DWI b-values must be finite and positive")
        self.register_buffer('bval',torch.as_tensor(bvalues[:,None],device=device,dtype=dtype))
        self.bias_basis = str(bias_basis)
        if self.bias_basis != "bspline":
            raise ValueError("This package supports only the accepted cubic B-spline bias model")
        self.bias_control_point_spacing_mm = float(
            control_point_spacing_mm
            if bias_control_point_spacing_mm is None
            else bias_control_point_spacing_mm
        )
        if (
            not math.isfinite(self.bias_control_point_spacing_mm)
            or self.bias_control_point_spacing_mm <= 0
        ):
            raise ValueError("Bias control-point spacing must be finite and positive")
        self.normalize_atlas_priors = bool(normalize_atlas_priors)
        self.force_sn_lower_than_s0 = bool(force_sn_lower_than_s0)
        self.deform_b0 = bool(deform_b0)
        self.shared_b0_deformation = bool(shared_b0_deformation)
        if self.shared_b0_deformation and not self.deform_b0:
            raise ValueError("A shared b0 deformation requires deform_b0=True")
        directions = tuple(f'dwi_direction_{i+1}' for i in range(self.n_directions))
        if self.shared_b0_deformation:
            self.FIELD_ORDER = ('shared_b0',) + directions
        elif self.deform_b0:
            self.FIELD_ORDER = tuple(f'b0_volume_{i+1}' for i in range(1,self.n_b0)) + directions
        else:
            self.FIELD_ORDER = directions

        self.lambdas = {
            "fa": float(lambda_fa),
            "v1": float(lambda_v1),
            "bias_coeff": float(lambda_bias_coeff),
            "disp": float(lambda_disp),
            "bend": float(lambda_bend),
        }
        if bias_regularization_domain != "field":
            raise ValueError(
                "This package penalizes the projected bias field, not coefficients"
            )
        self.bias_regularization_domain = bias_regularization_domain
        if any(not math.isfinite(value) or value < 0 for value in self.lambdas.values()):
            raise ValueError("All joint-objective weights must be finite and nonnegative")

        self.ffd = CubicBSplineFFD(
            len(self.FIELD_ORDER),
            self.spatial_shape,
            affine,
            control_point_spacing_mm=control_point_spacing_mm,
            max_displacement_mm=max_displacement_mm,
            device=device,
            dtype=dtype,
        )
        self.register_buffer(
            "source_b0", torch.as_tensor(b0_images, device=device, dtype=dtype).permute(3, 0, 1, 2)
        )
        self.register_buffer(
            "source_dwi",
            torch.as_tensor(diffusion_images, device=device, dtype=dtype).permute(3, 0, 1, 2),
        )
        self.register_buffer("fit_mask", torch.as_tensor(fit_mask, device=device, dtype=torch.bool))

        bvecs = np.asarray(bvecs, dtype=np.float64)
        if bvecs.shape == (self.n_directions, 3):
            bvecs = bvecs.T
        if bvecs.shape != (3, self.n_directions):
            raise ValueError(f"Expected b-vectors [3,{self.n_directions}], got {bvecs.shape}")
        norms = np.linalg.norm(bvecs, axis=0)
        if not np.isfinite(bvecs).all() or np.any(norms < 1e-8):
            raise ValueError("All diffusion b-vectors must be finite and nonzero")
        bvecs_t = torch.as_tensor(bvecs / norms[None, :], device=device, dtype=dtype)
        self.register_buffer("bvecs", bvecs_t)
        design = torch.stack(
            (
                bvecs_t[0].square(),
                bvecs_t[1].square(),
                bvecs_t[2].square(),
                2.0 * bvecs_t[0] * bvecs_t[1],
                2.0 * bvecs_t[0] * bvecs_t[2],
                2.0 * bvecs_t[1] * bvecs_t[2],
            ),
            dim=1,
        )
        if int(torch.linalg.matrix_rank(design).item()) < 6:
            raise ValueError("Tensor design matrix is rank deficient")
        self.register_buffer("design_matrix", design)
        self.register_buffer("design_pinv", torch.linalg.pinv(design))

        eps = torch.finfo(dtype).eps
        for name, volume in (("alpha", alpha), ("beta", beta), ("kappa", kappa)):
            if np.asarray(volume).shape != self.spatial_shape:
                raise ValueError(f"{name} atlas does not share the DTI grid")
        if np.asarray(v1_mu).shape != (*self.spatial_shape, 3):
            raise ValueError("v1_mu atlas must have shape [X,Y,Z,3]")
        alpha_vec = torch.as_tensor(np.asarray(alpha)[fit_mask], device=device, dtype=dtype)
        beta_vec = torch.as_tensor(np.asarray(beta)[fit_mask], device=device, dtype=dtype)
        kappa_vec = torch.as_tensor(np.asarray(kappa)[fit_mask], device=device, dtype=dtype)
        atlas_v1 = torch.as_tensor(np.asarray(v1_mu)[fit_mask], device=device, dtype=dtype)
        self.register_buffer("alpha", torch.nan_to_num(alpha_vec, nan=eps).clamp_min(eps))
        self.register_buffer("beta", torch.nan_to_num(beta_vec, nan=eps).clamp_min(eps))
        self.register_buffer("kappa", torch.nan_to_num(kappa_vec, nan=0.0).clamp_min(0.0))
        self.register_buffer("atlas_v1", F.normalize(torch.nan_to_num(atlas_v1), dim=1, eps=eps))

        quadrature_t, quadrature_w = np.polynomial.legendre.leggauss(48)
        self.register_buffer(
            "watson_t", torch.as_tensor((quadrature_t + 1.0) / 2.0, device=device, dtype=dtype)
        )
        self.register_buffer(
            "watson_log_weights",
            torch.log(torch.as_tensor(quadrature_w / 2.0, device=device, dtype=dtype)),
        )

        spatial_affine = np.asarray(affine, dtype=np.float64)[:3, :3]
        voxel_axis_mm = np.linalg.norm(spatial_affine, axis=0)
        bx, cpx = cubic_bspline_basis_matrix(
            self.spatial_shape[0],
            float(voxel_axis_mm[0]),
            self.bias_control_point_spacing_mm,
            device=device,
            dtype=dtype,
        )
        by, cpy = cubic_bspline_basis_matrix(
            self.spatial_shape[1],
            float(voxel_axis_mm[1]),
            self.bias_control_point_spacing_mm,
            device=device,
            dtype=dtype,
        )
        bz, cpz = cubic_bspline_basis_matrix(
            self.spatial_shape[2],
            float(voxel_axis_mm[2]),
            self.bias_control_point_spacing_mm,
            device=device,
            dtype=dtype,
        )
        self.register_buffer("bias_bspline_basis_x", bx)
        self.register_buffer("bias_bspline_basis_y", by)
        self.register_buffer("bias_bspline_basis_z", bz)
        self.bias_control_grid_shape = (cpx, cpy, cpz)
        n_bias_parameters = int(cpx * cpy * cpz)
        self.bias_coefficients = torch.nn.Parameter(
            torch.zeros((self.n_directions, n_bias_parameters, 1), device=device, dtype=dtype)
        )

        positive_s0 = self.source_b0.mean(dim=0)[self.fit_mask]
        positive_s0 = positive_s0[positive_s0 > 0]
        if positive_s0.numel() == 0:
            raise ValueError("No positive b0 samples exist inside the fit mask")
        signal_floor = (torch.median(positive_s0) * 1e-6).clamp_min(eps)
        self.register_buffer("signal_floor", signal_floor.detach())
        factor = 1000.0 / positive_s0.max()
        self.register_buffer("log_signal_factor", factor.detach())

        with torch.no_grad():
            initial_s0 = self.source_b0.mean(dim=0)
            initial_y = self.diffusion_observations(
                self.source_dwi, initial_s0, apply_bias=False
            )
            attenuation_scale = torch.median(torch.abs(initial_y)).clamp_min(1e-6)
        self.register_buffer("attenuation_scale", attenuation_scale.detach())

        with torch.no_grad():
            _, _, initial_fa, initial_v1, _ = self.tensor_quantities(
                self.source_dwi, initial_s0, apply_bias=False
            )
            initial_fa_safe = initial_fa.clamp(1e-6, 1.0 - 1e-6)
            initial_ll_fa = self.fa_log_likelihood(initial_fa_safe)
            initial_dot = (initial_v1 * self.atlas_v1).sum(dim=1).clamp(-1.0, 1.0)
            initial_ll_v1 = self.v1_log_likelihood(initial_fa_safe, initial_dot)
            fa_scale = (-initial_ll_fa.mean()).abs().clamp_min(1e-6)
            v1_scale = (-initial_ll_v1.mean()).abs().clamp_min(1e-6)
        self.register_buffer("fa_prior_scale", fa_scale.detach())
        self.register_buffer("v1_prior_scale", v1_scale.detach())

    @property
    def raw_coefficients(self):
        return self.ffd.raw_coefficients

    def dense_fields(self):

        dense = self.ffd()
        if self.shared_b0_deformation:
            return dense, dense[:1], dense[1:]
        if self.deform_b0:
            return dense, dense[:self.n_b0-1], dense[self.n_b0-1:]
        return dense, dense[:0], dense

    def joint_warp(self,):

        dense, b0_dense, dwi_dense = self.dense_fields()
        if self.shared_b0_deformation:
            shared_dense = b0_dense.expand(self.n_b0, -1, -1, -1, -1)
            warped_all_b0, b0_valid = pull_resample_dwi(
                self.source_b0, shared_dense, self.ffd.inverse_spatial_affine
            )
            warped_moving_b0 = warped_all_b0[1:]
            current_s0 = warped_all_b0.mean(dim=0)
        elif self.deform_b0 and self.n_b0>1:
            warped_moving_b0, b0_valid = pull_resample_dwi(
                self.source_b0[1:], b0_dense, self.ffd.inverse_spatial_affine
            )
            current_s0 = torch.cat((self.source_b0[:1], warped_moving_b0), dim=0).mean(dim=0)
        else:
            warped_moving_b0 = self.source_b0[1:]
            b0_valid = torch.ones_like(self.source_b0, dtype=torch.bool)
            current_s0 = self.source_b0.mean(dim=0)
        warped_dwi, dwi_valid = pull_resample_dwi(
            self.source_dwi, dwi_dense, self.ffd.inverse_spatial_affine
        )
        return warped_dwi, current_s0, warped_moving_b0, dense, dwi_valid, b0_valid

    def bias_field_masked(self):

        return self._evaluate_bspline_bias(
            self.projected_bias_coefficients()
        )[:, self.fit_mask]

    def _evaluate_bspline_bias(self, coefficients: torch.Tensor):

        if self.bias_basis != "bspline" or self.bias_control_grid_shape is None:
            raise RuntimeError("Invalid scalar B-spline bias lattice")
        lattice = coefficients.reshape(self.n_directions, *self.bias_control_grid_shape)
        along_x = torch.einsum(
            "dijk,xi->dxjk", lattice, self.bias_bspline_basis_x
        )
        along_xy = torch.einsum(
            "dxjk,yj->dxyk", along_x, self.bias_bspline_basis_y
        )
        return torch.einsum(
            "dxyk,zk->dxyz", along_xy, self.bias_bspline_basis_z
        )

    def bias_field_full(self, *, projected: bool = True):

        coefficients = (
            self.projected_bias_coefficients()
            if projected
            else self.bias_coefficients[..., 0]
        )
        return self._evaluate_bspline_bias(coefficients)

    def projected_bias_coefficients(self):

        raw = self.bias_coefficients[..., 0]
        return raw - raw.mean(dim=0, keepdim=True)

    def bias_amplitude_regularization(self):

        return self.bias_field_masked().square().mean()

    def corrected_log_dwi_masked(
        self,
        diffusion_images: torch.Tensor,
        b0_reference: torch.Tensor,
        *,
        apply_bias: bool,
    ):

        s0 = b0_reference[self.fit_mask].clamp_min(self.signal_floor)
        sd = diffusion_images[:, self.fit_mask].clamp_min(self.signal_floor)
        log_s0 = torch.log1p(self.log_signal_factor * s0)
        log_sd = torch.log1p(self.log_signal_factor * sd)
        if apply_bias:
            log_sd = log_sd - self.bias_field_masked()
        if self.force_sn_lower_than_s0:
            log_sd = torch.minimum(log_sd, log_s0[None, :] - 1e-4)
        return log_s0, log_sd

    def diffusion_observations(
        self,
        diffusion_images: torch.Tensor,
        b0_reference: torch.Tensor,
        *,
        apply_bias: bool,
    ):
        log_s0, log_sd = self.corrected_log_dwi_masked(
            diffusion_images, b0_reference, apply_bias=apply_bias
        )
        return (log_s0[None, :] - log_sd) / self.bval

    def tensor_quantities(
        self,
        diffusion_images: torch.Tensor,
        b0_reference: torch.Tensor,
        *,
        apply_bias: bool,
    ):

        y = self.diffusion_observations(
            diffusion_images, b0_reference, apply_bias=apply_bias
        )
        coefficients = self.design_pinv @ y
        tensor = torch.stack(
            (
                torch.stack((coefficients[0], coefficients[3], coefficients[4]), dim=1),
                torch.stack((coefficients[3], coefficients[1], coefficients[5]), dim=1),
                torch.stack((coefficients[4], coefficients[5], coefficients[2]), dim=1),
            ),
            dim=1,
        )
        jitter = self.attenuation_scale * 1e-4 * torch.arange(
            3, device=tensor.device, dtype=tensor.dtype
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(
            tensor + torch.diag(jitter)[None, ...]
        )
        eigenvalues = torch.nan_to_num(
            eigenvalues, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        v1 = eigenvectors[:, :, 2]
        mean_eigenvalue = eigenvalues.mean(dim=1, keepdim=True)
        numerator = torch.sqrt(
            1.5 * (eigenvalues - mean_eigenvalue).square().sum(dim=1)
        )
        denominator = torch.sqrt(eigenvalues.square().sum(dim=1)).clamp_min(
            torch.finfo(tensor.dtype).eps
        )
        fa = (numerator / denominator).clamp(0.0, 1.0)
        return y, coefficients, fa, v1, eigenvalues

    def fa_log_likelihood(self, fa: torch.Tensor):
        return (self.alpha - 1.0) * torch.log(fa) + (self.beta - 1.0) * torch.log(1.0 - fa)

    def v1_log_likelihood(self, fa: torch.Tensor, v1_dot: torch.Tensor):

        concentration = self.kappa * fa
        log_z = torch.logsumexp(
            concentration[:, None] * self.watson_t.square()
            + self.watson_log_weights,
            dim=1,
        )
        return concentration * v1_dot.square() - log_z


def bending_density(u, inverse):

    mid = u[:, :, 1:-1, 1:-1, 1:-1]
    xx = u[:, :, 2:, 1:-1, 1:-1]-2*mid+u[:, :, :-2, 1:-1, 1:-1]
    yy = u[:, :, 1:-1, 2:, 1:-1]-2*mid+u[:, :, 1:-1, :-2, 1:-1]
    zz = u[:, :, 1:-1, 1:-1, 2:]-2*mid+u[:, :, 1:-1, 1:-1, :-2]
    xy = (u[:, :, 2:, 2:, 1:-1]-u[:, :, 2:, :-2, 1:-1]-u[:, :, :-2, 2:, 1:-1]+u[:, :, :-2, :-2, 1:-1])/4
    xz = (u[:, :, 2:, 1:-1, 2:]-u[:, :, 2:, 1:-1, :-2]-u[:, :, :-2, 1:-1, 2:]+u[:, :, :-2, 1:-1, :-2])/4
    yz = (u[:, :, 1:-1, 2:, 2:]-u[:, :, 1:-1, 2:, :-2]-u[:, :, 1:-1, :-2, 2:]+u[:, :, 1:-1, :-2, :-2])/4
    h = torch.stack((torch.stack((xx,xy,xz),-1),torch.stack((xy,yy,yz),-1),torch.stack((xz,yz,zz),-1)),-2)
    h = torch.einsum('ai,dcxyzab,bj->dcxyzij', inverse, h, inverse)
    return h.square().sum((-1,-2)).mean((0,1))


def build(d, c):
    cache.clear()
    m = JointBiasGeometryDSWModel(
        d['seq'][...,d['low_idx']], d['seq'][...,d['dwi_idx']], d['fit'], d['bvec'], d['bval'],
        d['alpha'], d['beta'], d['v1'], d['kappa'], d['aff'],
        control_point_spacing_mm=c['grid'], max_displacement_mm=c['max_disp'],
        bias_basis='bspline', bias_control_point_spacing_mm=c['bias_grid'],
        bias_regularization_domain='field', deform_b0=c['shared_b0'],
        shared_b0_deformation=c['shared_b0'], device='cpu')
    m.wm_select = torch.as_tensor(np.isin(d['seg'], [2,7,41,46]))[m.fit_mask]
    m.register_buffer('bending_brain', torch.from_numpy(d['brain'][1:-1,1:-1,1:-1].copy()))
    if not m.wm_select.any() or not m.bending_brain.any():
        raise ValueError('WM tensor support or interior brain support is empty')
    return m


def parts(m):
    sd, s0, _, u, _, _ = m.joint_warp()
    y, _, fa, v1, _ = m.tensor_quantities(sd, s0, apply_bias=True)
    f = fa.clamp(1e-6, 1-1e-6)
    dot = (v1*m.atlas_v1).sum(-1).clamp(-1, 1)
    yy = y[:,m.wm_select]
    residual = (yy-m.design_matrix@(m.design_pinv@yy))/m.attenuation_scale
    tensor = F.smooth_l1_loss(residual.abs(), torch.zeros_like(residual), beta=1.)
    bd = bending_density(u, m.ffd.inverse_spatial_affine)
    ud = u[1:] if m.shared_b0_deformation else u
    return dict(tensor=tensor, fa=-m.fa_log_likelihood(f).mean(),
                v1=-m.v1_log_likelihood(f, dot).mean(), bias=m.bias_field_masked().square().mean(),
                disp=u[:,:,m.fit_mask].square().sum(1).mean(), bend=bd[m.bending_brain].mean(),
                common=ud.mean(0)[:,m.fit_mask].square().sum(0).mean())
