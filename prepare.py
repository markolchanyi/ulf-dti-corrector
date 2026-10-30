from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
from typing import Dict, Tuple
import nibabel as nib
import numpy as np
from scipy import ndimage

RAS_TO_LPS = np.diag([-1., -1., 1.])
GROUP_LABELS = {'cerebral_wm': (2,41), 'ventricles': (4,5,14,15,43,44),
                'thalami': (10,49), 'deep_gm': (10,11,12,13,17,18,26,28,49,50,51,52,53,54,58,60)}


def tensor_design(bvec):
    x,y,z = bvec
    return np.stack((x*x,y*y,z*z,2*x*y,2*x*z,2*y*z),axis=1)


def gradient_scheme(bvals_path, bvecs_path, nvol=None, threshold=50.):
    b = np.loadtxt(bvals_path).ravel()
    g = np.loadtxt(bvecs_path)
    if g.shape == (len(b),3) and g.shape != (3,len(b)):
        g = g.T
    if nvol is not None and len(b) != nvol:
        raise ValueError('b-values must have one entry per image volume')
    if g.shape != (3,len(b)):
        raise ValueError('b-vectors must have shape 3xN or Nx3 matching the b-values')
    if not np.isfinite(b).all() or not np.isfinite(g).all() or (b<0).any():
        raise ValueError('Gradients must be finite and b-values nonnegative')
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError('Low-B threshold must be finite and nonnegative')
    low, dw = np.flatnonzero(b<=threshold), np.flatnonzero(b>threshold)
    if not len(low):
        raise ValueError('At least one low-B frame is required; check --b0-threshold')
    if len(dw)<6:
        raise ValueError('At least six diffusion measurements are required for tensor fitting')
    norm = np.linalg.norm(g[:,dw],axis=0)
    if (norm<1e-8).any():
        raise ValueError('Zero diffusion gradient')
    h = tensor_design(g[:,dw]/norm)
    if np.linalg.matrix_rank(h)<6:
        raise ValueError('Tensor design is rank deficient: use six independent tensor measurements')
    if len(dw)==6:
        warnings.warn('Exactly six tensor measurements: tensor consistency has no residual degrees of freedom')
    if b[dw].max()>1500:
        warnings.warn('High-b data supplied the single-tensor approximation may be inaccurate')
    return b,g,low,dw

def pull_jacobian_lps(
    displacement_lps_mm: np.ndarray,
    fixed_affine_ras: np.ndarray,
):

    if displacement_lps_mm.ndim != 4 or displacement_lps_mm.shape[-1] != 3:
        raise ValueError("displacement_lps_mm must have shape [X,Y,Z,3]")
    if not np.isfinite(displacement_lps_mm).all():
        raise ValueError("Displacement field contains NaN or Inf")
    voxel_to_lps = RAS_TO_LPS @ np.asarray(fixed_affine_ras, dtype=np.float64)[:3, :3]
    if abs(np.linalg.det(voxel_to_lps)) < 1e-10:
        raise ValueError("Fixed spatial affine is singular")
    index_to_lps_inverse = np.linalg.inv(voxel_to_lps)

    du_di = np.empty(displacement_lps_mm.shape[:3] + (3, 3), dtype=np.float64)
    for component in range(3):
        gradients = np.gradient(displacement_lps_mm[..., component], edge_order=2)
        for index_axis, gradient in enumerate(gradients):
            du_di[..., component, index_axis] = gradient
    du_dx_lps = np.einsum("...ca,ab->...cb", du_di, index_to_lps_inverse)
    return du_dx_lps + np.eye(3, dtype=np.float64)

def closest_proper_rotation(forward_jacobian_lps: np.ndarray):

    u, _, vh = np.linalg.svd(forward_jacobian_lps)
    rotation = u @ vh
    reflected = np.linalg.det(rotation) < 0
    if np.any(reflected):
        u = u.copy()
        u[reflected, :, -1] *= -1.0
        rotation = u @ vh
    return rotation

def finite_strain_reorient(
    resampled_v1_ras: np.ndarray,
    pull_displacement_lps_mm: np.ndarray,
    fixed_affine_ras: np.ndarray,
    mask: np.ndarray,
):

    resampled_v1_ras = np.asarray(resampled_v1_ras, dtype=np.float64)
    pull_displacement_lps_mm = np.asarray(pull_displacement_lps_mm, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    expected_shape = pull_displacement_lps_mm.shape[:3]
    if resampled_v1_ras.shape != (*expected_shape, 3):
        raise ValueError("V1 and displacement spatial shapes do not match")
    if mask.shape != expected_shape:
        raise ValueError("Mask and displacement spatial shapes do not match")

    pull_jacobian = pull_jacobian_lps(pull_displacement_lps_mm, fixed_affine_ras)
    pull_determinant = np.linalg.det(pull_jacobian)
    invertible = np.isfinite(pull_determinant) & (np.abs(pull_determinant) > 1e-8)
    if not np.all(invertible[mask]):
        count = int(np.sum(mask & ~invertible))
        raise RuntimeError(f"Pull Jacobian is singular/non-finite at {count} masked voxels")

    forward_jacobian = np.broadcast_to(
        np.eye(3, dtype=np.float64), pull_jacobian.shape
    ).copy()
    forward_jacobian[invertible] = np.linalg.inv(pull_jacobian[invertible])
    forward_determinant = np.linalg.det(forward_jacobian)
    rotation_lps = closest_proper_rotation(forward_jacobian)

    input_norm = np.linalg.norm(resampled_v1_ras, axis=-1)
    valid_vector = np.isfinite(input_norm) & (input_norm > 1e-8)
    input_unit_ras = np.zeros_like(resampled_v1_ras)
    input_unit_ras[valid_vector] = (
        resampled_v1_ras[valid_vector] / input_norm[valid_vector, None]
    )
    input_unit_lps = np.einsum("ij,...j->...i", RAS_TO_LPS, input_unit_ras)
    output_unit_lps = np.einsum("...ij,...j->...i", rotation_lps, input_unit_lps)
    output_unit_ras = np.einsum("ij,...j->...i", RAS_TO_LPS, output_unit_lps)
    output_norm = np.linalg.norm(output_unit_ras, axis=-1)
    keep = mask & valid_vector & np.isfinite(output_norm) & (output_norm > 1e-8)
    output = np.zeros_like(output_unit_ras)
    output[keep] = output_unit_ras[keep] / output_norm[keep, None]

    trace = np.trace(rotation_lps, axis1=-2, axis2=-1)
    angle_deg = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    angle_deg[~mask] = 0.0
    fields = {
        "pull_jacobian_lps": pull_jacobian,
        "forward_jacobian_lps": forward_jacobian,
        "rotation_lps": rotation_lps,
        "pull_determinant": pull_determinant,
        "forward_determinant": forward_determinant,
        "rotation_angle_deg": angle_deg,
        "normalized_nonreoriented_v1_ras": input_unit_ras * mask[..., None],
        "valid_vector": keep,
    }
    return output, fields

def signed_distance(mask: np.ndarray, spacing_mm, clip_mm: float):

    inside = ndimage.distance_transform_edt(mask, sampling=spacing_mm)
    outside = ndimage.distance_transform_edt(~mask, sampling=spacing_mm)
    return np.clip((inside - outside) / clip_mm, -1.0, 1.0).astype(np.float32)

def make_cc_proxy(seg: np.ndarray, affine: np.ndarray):

    indices = np.moveaxis(np.indices(seg.shape, dtype=np.float64), 0, -1)
    world_ras = nib.affines.apply_affine(affine, indices)
    wm = np.isin(seg, GROUP_LABELS["cerebral_wm"])
    left_thalamus, right_thalamus = seg == 10, seg == 49
    thalami = left_thalamus | right_thalamus
    ventricles = np.isin(seg, GROUP_LABELS["ventricles"])
    if left_thalamus.any() and right_thalamus.any():
        midline_r = 0.5 * (
            float(world_ras[..., 0][left_thalamus].mean())
            + float(world_ras[..., 0][right_thalamus].mean())
        )
    else:
        midline_r = float(np.median(world_ras[..., 0][wm]))
    inferior_limit_s = float(np.median(world_ras[..., 2][thalami]))
    if ventricles.any():
        anterior_limit_a, posterior_limit_a = np.percentile(
            world_ras[..., 1][ventricles], [2.0, 98.0]
        )
    else:
        anterior_limit_a, posterior_limit_a = np.percentile(
            world_ras[..., 1][wm], [15.0, 85.0]
        )
    proxy = (
        wm
        & (np.abs(world_ras[..., 0] - midline_r) <= 9.0)
        & (world_ras[..., 2] >= inferior_limit_s)
        & (world_ras[..., 1] >= anterior_limit_a - 18.0)
        & (world_ras[..., 1] <= posterior_limit_a + 18.0)
    )
    metadata = {
        "definition": "broad midsagittal cerebral-WM bridge; not a hard CC segmentation",
        "midline_r_mm": midline_r,
        "inferior_limit_s_mm": inferior_limit_s,
        "anterior_limit_a_mm": float(anterior_limit_a - 18.0),
        "posterior_limit_a_mm": float(posterior_limit_a + 18.0),
        "midline_half_width_mm": 9.0,
        "voxels": int(proxy.sum()),
    }
    return proxy, metadata

def rotation_from_itk_text(txt_path: Path):

    text = txt_path.read_text()

    m = re.search(r'Parameters:\s*([eE0-9.\+\-\s]+)', text)
    if m:
        nums = np.fromstring(m.group(1), sep=' ')
        if nums.size >= 9:
            return nums[:9].reshape(3, 3)

    m = re.search(r'Matrix\s*=\s*([eE0-9.\+\-\s]+)', text)
    if m:
        nums = np.fromstring(m.group(1), sep=' ')
        if nums.size >= 9:
            return nums[:9].reshape(3, 3)

    raise RuntimeError(f'Could not find a 3×3 rotation block in {txt_path}')

def rotate_and_normalise(R: np.ndarray, v: np.ndarray):

    v2 = R @ v
    n = np.linalg.norm(v2)
    return v2 if n == 0 else v2 / n


def save(a, ref, path):
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(np.asarray(a,dtype=np.float32),ref.affine,header),str(path))


def same_grid(a, b):
    if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine,b.affine,atol=1e-5):
        raise ValueError('Images do not share a grid')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command',required=True)
    check = sub.add_parser('check')
    for key in ['dwi','bvals','bvecs']:
        check.add_argument('--'+key,required=True)
    check.add_argument('--b0-threshold',type=float,default=50.)
    check.add_argument('--layout',action='store_true')
    rot = sub.add_parser('rotate')
    for key in ['bvals','bvecs','affines','out']:
        rot.add_argument('--'+key,required=True)
    rot.add_argument('--b0-threshold',type=float,default=50.)
    masks = sub.add_parser('mask')
    for key in ['input','mask','out']:
        masks.add_argument('--'+key,required=True)
    chan = sub.add_parser('channels')
    chan.add_argument('--seg',required=True)
    chan.add_argument('--out',required=True)
    vec = sub.add_parser('vectors')
    for key in ['v1','pull','mask','out']:
        vec.add_argument('--'+key,required=True)
    qc = sub.add_parser('qc')
    for key in ['fixed','moving','mask','atlas-mask','out']:
        qc.add_argument('--'+key,required=True)
    qc.add_argument('--min-dice',type=float,default=.85)
    pose = sub.add_parser('pose')
    for key in ['mask','atlas-mask','out']:
        pose.add_argument('--'+key,required=True)
    overlap = sub.add_parser('overlap')
    overlap.add_argument('--mask',required=True)
    overlap.add_argument('--atlas-mask',required=True)
    overlap.add_argument('--min-dice',type=float,default=.85)
    a = p.parse_args()
    if a.command == 'pose':
        from scipy.io import savemat
        out = Path(a.out)
        out.mkdir(exist_ok=True)
        centers, spreads = [], []
        # Initialize the fixed-to-moving affine from physical brain-mask moments.
        for path,name in [(a.mask,'subject'),(a.atlas_mask,'atlas')]:
            im = nib.load(path)
            mask = im.get_fdata()>.5
            if not mask.any(): raise ValueError('Empty pose mask: '+path)
            xyz = nib.affines.apply_affine(im.affine,np.argwhere(mask))*[-1,-1,1]
            centers.append(xyz.mean(0)); spreads.append(xyz.std(0))
            spacing = np.linalg.norm(im.affine[:3,:3],axis=0)
            sdf = ndimage.distance_transform_edt(mask,sampling=spacing)-ndimage.distance_transform_edt(~mask,sampling=spacing)
            save(np.clip(sdf,-30,30)/30,im,out/f'{name}_distance.nii.gz')
        matrix = np.diag(spreads[1]/spreads[0])
        savemat(out/'mask_pose_initial.mat',{'AffineTransform_double_3_3':np.r_[matrix.ravel(),centers[1]-centers[0]][:,None],
                'fixed':centers[0][:,None]},format='4')
    elif a.command == 'overlap':
        im,am = nib.load(a.mask),nib.load(a.atlas_mask)
        same_grid(im,am)
        m,n = im.get_fdata()>.5,am.get_fdata()>.5
        dice = float(2*(m&n).sum()/max(m.sum()+n.sum(),1))
        print('Atlas mask Dice:',dice,flush=True)
        if dice<a.min_dice: raise RuntimeError('Poor atlas overlap; stopping before further processing')
    elif a.command == 'check':
        im = nib.load(a.dwi)
        if im.ndim != 4:
            raise ValueError('Expected a 4D diffusion sequence')
        bv,g,low,dw = gradient_scheme(a.bvals,a.bvecs,im.shape[-1],a.b0_threshold)
        if a.layout:
            print(len(bv),int(low[0]),','.join(map(str,low)))
    elif a.command == 'rotate':
        _,g,_,dw = gradient_scheme(a.bvals,a.bvecs,threshold=a.b0_threshold)
        # Preserve the historical full-affine gradient update for reproducibility.
        for i in dw:
            R = rotation_from_itk_text(Path(a.affines.format(i=i)))
            g[:,i] = rotate_and_normalise(R,g[:,i])
        np.savetxt(a.out,g,fmt='%.6f')
    elif a.command == 'mask':
        im, mask = nib.load(a.input), nib.load(a.mask)
        same_grid(im,mask)
        x = np.maximum(im.get_fdata(),0)*(mask.get_fdata()>0)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite registration image')
        save(x,im,a.out)
    elif a.command == 'channels':
        im = nib.load(a.seg)
        seg = np.rint(im.get_fdata()).astype(int)
        out = Path(a.out)
        out.mkdir(parents=True,exist_ok=False)
        masks = {k:np.isin(seg,v) for k,v in GROUP_LABELS.items()}
        masks['cc_proxy'], _ = make_cc_proxy(seg,im.affine)
        for key,mask in masks.items():
            if not mask.any():
                raise ValueError(f'Empty registration channel: {key}')
            save(mask,im,out/f'{key}.mask.nii.gz')
            save(signed_distance(mask,im.header.get_zooms()[:3],18.),im,out/f'{key}.distance.nii.gz')
    elif a.command == 'vectors':
        im, pull, mask = nib.load(a.v1), nib.load(a.pull), nib.load(a.mask)
        same_grid(im,pull)
        same_grid(im,mask)
        v, qc = finite_strain_reorient(im.get_fdata(),np.squeeze(pull.get_fdata()),
                                      im.affine,mask.get_fdata()>0)
        save(v,im,a.out)
        save(qc['pull_determinant'],im,str(Path(a.out).parent/'atlas_pull_jacobian.nii.gz'))
    elif a.command == 'qc':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        im = nib.load(a.fixed)
        moving, mask, amask = [nib.load(x) for x in [a.moving,a.mask,a.atlas_mask]]
        for x in [moving,mask,amask]:
            same_grid(im,x)
        m, n = mask.get_fdata()>0, amask.get_fdata()>0
        dice = 2*(m&n).sum()/max(m.sum()+n.sum(),1)
        out = Path(a.out)
        fig,axes = plt.subplots(3,3,figsize=(10,10),facecolor='black')
        x,y = im.get_fdata(),moving.get_fdata()
        for axis in range(3):
            idx = int(np.median(np.argwhere(m)[:,axis]))
            sl = lambda z: np.rot90(np.take(z,idx,axis))
            for col in range(3):
                ax = axes[axis,col]
                base = y if col==1 else x
                support = n if col==1 else m
                ax.imshow(sl(base),cmap='gray',vmin=0,vmax=max(np.percentile(base[support],99),1e-8))
                if col==2:
                    ax.imshow(np.ma.masked_where(~sl(n),sl(y)),cmap='hot',alpha=.4,
                              vmin=0,vmax=max(np.percentile(y[n],99),1e-8))
                ax.set_title(['Subject','Registered atlas','Overlay'][col],color='white')
                ax.axis('off')
        fig.tight_layout()
        fig.savefig(out/'registration.png',dpi=150,facecolor='black')
        plt.close(fig)
        (out/'registration_qc.json').write_text(json.dumps(dict(brain_dice=float(dice),
             note='Inspect registration.png before interpreting correction'),indent=2))
        if dice<a.min_dice:
            raise RuntimeError('Poor atlas registration; inspect registration.png before correction')


if __name__ == '__main__':
    main()
