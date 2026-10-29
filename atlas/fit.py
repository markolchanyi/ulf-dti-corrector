import argparse, numpy as np
from pathlib import Path
import nibabel as nib

def clamp01(x):
    x = np.asarray(x, dtype=np.float32)
    x[x < 0] = 0.0
    x[x > 1] = 1.0
    return x

def precompute_logZ(xxmax=200.0, step=0.01):

    xx = np.arange(0.0, xxmax + 1e-12, step, dtype=np.float64)
    nmax = 10000
    n = np.arange(nmax + 1, dtype=np.float64)

    from math import lgamma
    log_fact = np.fromiter((lgamma(int(k) + 1) for k in n), dtype=np.float64)
    logNph = np.log(n + 0.5)

    yy = np.empty_like(xx)
    for i, kappa in enumerate(xx):
        if kappa == 0.0:
            yy[i] = np.log(0.5)
            continue
        logterms = n * np.log(kappa) - logNph - log_fact
        m = np.max(logterms)
        yy[i] = np.log(0.5) + m + np.log(np.sum(np.exp(logterms - m)))
    return xx, yy

class LogZInterp:

    def __init__(self, xx, yy):
        self.xx, self.yy = xx, yy
    def __call__(self, k):
        k = np.clip(k, self.xx[0], self.xx[-1])
        idx = np.minimum(np.searchsorted(self.xx, k), len(self.xx) - 1)
        idx0 = np.maximum(idx - 1, 0)
        x0, x1 = self.xx[idx0], self.xx[idx]
        y0, y1 = self.yy[idx0], self.yy[idx]
        t = np.divide(k - x0, (x1 - x0), out=np.zeros_like(k, dtype=np.float64), where=(x1 != x0))
        return (1 - t) * y0 + t * y1

def beta_params_moments(mean, var):

    eps = 1e-8
    var = np.maximum(var, eps)
    mean = np.clip(mean, 1e-6, 1 - 1e-6)
    t = mean * (1 - mean) / var - 1.0
    t = np.maximum(t, 0.0)
    alpha = np.maximum(mean * t, 1e-3)
    beta  = np.maximum((1 - mean) * t, 1e-3)
    return alpha.astype(np.float32), beta.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description="FAST beta-DSW atlas (alpha, beta, kappa, v1_mu) at 3 mm.")
    ap.add_argument("--datadir", required=True, help="Folder with *.fa.nii.gz and matching *.v1.nii.gz (3 mm, coregistered)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--ref", required=True, help="3 mm reference NIfTI (shape/affine)")
    ap.add_argument("--glob-pattern", default="*.fa.nii.gz")
    ap.add_argument("--Nmin", type=int, default=10, help="Minimum #subjects per voxel (overrides if --mask-frac provided)")
    ap.add_argument("--mask-frac", type=float, default=None, help="Use ceil(frac * #subjects) instead of Nmin")
    ap.add_argument("--kappa-max", type=float, default=200.0)
    ap.add_argument("--kappa-step", type=float, default=0.01)
    ap.add_argument("--approx-kappa", action="store_true", help="Use κ ≈ (3λ1−1)/(1−λ1) (very fast, monotone)")
    ap.add_argument("--debug", action="store_true", help="Print extra diagnostics")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(args.ref); aff, hdr = ref_img.affine, ref_img.header
    siz = ref_img.shape[:3]

    fa_paths = sorted(Path(args.datadir).glob(args.glob_pattern))
    if not fa_paths:
        raise RuntimeError("No FA files found")
    nS = len(fa_paths)
    eff_Nmin = args.Nmin if args.mask_frac is None else max(1, int(np.ceil(args.mask_frac * nS)))
    print(f"subjects={nS}  effective Nmin={eff_Nmin}")

    sum_fa   = np.zeros(siz, dtype=np.float64)
    sum_fa2  = np.zeros(siz, dtype=np.float64)
    count    = np.zeros(siz, dtype=np.float64)

    s_xx = np.zeros(siz, dtype=np.float64); s_yy = np.zeros(siz, dtype=np.float64); s_zz = np.zeros(siz, dtype=np.float64)
    s_xy = np.zeros(siz, dtype=np.float64); s_xz = np.zeros(siz, dtype=np.float64); s_yz = np.zeros(siz, dtype=np.float64)
    w_sum = np.zeros(siz, dtype=np.float64)

    for i, fa_p in enumerate(fa_paths, 1):
        v1_p = fa_p.with_name(fa_p.name.replace(".fa.nii.gz", ".v1.nii.gz"))
        print(f"  {i}/{nS}: {fa_p.name}")

        fa = clamp01(nib.load(str(fa_p)).get_fdata(dtype=np.float32))
        v1 = nib.load(str(v1_p)).get_fdata(dtype=np.float32)
        if v1.ndim != 4 or v1.shape[-1] != 3 or v1.shape[:3] != fa.shape[:3]:
            raise RuntimeError(f"Shape mismatch for {fa_p.name}")

        fa = np.nan_to_num(fa, nan=0.0, posinf=0.0, neginf=0.0)
        v1 = np.nan_to_num(v1, nan=0.0, posinf=0.0, neginf=0.0)

        M = (fa > 0).astype(np.float64)
        sum_fa  += fa
        sum_fa2 += fa * fa
        count   += M

        vx, vy, vz = v1[..., 0], v1[..., 1], v1[..., 2]

        s_xx += fa * vx * vx; s_yy += fa * vy * vy; s_zz += fa * vz * vz
        s_xy += fa * vx * vy; s_xz += fa * vx * vz; s_yz += fa * vy * vz
        w_sum += fa

    mask = count >= eff_Nmin
    C_safe = np.where(count == 0, 1.0, count)
    mean_fa = (sum_fa / C_safe)
    var_fa  = (sum_fa2 / C_safe) - mean_fa**2
    alpha, beta = beta_params_moments(mean_fa, var_fa)

    W_safe = np.where(w_sum == 0, 1.0, w_sum)
    s_xx /= W_safe; s_yy /= W_safe; s_zz /= W_safe
    s_xy /= W_safe; s_xz /= W_safe; s_yz /= W_safe

    for arr in (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz):
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    S = np.stack([
        np.stack([s_xx, s_xy, s_xz], axis=-1),
        np.stack([s_xy, s_yy, s_yz], axis=-1),
        np.stack([s_xz, s_yz, s_zz], axis=-1)
    ], axis=-2).astype(np.float64)

    eps = 1e-6
    S = 0.5 * (S + np.swapaxes(S, -1, -2))
    S[..., 0, 0] += eps; S[..., 1, 1] += eps; S[..., 2, 2] += eps

    finite_S = np.isfinite(S).all(axis=(-1, -2))
    good = mask & finite_S

    idx = np.where(good)
    if len(idx[0]) == 0:
        zero = np.zeros(siz, dtype=np.float32)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff, hdr.copy()), str(outdir / "mask.nii.gz"))
        nib.save(nib.Nifti1Image(zero, aff, hdr.copy()), str(outdir / "alpha.nii.gz"))
        nib.save(nib.Nifti1Image(zero, aff, hdr.copy()), str(outdir / "beta.nii.gz"))
        nib.save(nib.Nifti1Image(zero, aff, hdr.copy()), str(outdir / "kappa.nii.gz"))
        nib.save(nib.Nifti1Image(np.zeros(siz + (3,), np.float32), aff, hdr.copy()), str(outdir / "v1_mu.nii.gz"))
        nib.save(nib.Nifti1Image(zero, aff, hdr.copy()), str(outdir / "FAav.nii.gz"))
        nib.save(nib.Nifti1Image(zero, aff, hdr.copy()), str(outdir / "FAstd.nii.gz"))
        nib.save(nib.Nifti1Image(np.zeros(siz + (3,), np.float32), aff, hdr.copy()), str(outdir / "RGBav.nii.gz"))
        nib.save(nib.Nifti1Image(count.astype(np.float32), aff, hdr.copy()), str(outdir / "count.nii.gz"))
        nib.save(nib.Nifti1Image((mask & ~finite_S).astype(np.uint8), aff, hdr.copy()), str(outdir / "bad_scatter_mask.nii.gz"))
        print("No voxels passed Nmin; wrote zeros.")
        return

    S_mask = S[idx]
    try:
        vals, vecs = np.linalg.eigh(S_mask)
    except np.linalg.LinAlgError:

        vals_list, vecs_list = [], []
        batch = 50000
        for start in range(0, S_mask.shape[0], batch):
            Sb = S_mask[start:start + batch]
            try:
                vb, qb = np.linalg.eigh(Sb)
            except np.linalg.LinAlgError:
                vb = np.zeros((Sb.shape[0], 3), dtype=np.float64)
                qb = np.zeros((Sb.shape[0], 3, 3), dtype=np.float64)
                for t in range(Sb.shape[0]):
                    try:
                        vvt, qqt = np.linalg.eigh(Sb[t])
                    except np.linalg.LinAlgError:
                        vvt = np.array([1/3, 1/3, 1/3], dtype=np.float64)
                        qqt = np.eye(3, dtype=np.float64)
                    vb[t], qb[t] = vvt, qqt
            vals_list.append(vb); vecs_list.append(qb)
        vals = np.concatenate(vals_list, axis=0)
        vecs = np.concatenate(vecs_list, axis=0)

    lam1 = np.clip(vals[:, 2], 1.0/3.0, 1.0 - 1e-8)
    mu   = vecs[:, :, 2]
    nrm = np.linalg.norm(mu, axis=1, keepdims=True)
    mu = np.divide(mu, nrm, out=np.zeros_like(mu), where=(nrm > 0))

    if args.approx_kappa:
        num = 3.0 * lam1 - 1.0
        den = np.maximum(1e-6, 1.0 - lam1)
        kappa_vec = np.maximum(0.0, num / den)
    else:
        xx, yy = precompute_logZ(args.kappa_max, args.kappa_step)
        logZk = yy
        kgrid = xx
        ws = np.where(w_sum == 0, 0.0, w_sum)[idx]

        J = (ws[:, None] * (lam1[:, None] * kgrid[None, :] - logZk[None, :])).astype(np.float64)
        km_idx = np.argmax(J, axis=1)
        kappa_vec = kgrid[km_idx]

    ALPHA = np.zeros(siz, dtype=np.float32);  ALPHA[idx] = alpha[idx].astype(np.float32)
    BETA  = np.zeros(siz, dtype=np.float32);  BETA[idx]  = beta[idx].astype(np.float32)
    KAPPA = np.zeros(siz, dtype=np.float32);  KAPPA[idx] = kappa_vec.astype(np.float32)

    V1MU = np.zeros(siz + (3,), dtype=np.float32)
    V1MU[idx] = mu.astype(np.float32)

    FAav = np.divide(ALPHA, np.maximum(ALPHA + BETA, 1e-6)).astype(np.float32)
    FAvar = (ALPHA.astype(np.float64) * BETA.astype(np.float64)) /\
            (np.maximum((ALPHA + BETA).astype(np.float64), 1e-6) ** 2 *\
             np.maximum((ALPHA + BETA + 1).astype(np.float64), 1e-6))
    FAstd = np.sqrt(np.maximum(FAvar, 0.0)).astype(np.float32)
    RGBav = (np.abs(V1MU) * FAav[..., None] * 255.0).astype(np.float32)

    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff, hdr.copy()), str(outdir / "mask.nii.gz"))
    nib.save(nib.Nifti1Image(ALPHA, aff, hdr.copy()), str(outdir / "alpha.nii.gz"))
    nib.save(nib.Nifti1Image(BETA,  aff, hdr.copy()), str(outdir / "beta.nii.gz"))
    nib.save(nib.Nifti1Image(KAPPA, aff, hdr.copy()), str(outdir / "kappa.nii.gz"))
    nib.save(nib.Nifti1Image(V1MU,  aff, hdr.copy()), str(outdir / "v1_mu.nii.gz"))
    nib.save(nib.Nifti1Image(FAav,  aff, hdr.copy()), str(outdir / "FAav.nii.gz"))
    nib.save(nib.Nifti1Image(FAstd, aff, hdr.copy()), str(outdir / "FAstd.nii.gz"))
    nib.save(nib.Nifti1Image(RGBav, aff, hdr.copy()), str(outdir / "RGBav.nii.gz"))

    COUNT = count.astype(np.float32)
    BAD = (mask & ~finite_S).astype(np.uint8)
    nib.save(nib.Nifti1Image(COUNT, aff, hdr.copy()), str(outdir / "count.nii.gz"))
    nib.save(nib.Nifti1Image(BAD,   aff, hdr.copy()), str(outdir / "bad_scatter_mask.nii.gz"))

    if args.debug:
        covered = (COUNT >= eff_Nmin).mean()
        print(f"mask coverage: {covered*100:.1f}% voxels pass (>= {eff_Nmin})")

    print("Done:", outdir)

if __name__ == "__main__":
    main()
