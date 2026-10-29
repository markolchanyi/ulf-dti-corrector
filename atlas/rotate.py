import argparse, numpy as np, nibabel as nib

def polar_rotation(F: np.ndarray):
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

def main():
    ap = argparse.ArgumentParser(description="Rotate a V1 field by the polar rotation of inv(aff).")
    ap.add_argument("--aff", required=True, help=".aff file from reg_aladin")
    ap.add_argument("--in-v1", required=True, help="resampled V1 (nearest) in target space")
    ap.add_argument("--out-v1", required=True, help="output rotated V1")
    ap.add_argument("--flipx", action="store_true", help="flip x-component before rotation (match MATLAB)")
    args = ap.parse_args()

    A = np.loadtxt(args.aff)
    T = np.linalg.inv(A)[:3, :3]
    Q = polar_rotation(T)

    img = nib.load(args.in_v1)
    v1 = img.get_fdata(dtype=np.float32)
    if v1.ndim != 4 or v1.shape[-1] != 3:
        raise SystemExit(f"V1 must be 4D with last dim=3, got {v1.shape}")

    if args.flipx:
        v1[..., 0] *= -1.0

    v1f = v1.reshape(-1, 3)
    v1r = (Q @ v1f.T).T.reshape(v1.shape)
    nib.save(nib.Nifti1Image(v1r, img.affine, img.header), args.out_v1)
    print(f"Wrote {args.out_v1}")

if __name__ == "__main__":
    main()
