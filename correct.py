import argparse, csv, hashlib, json, time
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
import model
from prepare import gradient_scheme, tensor_design


def save(a, path, ref):
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    header.set_intent('none')
    nib.save(nib.Nifti1Image(np.asarray(a, dtype=np.float32), ref.affine, header), str(path))


def table(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(args):
    ref = nib.load(args.dwi)
    if ref.ndim != 4:
        raise ValueError('Expected a 4D diffusion sequence')
    def read(path):
        im = nib.load(str(path))
        if im.shape[:3] != ref.shape[:3] or not np.allclose(im.affine, ref.affine, atol=1e-5):
            raise ValueError(f'Grid mismatch: {path}')
        a = im.get_fdata(dtype=np.float32)
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite input: {path}')
        return a
    bvals,bvecs,low,dw = gradient_scheme(args.bvals,args.bvecs,ref.shape[-1],args.b0_threshold)
    bvec = bvecs[:,dw]/np.linalg.norm(bvecs[:,dw],axis=0)
    atlas = Path(args.atlas)
    seg = read(args.seg)
    brain = read(args.mask) > .5
    support = read(atlas/'deformed_mni_mask.nii.gz') > 0
    fit = (seg > 0) & (seg < 900) & (seg != 24) & support
    if args.fit_mask:
        fit = read(args.fit_mask) > .5
    if not brain.any() or not fit.any():
        raise ValueError('Empty brain/fit mask')
    d = dict(seq=read(args.dwi), ref=ref, aff=ref.affine, brain=brain, seg=seg,
             fit=fit, bvec=bvec, bvals=bvals, bvecs=bvecs, bval=bvals[dw],low_idx=low,dwi_idx=dw)
    for key,name in [('alpha','alpha'),('beta','beta'),('kappa','kappa'),('v1','v1mu')]:
        d[key] = read(atlas/f'{name}_deformed_resampled.nii.gz')
    if args.fa_tissue_split:
        wm = brain & np.isin(seg,[2,7,41,46])
        gm = brain & (np.isin(seg,[3,42,8,47,10,11,12,13,17,18,26,49,50,51,52,53,54,58,865,866,819,820,869,870]) |
                      ((seg>=1001)&(seg<=1035)) | ((seg>=2001)&(seg<=2035)))
        alpha = np.ones_like(d['alpha'])
        beta = np.ones_like(d['beta'])
        qc = {}
        for tissue,mask in [('gm',gm),('wm',wm)]:
            coverage = read(atlas/f'mask_{tissue}_deformed_resampled.nii.gz')
            a = read(atlas/f'alpha_{tissue}_deformed_resampled.nii.gz')
            b = read(atlas/f'beta_{tissue}_deformed_resampled.nii.gz')
            # Remove dilution by zero-padded atlas support during interpolation.
            a = np.divide(a,coverage,out=np.ones_like(a),where=coverage>1e-6)
            b = np.divide(b,coverage,out=np.ones_like(b),where=coverage>1e-6)
            valid = mask & (coverage>=.5) & (a>0) & (b>0)
            if not (valid & fit).any():
                raise ValueError(f'No supported {tissue.upper()} Beta prior in fit mask')
            alpha[valid],beta[valid] = a[valid],b[valid]
            qc[tissue] = dict(fit_voxels=int((mask & fit).sum()),supported_voxels=int((valid & fit).sum()))
        d.update(alpha=alpha,beta=beta,fa_prior_qc=qc)
    return d


def design(bvec):
    return tensor_design(bvec)


def eigenvalues(seq, brain, bvec, bval, low, dw):
    s0 = np.maximum(seq[...,low], 0).mean(-1)
    sd = np.maximum(seq[...,dw], 0)
    positive = s0[brain & (s0 > 0)]
    if not positive.size:
        raise ValueError('No positive low-B samples')
    floor = max(float(np.median(positive))*1e-6, np.finfo(np.float32).eps)
    if np.all(bval==bval[0]):
        bval = float(bval[0])
    y = (np.log(np.maximum(s0[brain],floor))[:,None]-np.log(np.maximum(sd[brain],floor)))/bval
    coeff = y@np.linalg.pinv(design(bvec)).T
    matrix = np.stack([coeff[:,[0,3,4]], coeff[:,[3,1,5]], coeff[:,[4,5,2]]], axis=1)
    return np.linalg.eigh(matrix.astype(np.float64))


def maps(seq, d):
    mask = d['brain']
    ev, vec = eigenvalues(seq, mask, d['bvec'], d['bval'], d['low_idx'], d['dwi_idx'])
    clipped = np.maximum(ev, 0)
    md = clipped.mean(1)
    num = np.sqrt(1.5*((clipped-md[:,None])**2).sum(1))
    den = np.sqrt((clipped**2).sum(1))
    fa = np.clip(np.divide(num,den,out=np.zeros_like(num),where=den>0),0,1)
    result = {}
    for key, values in [('adc',md),('fa',fa),('v1',vec[:,:,2]),('eigenvalues',clipped),('eigenvalues_preclip',ev)]:
        a = np.zeros(mask.shape+values.shape[1:], dtype=np.float32)
        a[mask] = values
        result[key] = a
    result['color_fa'] = abs(result['v1'])*result['fa'][...,None]
    result['mean_b0'] = seq[...,d['low_idx']].mean(-1)
    result['mean_dwi'] = seq[...,d['dwi_idx']].mean(-1)
    return result


def b0_prefit(d, out, iterations=500):
    # Fit one bounded common low-B log gain; leave every DWI unchanged.
    out.mkdir()
    brain, seg, seq, b = d['brain'], d['seg'], d['seq'].astype(float), d['bval']
    spacing = np.linalg.norm(d['aff'][:3,:3], axis=0)
    depth = distance_transform_edt(np.pad(binary_fill_holes(brain),1),sampling=spacing)[1:-1,1:-1,1:-1]
    wm = np.isin(seg,[2,41])
    if not (wm & brain).any():
        raise ValueError('No cerebral WM for b0 prefit')
    cut = np.quantile(depth[wm & brain],1/3)
    masks = [binary_erosion(np.isin(seg,labels)) & brain for labels in [[3,42],[2,41],[10,49],[16]]]
    masks[1] &= depth <= cut
    if any(m.sum() <= 10 for m in masks):
        raise ValueError(f'Insufficient b0 class support: {[int(m.sum()) for m in masks]}')
    ev,vec = eigenvalues(seq,brain,d['bvec'],b,d['low_idx'],d['dwi_idx'])
    et = torch.tensor(ev,dtype=torch.float64)
    # A common S0 log gain changes the tensor by H+ (1/b_d) times that gain.
    response = np.linalg.pinv(design(d['bvec']))@(1/b)
    response = np.array([[response[0],response[3],response[4]],
                         [response[3],response[1],response[5]],
                         [response[4],response[5],response[2]]])
    single_shell = np.all(b==b[0])
    if not single_shell:
        tensor = torch.tensor((vec*ev[:,None,:])@vec.transpose(0,2,1),dtype=torch.float64)
        response_t = torch.tensor(response,dtype=torch.float64)
    bm = torch.tensor(brain)
    fm = [torch.tensor(m[brain]) for m in masks]
    xyz = np.meshgrid(*[np.linspace(-1,1,n) for n in brain.shape],indexing='ij')
    powers = [(i,j,k) for i in range(4) for j in range(4) for k in range(4) if 0<i+j+k<=3]
    basis = torch.tensor(np.stack([xyz[0]**i*xyz[1]**j*xyz[2]**k for i,j,k in powers],-1),dtype=torch.float64)
    coef = torch.nn.Parameter(torch.zeros(len(powers),dtype=torch.float64))
    opt = torch.optim.Adam([coef],lr=.02)
    qt = torch.tensor(np.arange(.2,.81,.1),dtype=torch.float64)
    cap = b[0]*.00015 if single_shell else .00015/np.max(abs(np.linalg.eigvalsh(response)))
    weights = dict(relationship=100.,amplitude=.1,bending=.01,new_negative=10.)
    rows = []
    for it in range(iterations+1):
        opt.zero_grad()
        z = torch.tanh(basis@coef)
        A = cap*.5*(z-z[bm].mean())
        eig = et-A[bm][:,None]/b[0] if single_shell else torch.linalg.eigvalsh(tensor-A[bm][:,None,None]*response_t)
        md = eig.mean(-1)
        qs = torch.stack([torch.quantile(md[m],qt) for m in fm])
        qs = qs-qs.mean(0,keepdim=True)
        bend = A.new_zeros(())
        for i in range(3):
            bend = bend+torch.diff(A,n=2,dim=i).square().mean()/spacing[i]**4
            for j in range(i+1,3):
                bend = bend+2*torch.diff(torch.diff(A,dim=i),dim=j).square().mean()/(spacing[i]*spacing[j])**2
        terms = dict(relationship=(qs/.00015).square().mean(),amplitude=(A[bm]/cap).square().mean(),
                     bending=bend*40**4/cap**2,
                     new_negative=(torch.relu(torch.relu(-eig)-torch.relu(-et))/.00015).square().mean())
        loss = sum(weights[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite b0 objective')
        if it%25 == 0 or it == iterations:
            row = dict(iteration=it,total=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()})
            rows.append(row)
            print('B0',json.dumps(row),flush=True)
        if it == iterations:
            break
        loss.backward()
        if not torch.isfinite(coef.grad).all():
            raise FloatingPointError('Nonfinite b0 gradient')
        opt.step()
    field = A.detach().numpy()
    field[~brain] = 0
    seq[...,d['low_idx']] *= np.exp(-field)[...,None]
    np.testing.assert_array_equal(seq[...,d['dwi_idx']],d['seq'][...,d['dwi_idx']])
    save(-field,out/'b0_log_gain.nii.gz',d['ref'])
    save(seq,out/'full_sequence.prefit.nii.gz',d['ref'])
    for name,mask in zip(['cortex','peripheral_wm','thalami','brainstem'],masks):
        save(mask,out/f'{name}.nii.gz',d['ref'])
    table(out/'loss.csv',rows)
    (out/'configuration.json').write_text(json.dumps(dict(iterations=iterations,lr=.02,weights=weights,
        cap_adc=.00015,cap_log_gain=float(cap),tensor_response_per_log_gain=response.tolist(),
        low_b_indices=d['low_idx'].tolist(),class_voxels=[int(m.sum()) for m in masks],powers=powers),indent=2))
    return seq.astype(np.float32)


@torch.no_grad()
def export(m, d, out):
    out.mkdir()
    sd,s0,_,all_u,valid,bvalid = m.joint_warp()
    _,ub,u = m.dense_fields()
    b0 = m.source_b0
    if m.shared_b0_deformation:
        b0,_ = model.pull_resample_dwi(b0,ub.expand(m.n_b0,-1,-1,-1,-1),m.ffd.inverse_spatial_affine)
    B = m.bias_field_full()
    logsd = torch.log1p(m.log_signal_factor*sd.clamp_min(m.signal_floor))-B
    cap = torch.log1p(m.log_signal_factor*s0.clamp_min(m.signal_floor))[None]-1e-4
    corrected = (torch.expm1(torch.minimum(logsd,cap))/m.log_signal_factor).clamp_min(0)
    seq,geom = np.empty_like(d['seq']),np.empty_like(d['seq'])
    seq[...,d['low_idx']] = geom[...,d['low_idx']] = b0.numpy().transpose(1,2,3,0)
    seq[...,d['dwi_idx']] = corrected.numpy().transpose(1,2,3,0)
    geom[...,d['dwi_idx']] = sd.numpy().transpose(1,2,3,0)
    un = u.numpy().transpose(2,3,4,1,0)
    bn = B.numpy().transpose(1,2,3,0)
    jac = model.jacobian_determinant(all_u,m.ffd.inverse_spatial_affine).numpy().transpose(1,2,3,0)
    sum_tolerance = max(1e-5,8*np.finfo(np.float32).eps*m.n_directions*max(1.,float(abs(bn).max())))
    if not np.isfinite(seq).all() or abs(bn.sum(-1)).max() >= sum_tolerance:
        raise FloatingPointError('Invalid output or failed mean-zero bias projection')
    if not m.deform_b0:
        np.testing.assert_array_equal(seq[...,d['low_idx']],d['seq'][...,d['low_idx']])
    for name,a in [('full_sequence.joint_corrected',seq),('dmri.joint_corrected',seq[...,d['dwi_idx']]),
                   ('full_sequence.geom_component',geom),('joint_dwi_displacement_field',un),
                   ('joint_log_bias_field',bn),('geom_jacobian',jac),
                   ('joint_log_bias_field_raw_unconstrained',m.bias_field_full(projected=False).numpy().transpose(1,2,3,0)),
                   ('displacement_magnitude',np.linalg.norm(un,axis=3)),('fixed_fit_mask',d['fit'])]:
        save(a,out/f'{name}.nii.gz',d['ref'])
    if ub.numel():
        save(ub.numpy().transpose(2,3,4,1,0),out/'b0_displacement_field.nii.gz',d['ref'])
    result = maps(seq,d)
    for key,a in result.items():
        save(a,out/f'{key}.nii.gz',d['ref'])
    np.savetxt(out/'full_sequence.bval',d['bvals'][None],fmt='%.8g')
    np.savetxt(out/'full_sequence.bvec',d['bvecs'],fmt='%.8g')
    np.savetxt(out/'dmri.bval',d['bval'][None],fmt='%.8g')
    np.savetxt(out/'dmri.bvec',d['bvecs'][:,d['dwi_idx']],fmt='%.8g')
    (out/'volume_indices.json').write_text(json.dumps(dict(low_b=d['low_idx'].tolist(),
        dwi=d['dwi_idx'].tolist(),note='Zero-based input volume indices; DWI field axes follow dwi order'),indent=2))
    brain = torch.as_tensor(d['brain'])
    mag = np.linalg.norm(un,axis=3)[d['brain']]
    stats = dict(mean_u_mm=float(mag.mean()),p95_u_mm=float(np.percentile(mag,95)),max_u_mm=float(mag.max()),
        max_abs_bias_direction_sum=float(abs(bn.sum(-1)).max()),bias_direction_sum_tolerance=sum_tolerance,
        mean_abs_bias=float(abs(bn[d['brain']]).mean()),
        invalid_fit_fraction=float((~valid[:,m.fit_mask]).float().mean()),
        invalid_b0_fit_fraction=float((~bvalid[:,m.fit_mask]).float().mean()),
        negative_brain_jacobian_percent=float(100*(jac[d['brain']]<0).mean()),
        negative_interpolated_DWI_fraction=float((sd[:,brain]<0).float().mean()),
        bias_corrected_DWI_S0_cap_fraction=float((logsd[:,brain]>cap[:,brain]).float().mean()),
        mean_fa=float(result['fa'][d['brain']].mean()),mean_md=float(result['adc'][d['brain']].mean()),
        negative_eigenvalue_voxel_fraction=float((result['eigenvalues_preclip'][d['brain']]<0).any(-1).mean()))
    (out/'metrics.json').write_text(json.dumps(stats,indent=2))
    return stats


def main():
    p = argparse.ArgumentParser(description='Joint cubic FFD and mean-zero spline bias correction (CPU)')
    for key in ['dwi','bvals','bvecs','mask','seg','atlas','out']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--fit-mask')
    p.add_argument('--b0-threshold',type=float,default=50.,help='Volumes with b <= this value are low-B references (s/mm^2)')
    p.add_argument('--skip-b0-prefit',action='store_true')
    p.add_argument('--fixed-b0',action='store_true')
    p.add_argument('--check',action='store_true')
    p.add_argument('--cycles',type=int,default=300)
    p.add_argument('--threads',type=int,default=2)
    # chosen grid spacing for distortion and bias models. We found 30mm to be best for our data, but your distortions (that you care about) may be finer or coarser!!
    p.add_argument('--grid',type=float,choices=[20.,30.,40.],default=30.,help='Shared cubic-spline spacing for distortion and bias, in mm')
    p.add_argument('--bias-grid',type=float,choices=[20.,30.,40.],help='Legacy option; must equal --grid')
    # use tissue class separation if your FA is pretty homogenous
    p.add_argument('--fa-tissue-split',action='store_true',help='Use GM/WM-specific Beta priors instead of pooled priors')
    p.add_argument('--max-disp',type=float,default=20.)
    p.add_argument('--geo-lr',type=float,default=300.)
    p.add_argument('--bias-lr',type=float,default=.01)
    for key,value in dict(tensor=400.,fa=.05,v1=.2,bias=.2,disp=.0025,bend=.625,common=.02).items():
        p.add_argument('--lambda-'+key,type=float,default=value)
    a = p.parse_args()
    if a.bias_grid is not None and a.bias_grid != a.grid:
        p.error('--bias-grid must equal --grid; the shared spacing controls both models')
    a.bias_grid = a.grid
    if a.cycles < 0 or a.threads < 1 or min(a.grid,a.bias_grid,a.max_disp,a.geo_lr,a.bias_lr) <= 0:
        p.error('Invalid cycles, threads, spacing, cap or learning rate')
    weights = {k:getattr(a,'lambda_'+k) for k in ['tensor','fa','v1','bias','disp','bend','common']}
    if not all(np.isfinite(x) and x >= 0 for x in weights.values()):
        p.error('Weights must be finite and nonnegative')
    torch.set_num_threads(a.threads)
    torch.manual_seed(17)
    d = load(a)
    c = dict(grid=a.grid,bias_grid=a.bias_grid,max_disp=a.max_disp,shared_b0=not a.fixed_b0,
             cycles=a.cycles,geo_lr=a.geo_lr,bias_lr=a.bias_lr,weights=weights,
             b0_prefit=not a.skip_b0_prefit,geometry_updates_per_bias=5,device='cpu',
             interpolation='prefiltered cubic, intensity space',geometry_optimizer='SGD',bias_optimizer='Adam',
             fa_tissue_split=a.fa_tissue_split,fa_prior_qc=d.get('fa_prior_qc',{}),
             b0_threshold=a.b0_threshold,low_b_indices=d['low_idx'].tolist(),dwi_indices=d['dwi_idx'].tolist(),
             dwi_bvalues=d['bval'].tolist())
    if a.check:
        print(json.dumps(dict(shape=d['seq'].shape,fit_voxels=int(d['fit'].sum()),configuration=c),indent=2))
        return
    if not model.LIBRARY.is_file():
        raise FileNotFoundError('Build the sampler first: bash run.sh build')
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    (out/'configuration.json').write_text(json.dumps(c,indent=2))
    hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')}
    (out/'provenance.json').write_text(json.dumps(dict(inputs=vars(a),source_hashes=hashes,
        torch=torch.__version__,numpy=np.__version__,selection='lowest training objective; no HF'),indent=2))
    raw = out/'raw'
    raw.mkdir()
    for key,x in maps(d['seq'],d).items():
        save(x,raw/f'{key}.nii.gz',d['ref'])
    if not a.skip_b0_prefit:
        d['seq'] = b0_prefit(d,out/'b0_prefit')
    m = model.build(d,c)
    ob = torch.optim.Adam([m.bias_coefficients],lr=a.bias_lr)
    og = torch.optim.SGD([m.raw_coefficients],lr=a.geo_lr)
    rows, best, start = [], float('inf'), time.time()
    for cycle in range(a.cycles+1):
        if cycle:
            for block in ['bias']+['geometry']*5:
                m.bias_coefficients.requires_grad_(block=='bias')
                m.raw_coefficients.requires_grad_(block=='geometry')
                ob.zero_grad(set_to_none=True)
                og.zero_grad(set_to_none=True)
                terms = model.parts(m)
                loss = sum(weights[k]*v for k,v in terms.items())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite {block} loss at cycle {cycle}')
                loss.backward()
                param = m.bias_coefficients if block=='bias' else m.raw_coefficients
                if param.grad is None or not torch.isfinite(param.grad).all():
                    raise FloatingPointError(f'Nonfinite {block} gradient at cycle {cycle}')
                torch.nn.utils.clip_grad_norm_([param],5 if block=='bias' else 1)
                (ob if block=='bias' else og).step()
        with torch.no_grad():
            terms = model.parts(m)
            total = float(sum(weights[k]*v for k,v in terms.items()))
            if not np.isfinite(total):
                raise FloatingPointError('Nonfinite objective')
            if total < best:
                best, best_cycle = total, cycle
                state = (m.raw_coefficients.clone(),m.bias_coefficients.clone())
            if cycle%10 == 0 or cycle == a.cycles:
                row = dict(cycle=cycle,seconds=time.time()-start,total=total,
                    **{k+'_raw':float(v) for k,v in terms.items()},
                    **{k+'_weighted':weights[k]*float(v) for k,v in terms.items()})
                rows.append(row)
                table(out/'loss.csv',rows)
                print('JOINT',json.dumps(row),flush=True)
            if cycle and cycle%25 == 0:
                torch.save(dict(ffd=m.raw_coefficients,bias=m.bias_coefficients,cycle=cycle,
                    geometry_optimizer=og.state_dict(),bias_optimizer=ob.state_dict()),out/'checkpoint.pt')
    with torch.no_grad():
        m.raw_coefficients.copy_(state[0])
        m.bias_coefficients.copy_(state[1])
    stats = export(m,d,out/'final')
    stats.update(best_cycle=best_cycle,total_loss=best,seconds=time.time()-start)
    torch.save(dict(ffd=state[0],bias=state[1],config=c,best_cycle=best_cycle),out/'parameters.pt')
    (out/'result.json').write_text(json.dumps(stats,indent=2))
    (out/'COMPLETE').write_text('CPU_ONLY\n')
    print('COMPLETE',json.dumps(stats),flush=True)


if __name__ == '__main__':
    main()
