"""Natural Scenes Dataset adapter: 8 subjects, real 3T diffusion (acquired at 1.5 mm, b = 1500 and 3000, distributed
resampled to the 0.8 mm anatomy), FreeSurfer surfaces, measured pRF maps and hand-drawn V1-V3 labels.
Produces the same `anat` dictionary and diffusion crop as subject.py, on a 1.25 mm grid, so everything downstream
(fODF, tracking, scores, network inference) runs unchanged. Public S3 bucket, no keys. Terms: research use only,
no redistribution."""
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from .degrade import resample
from .download import anon_client, s3_get
from .subject import crop_box, shifted_affine, tissue_maps

BUCKET = "natural-scenes-dataset"
BRIDGE = ("https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/"
          "standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii")
ROI_TO_AREA = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}             # prf-visualrois: V1v V1d V2v V2d V3v V3d (7 = hV4)
FS = {"L": "lh", "R": "rh"}


def nsd_keys(sid, run=1):
    d, f, a = f"nsddata_diffusion/ppdata/{sid}", f"nsddata/freesurfer/{sid}", f"nsddata/ppdata/{sid}/anat"
    keys = [f"{d}/run_{run}/dwi/dwi.{e}" for e in ("nii.gz", "bvals", "bvecs")] + [f"{a}/aseg_0pt8.nii.gz"]
    for h in FS.values():
        keys += [f"{d}/run_{run}/cortexmap/surf/{h}.cras.{s}.surf.gii" for s in ("white", "pial")]
        keys += [f"{f}/label/{h}.{m}.mgz" for m in ("prfangle", "prfeccentricity", "prfR2", "prf-visualrois")]
        keys += [f"{f}/surf/{h}.sphere.reg"]
    return keys


def bridge_file(cfg, h):
    """HCP's standard sphere that places the fsLR 32k mesh on the fsaverage sphere (downloaded once, about 0.7 MB)."""
    import requests
    f = cfg.P.osf / "fsaverage_bridge" / f"fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(BRIDGE.format(h=h), timeout=120); r.raise_for_status(); f.write_bytes(r.content)
    return f


def template_prf(cfg, sid, h):
    """HCP group-average retinotopy carried to this subject's surface through the fsaverage sphere (nearest vertex):
    an atlas-style key that uses nothing measured in this subject. Returns {ang, ecc, r2} per native vertex."""
    from .subject import load_prf
    sph = nib.freesurfer.read_geometry(cfg.P.nsd / f"nsddata/freesurfer/{sid}/surf/{FS[h]}.sphere.reg")[0]
    lr = nib.load(bridge_file(cfg, h)).darrays[0].data
    nn = cKDTree(lr).query(sph)[1]
    g = load_prf(cfg, "999999")[h]
    return {q: g[q][nn] for q in ("ang", "ecc", "r2")}


def nsd_download(cfg, sid, run=1, s3=None):
    s3 = s3 or anon_client()
    return [s3_get(s3, BUCKET, k, cfg.P.nsd / k) for k in nsd_keys(sid, run)]


def _mgz(cfg, sid, h, name):
    return np.asarray(nib.load(cfg.P.nsd / f"nsddata/freesurfer/{sid}/label/{FS[h]}.{name}.mgz").dataobj).ravel()


def build_anat(cfg, sid, run=1):
    """Same `anat` dict as subject.build_anat, from NSD native surfaces, measured pRF maps (fit 0 only; fits 1 and 2
    are NaN) and prf-visualrois labels, plus template keys tpl_*. Vertices outside the crop box are dropped."""
    hemis = "".join(cfg.hemis)
    d = cfg.P.nsd / f"nsddata_diffusion/ppdata/{sid}/run_{run}"
    img = nib.load(d / "dwi/dwi.nii.gz")
    V, N, H, A, D, PRF, TPL, Fc, off = [], [], [], [], [], {q: [] for q in ("ang", "ecc", "r2")}, {q: [] for q in ("ang", "ecc", "r2")}, [], 0
    for hi, h in enumerate(hemis):
        gw = nib.load(d / f"cortexmap/surf/{FS[h]}.cras.white.surf.gii")
        w = gw.darrays[0].data.astype(np.float32); Fc.append(gw.darrays[1].data.astype(np.int64) + off); off += len(w)
        for q, v in template_prf(cfg, sid, h).items(): TPL[q].append(v)
        p = nib.load(d / f"cortexmap/surf/{FS[h]}.cras.pial.surf.gii").darrays[0].data.astype(np.float32)
        n = p - w; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        roi = _mgz(cfg, sid, h, "prf-visualrois").astype(int)
        V.append(w); N.append(n); H.append(np.full(len(w), hi, np.int8))
        A.append(np.vectorize(lambda v: ROI_TO_AREA.get(v, 0))(roi).astype(np.int8)); D.append(np.isin(roi, [2, 4, 6]))
        for q, name in (("ang", "prfangle"), ("ecc", "prfeccentricity"), ("r2", "prfR2")):
            x = _mgz(cfg, sid, h, name).astype(np.float32)
            PRF[q].append(np.stack([x, np.full_like(x, np.nan), np.full_like(x, np.nan)], 1))    # no split-half fits in NSD
    V, N, H, A, D = (np.concatenate(x) for x in (V, N, H, A, D))
    PRF = {q: np.concatenate(v) for q, v in PRF.items()}; TPL = {q: np.concatenate(v) for q, v in TPL.items()}; Fc = np.concatenate(Fc)
    lo, hi_ = crop_box(V[A > 0], img.affine, img.shape, cfg.crop_margin_mm)
    shape08 = tuple(int(x) for x in hi_ - lo); aff08 = shifted_affine(img.affine, lo)
    _, aff = resample(np.zeros(shape08, np.float32), aff08, cfg.ref_voxel, lowpass=False)        # 1.25 mm grid, same box
    shape = tuple(max(2, int(round(s * abs(aff08[0, 0]) / cfg.ref_voxel))) for s in shape08)
    pve, sdist = tissue_maps(nib.load(cfg.P.nsd / f"nsddata/ppdata/{sid}/anat/aseg_0pt8.nii.gz"), aff, shape)
    inv = np.linalg.inv(aff); vv = nib.affines.apply_affine(inv, V)
    keep = np.all((vv > -1) & (vv < np.array(shape)), axis=1)                 # only vertices inside the box
    rot = inv[:3, :3] * np.abs(np.diag(aff)[:3])
    new = np.full(len(V), -1); new[keep] = np.arange(keep.sum()); Fc = new[Fc]; Fc = Fc[(Fc >= 0).all(1)]     # mesh inside the box
    out = dict(sid=str(sid), affine=aff.astype(np.float32), start=lo, shape=np.array(shape), shape08=np.array(shape08),
               pve=pve, sdist=sdist, verts=V[keep], normals=N[keep], verts_vox=vv[keep].astype(np.float32),
               normals_vox=(N[keep] @ rot.T).astype(np.float32), hemi=H[keep], area=A[keep], dorsal=D[keep],
               vert_id=np.flatnonzero(keep).astype(np.int32), faces=Fc.astype(np.int32), hemis=hemis,
               **{f"prf_{q}": v[keep] for q, v in PRF.items()}, **{f"tpl_{q}": v[keep] for q, v in TPL.items()})
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    dist, nn = cKDTree(out["verts_vox"]).query(grid, distance_upper_bound=5.0 / cfg.ref_voxel)
    nf = np.zeros((len(grid), 3), np.float32); ok = np.isfinite(dist); nf[ok] = -out["normals_vox"][nn[ok]]
    out["normal_field"] = nf.reshape(*shape, 3).transpose(3, 0, 1, 2)
    return out


def load_dwi_crop(cfg, sid, anat, run=1):
    """Diffusion crop resampled from the distributed 0.8 mm grid to 1.25 mm (the acquisition was 1.5 mm, so no
    extra smoothing is applied). b-values rounded to the nearest 500."""
    d = cfg.P.nsd / f"nsddata_diffusion/ppdata/{sid}/run_{run}/dwi"
    s, n = anat["start"], anat["shape08"]
    sl = tuple(slice(int(a), int(a + b)) for a, b in zip(s, n))
    img = nib.load(d / "dwi.nii.gz")
    dwi = np.asarray(img.dataobj[sl + (slice(None),)], dtype=np.float32)
    dwi, _ = resample(dwi, shifted_affine(img.affine, s), cfg.ref_voxel, lowpass=False)
    bvals = (np.round(np.loadtxt(d / "dwi.bvals") / 500) * 500).astype(int)
    bvecs = np.loadtxt(d / "dwi.bvecs").T.astype(np.float32)
    if cfg.nsd.flip_x:                      # NSD b-vectors need their x sign flipped for this file orientation
        bvecs[:, 0] *= -1
    return dwi, bvals, bvecs, dwi[..., bvals == 0].mean(-1) > 0
