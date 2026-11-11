"""Load one HCP subject into a compact occipital crop: diffusion, tissue maps, surfaces, retinotopy."""
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .degrade import resample, to_grid

ASSETS = Path(__file__).parent / "assets"     # fslr32k_cortex_index.npz: which fsLR 32k vertices are cortex (standard HCP mesh, no subject data)

WANG_TO_AREA = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}          # V1v,V1d,V2v,V2d,V3v,V3d -> 1,2,3
WM_LABELS = [2, 41, 7, 46, 16, 28, 60, 77, 85] + list(range(251, 256))
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44, 31, 63]
QUANT = ["ang", "ecc", "gain", "meanvol", "r2", "rfsize"]     # order in prfresults.mat


# ---------------------------------------------------------------- retinotopy (OSF bw9ec)
def prf_subject_ids(cfg):
    """Subject IDs in prfresults.mat as strings (includes the 9999xx group averages)."""
    with h5py.File(cfg.P.osf / "prfresults.mat", "r") as f:
        return [str(int(s)) for s in np.array(f["subjectids"]).ravel()]


def load_prf(cfg, sid):
    """Per-vertex pRF results on the fsLR 32k mesh. Returns {hemi: {quantity: (32492, 3 fits)}}; NaN off cortex.
    Fits: 0 = all runs, 1 = first half, 2 = second half."""
    idx = np.load(ASSETS / "fslr32k_cortex_index.npz")
    with h5py.File(cfg.P.osf / "prfresults.mat", "r") as f:
        ids = [str(int(s)) for s in np.array(f["subjectids"]).ravel()]
        # h5py axis order: (fit, subject, quantity, grayordinate). The first 59412 grayordinates are the cortical
        # vertices, left then right; `idx` places them on the 32492-vertex fsLR meshes.
        res = f["allresults"][:, ids.index(str(sid)), :, :59412]      # (fit, quantity, grayordinate)
    out, off = {}, 0
    for h in "LR":
        n = len(idx[h]); d = {}
        for q, name in enumerate(QUANT):
            a = np.full((32492, 3), np.nan, np.float32)
            a[idx[h]] = res[:, q, off:off + n].T
            d[name] = a
        out[h] = d; off += n
    return out


def load_wang(cfg):
    """Group atlas labels (Wang et al. 2015) on the fsLR 32k mesh -> {hemi: area (0, 1=V1, 2=V2, 3=V3), dorsal flag}."""
    idx = np.load(ASSETS / "fslr32k_cortex_index.npz")
    with h5py.File(cfg.P.osf / "atlas.mat", "r") as f:
        w = np.array(f["wang2015"]).ravel()[:59412].astype(int)
    out, off = {}, 0
    for h in "LR":
        n = len(idx[h]); area = np.zeros(32492, np.int8); dorsal = np.zeros(32492, bool)
        lab = w[off:off + n]
        area[idx[h]] = np.vectorize(lambda v: WANG_TO_AREA.get(v, 0))(lab)
        dorsal[idx[h]] = np.isin(lab, [2, 4, 6])
        out[h] = (area, dorsal); off += n
    return out


# ---------------------------------------------------------------- anatomy
def subject_dir(cfg, sid, release="HCP_1200"):
    return cfg.P.hcp / release / str(sid) / "T1w"


def load_surfaces(cfg, sid, release="HCP_1200", hemis="LR"):
    """White-surface vertices (world mm), faces and outward unit normals (towards the pial surface)."""
    d = subject_dir(cfg, sid, release) / "fsaverage_LR32k"
    out = {}
    for h in hemis:
        w = nib.load(d / f"{sid}.{h}.white_MSMAll.32k_fs_LR.surf.gii")
        p = nib.load(d / f"{sid}.{h}.pial_MSMAll.32k_fs_LR.surf.gii")
        v, faces = w.darrays[0].data.astype(np.float32), w.darrays[1].data.astype(np.int32)
        n = p.darrays[0].data - v
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-8
        out[h] = dict(verts=v, faces=faces, normals=n.astype(np.float32))
    return out


def crop_box(verts_world, affine, shape, margin_mm):
    """Voxel bounding box (start, stop) around a set of world points."""
    ijk = nib.affines.apply_affine(np.linalg.inv(affine), verts_world)
    vox = np.abs(np.diag(affine)[:3])
    lo = np.floor(ijk.min(0) - margin_mm / vox).astype(int)
    hi = np.ceil(ijk.max(0) + margin_mm / vox).astype(int) + 1
    return np.clip(lo, 0, None), np.minimum(hi, shape[:3])


def shifted_affine(affine, start):
    a = affine.copy(); a[:3, 3] = nib.affines.apply_affine(affine, start); return a


def tissue_maps(seg_img, crop_affine, crop_shape):
    """Partial-volume fractions (wm, gm, csf) and signed distance to the white-matter boundary (mm, negative inside
    white matter) on the crop grid, by averaging the FreeSurfer segmentation (0.7 mm HCP, 0.8 mm NSD) into each
    crop voxel. Label 0 (background) counts as csf; everything not wm or csf is gm. The distance is computed on
    the segmentation grid (cropped with a 6 mm margin) before averaging. Returns (3, X, Y, Z), (X, Y, Z) float32."""
    seg = np.asanyarray(seg_img.dataobj)
    lo, hi = crop_box(nib.affines.apply_affine(crop_affine, np.array([[0, 0, 0], np.array(crop_shape) - 1.0])),
                      seg_img.affine, seg.shape, 6.0)
    seg = seg[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    wm = np.isin(seg, WM_LABELS); csf = np.isin(seg, CSF_LABELS) | (seg == 0); gm = ~wm & ~csf
    hv = float(abs(seg_img.affine[0, 0]))
    sd = (ndimage.distance_transform_edt(~wm) - ndimage.distance_transform_edt(wm)) * hv
    # nearest crop-grid index of every high-resolution voxel; fractions are the mean over the voxels per crop voxel
    ii = np.stack(np.meshgrid(*[np.arange(s) for s in seg.shape], indexing="ij"), -1).reshape(-1, 3) + lo
    to_crop = np.linalg.inv(crop_affine) @ seg_img.affine
    ijk = np.rint(nib.affines.apply_affine(to_crop, ii)).astype(int)
    ok = np.all((ijk >= 0) & (ijk < np.array(crop_shape)), axis=1)
    flat = np.ravel_multi_index(ijk[ok].T, crop_shape); n = int(np.prod(crop_shape))
    cnt = np.bincount(flat, minlength=n).astype(np.float32); cnt[cnt == 0] = 1
    avg = lambda a: (np.bincount(flat, a.reshape(-1)[ok].astype(np.float32), minlength=n) / cnt).reshape(crop_shape)
    return np.stack([avg(wm), avg(gm), avg(csf)]).astype(np.float32), avg(sd).astype(np.float32)


def build_anat(cfg, sid, release="HCP_1200", dwi_kind="Diffusion"):
    """Everything except the diffusion signal. Coordinates are stored both in world mm and crop-voxel units."""
    hemis = "".join(cfg.hemis)
    d = subject_dir(cfg, sid, release)
    img = nib.load(d / dwi_kind / "data.nii.gz")
    surf, wang = load_surfaces(cfg, sid, release, hemis), load_wang(cfg)
    roi = np.concatenate([surf[h]["verts"][wang[h][0] > 0] for h in hemis])
    start, stop = crop_box(roi, img.affine, img.shape, cfg.crop_margin_mm)
    shape = tuple(int(x) for x in stop - start)
    aff = shifted_affine(img.affine, start)
    vox, native = float(nib.affines.voxel_sizes(img.affine).mean()), None
    if abs(vox - cfg.ref_voxel) > 0.01:
        # 7T diffusion is 1.05 mm. Its crop is put on the reference voxel size (same box), so that fODFs, tissue
        # maps and network inputs have the scale the rest of the pipeline and the networks expect.
        native = shape
        _, aff = resample(np.zeros(shape, np.float32), aff, cfg.ref_voxel, lowpass=False)
        shape = tuple(max(2, int(round(s * vox / cfg.ref_voxel))) for s in shape)
    pve, sdist = tissue_maps(nib.load(d / "aparc+aseg.nii.gz"), aff, shape)

    inv = np.linalg.inv(aff); rot = inv[:3, :3] * np.abs(np.diag(aff)[:3])     # world direction -> voxel-axis frame
    V, N, H, A, D, I, Fc, off = [], [], [], [], [], [], [], 0
    for hi, h in enumerate(hemis):
        V.append(surf[h]["verts"]); N.append(surf[h]["normals"]); H.append(np.full(len(V[-1]), hi, np.int8))
        A.append(wang[h][0]); D.append(wang[h][1]); I.append(np.arange(len(V[-1]), dtype=np.int32))
        Fc.append(surf[h]["faces"] + off); off += len(V[-1])
    V, N = np.concatenate(V), np.concatenate(N)
    out = dict(sid=str(sid), affine=aff.astype(np.float32), start=start, shape=np.array(shape),
               pve=pve, sdist=sdist, verts=V, normals=N, verts_vox=nib.affines.apply_affine(inv, V).astype(np.float32),
               normals_vox=(N @ rot.T).astype(np.float32), hemi=np.concatenate(H), area=np.concatenate(A),
               dorsal=np.concatenate(D), vert_id=np.concatenate(I), faces=np.concatenate(Fc), hemis=hemis)
    # voxel-wise inward-normal field near the surface (anatomical input channels)
    tree = cKDTree(out["verts_vox"])
    grid = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    dist, nn = tree.query(grid, distance_upper_bound=5.0 / cfg.ref_voxel)
    nf = np.zeros((len(grid), 3), np.float32); ok = np.isfinite(dist)
    nf[ok] = -out["normals_vox"][nn[ok]]
    out["normal_field"] = nf.reshape(*shape, 3).transpose(3, 0, 1, 2)
    if native:
        out["shape_native"] = np.array(native)
    try:      # retinotopy exists only for the 7T cohort
        prf = load_prf(cfg, sid)
        for q in ("ang", "ecc", "r2"):
            out[f"prf_{q}"] = np.concatenate([prf[h][q] for h in hemis])
    except (ValueError, OSError, KeyError):
        pass
    return out


def load_dwi_crop(cfg, sid, anat, release="HCP_1200", dwi_kind="Diffusion"):
    """Cropped diffusion volume (X, Y, Z, N) float32, b-values rounded to the nearest 1000, b-vectors (N, 3),
    brain mask. A scan with another voxel size (7T) is resampled to the grid of `anat`."""
    d = subject_dir(cfg, sid, release) / dwi_kind
    s, n = anat["start"], anat.get("shape_native", anat["shape"])
    sl = tuple(slice(int(a), int(a + b)) for a, b in zip(s, n))
    dwi = np.asarray(nib.load(d / "data.nii.gz").dataobj[sl + (slice(None),)], dtype=np.float32)
    mask = np.asarray(nib.load(d / "nodif_brain_mask.nii.gz").dataobj[sl]) > 0
    if "shape_native" in anat:
        dwi, _ = resample(dwi, shifted_affine(nib.load(d / "data.nii.gz").affine, s), cfg.ref_voxel)
        mask = to_grid(mask.astype(np.float32), anat["shape"]) > 0.5
    bvals = np.round(np.loadtxt(d / "bvals"), -3).astype(int)
    bvecs = np.loadtxt(d / "bvecs").T.astype(np.float32)
    return dwi, bvals, bvecs, mask
