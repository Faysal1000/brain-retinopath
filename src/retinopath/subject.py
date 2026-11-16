"""Load one HCP subject into a compact occipital crop: diffusion, tissue maps, surfaces, retinotopy."""

from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from .degrade import resample, to_grid

ASSETS = Path(__file__).parent / "assets"  # fslr32k_cortex_index.npz: cortical vertices of the standard fsLR 32k mesh

WANG_TO_AREA = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}  # V1v,V1d,V2v,V2d,V3v,V3d -> 1,2,3
WM_LABELS = [2, 41, 7, 46, 16, 28, 60, 77, 85] + list(range(251, 256))
CSF_LABELS = [4, 5, 14, 15, 24, 43, 44, 31, 63]
QUANT = ["ang", "ecc", "gain", "meanvol", "r2", "rfsize"]  # order in prfresults.mat


# ---------------------------------------------------------------- retinotopy (OSF bw9ec)
def prf_subject_ids(cfg):
    """Subject IDs in prfresults.mat as strings (includes the 9999xx group averages)."""
    with h5py.File(cfg.P.osf / "prfresults.mat", "r") as file:
        return [str(int(subject)) for subject in np.array(file["subjectids"]).ravel()]


def load_prf(cfg, sid):
    """Per-vertex pRF results on the fsLR 32k mesh. Returns {hemi: {quantity: (32492, 3 fits)}}; NaN off cortex.
    Fits: 0 = all runs, 1 = first half, 2 = second half."""
    cortex_index = np.load(ASSETS / "fslr32k_cortex_index.npz")
    with h5py.File(cfg.P.osf / "prfresults.mat", "r") as file:
        subject_ids = [str(int(subject)) for subject in np.array(file["subjectids"]).ravel()]
        # axes (fit, subject, quantity, grayordinate); the first 59412 grayordinates are cortex, left then right
        results = file["allresults"][:, subject_ids.index(str(sid)), :, :59412]  # (fit, quantity, grayordinate)
    out, offset = {}, 0
    for hemi in "LR":
        count = len(cortex_index[hemi])
        maps = {}
        for column, name in enumerate(QUANT):
            values = np.full((32492, 3), np.nan, np.float32)
            values[cortex_index[hemi]] = results[:, column, offset : offset + count].T
            maps[name] = values
        out[hemi] = maps
        offset += count
    return out


def load_wang(cfg):
    """Group atlas labels (Wang et al. 2015) on the fsLR 32k mesh -> {hemi: area (0, 1=V1, 2=V2, 3=V3), dorsal flag}."""
    cortex_index = np.load(ASSETS / "fslr32k_cortex_index.npz")
    with h5py.File(cfg.P.osf / "atlas.mat", "r") as file:
        all_labels = np.array(file["wang2015"]).ravel()[:59412].astype(int)
    out, offset = {}, 0
    for hemi in "LR":
        count = len(cortex_index[hemi])
        area = np.zeros(32492, np.int8)
        dorsal = np.zeros(32492, bool)
        labels = all_labels[offset : offset + count]
        area[cortex_index[hemi]] = np.vectorize(lambda label: WANG_TO_AREA.get(label, 0))(labels)
        dorsal[cortex_index[hemi]] = np.isin(labels, [2, 4, 6])
        out[hemi] = (area, dorsal)
        offset += count
    return out


# ---------------------------------------------------------------- anatomy
def subject_dir(cfg, sid, release="HCP_1200"):
    return cfg.P.hcp / release / str(sid) / "T1w"


def load_surfaces(cfg, sid, release="HCP_1200", hemis="LR"):
    """White-surface vertices (world mm), faces and outward unit normals (towards the pial surface)."""
    folder = subject_dir(cfg, sid, release) / "fsaverage_LR32k"
    out = {}
    for hemi in hemis:
        white = nib.load(folder / f"{sid}.{hemi}.white_MSMAll.32k_fs_LR.surf.gii")
        pial = nib.load(folder / f"{sid}.{hemi}.pial_MSMAll.32k_fs_LR.surf.gii")
        vertices, faces = white.darrays[0].data.astype(np.float32), white.darrays[1].data.astype(np.int32)
        normals = pial.darrays[0].data - vertices
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        out[hemi] = dict(verts=vertices, faces=faces, normals=normals.astype(np.float32))
    return out


def inflated_surface(cfg, sid, hemi, release="HCP_1200"):
    """Very inflated surface of one hemisphere (same mesh as the white surface), for figures; None if missing."""
    path = subject_dir(cfg, sid, release) / "fsaverage_LR32k" / f"{sid}.{hemi}.very_inflated_MSMAll.32k_fs_LR.surf.gii"
    return nib.load(path).darrays[0].data.astype(np.float32) if path.exists() else None


def crop_box(verts_world, affine, shape, margin_mm):
    """Voxel bounding box (start, stop) around a set of world points."""
    voxels = nib.affines.apply_affine(np.linalg.inv(affine), verts_world)
    voxel_mm = np.abs(np.diag(affine)[:3])
    lower = np.floor(voxels.min(0) - margin_mm / voxel_mm).astype(int)
    upper = np.ceil(voxels.max(0) + margin_mm / voxel_mm).astype(int) + 1
    return np.clip(lower, 0, None), np.minimum(upper, shape[:3])


def shifted_affine(affine, start):
    shifted = affine.copy()
    shifted[:3, 3] = nib.affines.apply_affine(affine, start)
    return shifted


def tissue_maps(seg_img, crop_affine, crop_shape):
    """Tissue fractions (wm, gm, csf) and signed distance to the white-matter boundary (mm, negative inside), averaged
    from the FreeSurfer segmentation into each crop voxel. Background counts as csf. Returns (3, X, Y, Z), (X, Y, Z)."""
    seg = np.asanyarray(seg_img.dataobj)
    lower, upper = crop_box(
        nib.affines.apply_affine(crop_affine, np.array([[0, 0, 0], np.array(crop_shape) - 1.0])),
        seg_img.affine,
        seg.shape,
        6.0,
    )
    seg = seg[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]
    wm = np.isin(seg, WM_LABELS)
    csf = np.isin(seg, CSF_LABELS) | (seg == 0)
    gm = ~wm & ~csf
    voxel_mm = float(abs(seg_img.affine[0, 0]))
    signed_distance = (ndimage.distance_transform_edt(~wm) - ndimage.distance_transform_edt(wm)) * voxel_mm
    # nearest crop-grid index of every high-resolution voxel; fractions are the mean over the voxels per crop voxel
    fine_voxels = (
        np.stack(np.meshgrid(*[np.arange(length) for length in seg.shape], indexing="ij"), -1).reshape(-1, 3) + lower
    )
    to_crop = np.linalg.inv(crop_affine) @ seg_img.affine
    crop_voxels = np.rint(nib.affines.apply_affine(to_crop, fine_voxels)).astype(int)
    inside = np.all((crop_voxels >= 0) & (crop_voxels < np.array(crop_shape)), axis=1)
    flat_index = np.ravel_multi_index(crop_voxels[inside].T, crop_shape)
    n_voxels = int(np.prod(crop_shape))
    counts = np.bincount(flat_index, minlength=n_voxels).astype(np.float32)
    counts[counts == 0] = 1
    average = lambda values: (
        np.bincount(flat_index, values.reshape(-1)[inside].astype(np.float32), minlength=n_voxels) / counts
    ).reshape(crop_shape)
    return np.stack([average(wm), average(gm), average(csf)]).astype(np.float32), average(signed_distance).astype(
        np.float32
    )


def build_anat(cfg, sid, release="HCP_1200", dwi_kind="Diffusion"):
    """Everything except the diffusion signal. Coordinates are stored both in world mm and crop-voxel units."""
    hemis = "".join(cfg.hemis)
    folder = subject_dir(cfg, sid, release)
    image = nib.load(folder / dwi_kind / "data.nii.gz")
    surfaces, wang = load_surfaces(cfg, sid, release, hemis), load_wang(cfg)
    roi_vertices = np.concatenate([surfaces[hemi]["verts"][wang[hemi][0] > 0] for hemi in hemis])
    start, stop = crop_box(roi_vertices, image.affine, image.shape, cfg.crop_margin_mm)
    shape = tuple(int(length) for length in stop - start)
    affine = shifted_affine(image.affine, start)
    voxel_mm, native_shape = float(nib.affines.voxel_sizes(image.affine).mean()), None
    if abs(voxel_mm - cfg.ref_voxel) > 0.01:
        # 7T (1.05 mm): same box on the 1.25 mm grid
        native_shape = shape
        _, affine = resample(np.zeros(shape, np.float32), affine, cfg.ref_voxel, lowpass=False)
        shape = tuple(max(2, int(round(length * voxel_mm / cfg.ref_voxel))) for length in shape)
    pve, sdist = tissue_maps(nib.load(folder / "aparc+aseg.nii.gz"), affine, shape)

    to_voxel = np.linalg.inv(affine)
    rotation = to_voxel[:3, :3] * np.abs(np.diag(affine)[:3])  # world direction -> voxel-axis frame
    vertices, normals, hemi_index, areas, dorsal, vertex_ids, faces, offset = [], [], [], [], [], [], [], 0
    for hemi_number, hemi in enumerate(hemis):
        vertices.append(surfaces[hemi]["verts"])
        normals.append(surfaces[hemi]["normals"])
        hemi_index.append(np.full(len(vertices[-1]), hemi_number, np.int8))
        areas.append(wang[hemi][0])
        dorsal.append(wang[hemi][1])
        vertex_ids.append(np.arange(len(vertices[-1]), dtype=np.int32))
        faces.append(surfaces[hemi]["faces"] + offset)
        offset += len(vertices[-1])
    vertices, normals = np.concatenate(vertices), np.concatenate(normals)
    anat = dict(
        sid=str(sid),
        affine=affine.astype(np.float32),
        start=start,
        shape=np.array(shape),
        pve=pve,
        sdist=sdist,
        verts=vertices,
        normals=normals,
        verts_vox=nib.affines.apply_affine(to_voxel, vertices).astype(np.float32),
        normals_vox=(normals @ rotation.T).astype(np.float32),
        hemi=np.concatenate(hemi_index),
        area=np.concatenate(areas),
        dorsal=np.concatenate(dorsal),
        vert_id=np.concatenate(vertex_ids),
        faces=np.concatenate(faces),
        hemis=hemis,
    )
    # voxel-wise inward-normal field near the surface (anatomical input channels)
    tree = cKDTree(anat["verts_vox"])
    grid = np.stack(np.meshgrid(*[np.arange(length) for length in shape], indexing="ij"), -1).reshape(-1, 3)
    distance, nearest = tree.query(grid, distance_upper_bound=5.0 / cfg.ref_voxel)
    normal_field = np.zeros((len(grid), 3), np.float32)
    near = np.isfinite(distance)
    normal_field[near] = -anat["normals_vox"][nearest[near]]
    anat["normal_field"] = normal_field.reshape(*shape, 3).transpose(3, 0, 1, 2)
    if native_shape:
        anat["shape_native"] = np.array(native_shape)
    try:  # retinotopy exists only for the 7T cohort
        prf = load_prf(cfg, sid)
        for quantity in ("ang", "ecc", "r2"):
            anat[f"prf_{quantity}"] = np.concatenate([prf[hemi][quantity] for hemi in hemis])
    except (ValueError, OSError, KeyError):
        pass
    return anat


def load_dwi_crop(cfg, sid, anat, release="HCP_1200", dwi_kind="Diffusion"):
    """Cropped diffusion (X, Y, Z, N), b-values rounded to 1000, b-vectors (N, 3), mask; 7T resampled to `anat`."""
    folder = subject_dir(cfg, sid, release) / dwi_kind
    start, shape = anat["start"], anat.get("shape_native", anat["shape"])
    slices = tuple(slice(int(offset), int(offset + size)) for offset, size in zip(start, shape))
    dwi = np.asarray(nib.load(folder / "data.nii.gz").dataobj[slices + (slice(None),)], dtype=np.float32)
    mask = np.asarray(nib.load(folder / "nodif_brain_mask.nii.gz").dataobj[slices]) > 0
    if "shape_native" in anat:
        dwi, _ = resample(dwi, shifted_affine(nib.load(folder / "data.nii.gz").affine, start), cfg.ref_voxel)
        mask = to_grid(mask.astype(np.float32), anat["shape"]) > 0.5
    bvals = np.round(np.loadtxt(folder / "bvals"), -3).astype(int)
    bvecs = np.loadtxt(folder / "bvecs").T.astype(np.float32)
    return dwi, bvals, bvecs, mask
