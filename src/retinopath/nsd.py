"""Natural Scenes Dataset (NSD) crops: the same `anat` dictionary and diffusion crop as subject.py, on a 1.25 mm grid,
from NSD diffusion, FreeSurfer surfaces, pRF maps and hand-drawn V1-V3 labels. Research use only, no redistribution."""

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from .degrade import resample
from .download import anon_client, http_get, s3_get
from .subject import crop_box, shifted_affine, tissue_maps

BUCKET = "natural-scenes-dataset"
BRIDGE = (
    "https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/"
    "standard_mesh_atlases/resample_fsaverage/fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii"
)
ROI_TO_AREA = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}  # prf-visualrois: V1v V1d V2v V2d V3v V3d (7 = hV4)
FS = {"L": "lh", "R": "rh"}


def nsd_keys(sid, run=1):
    diffusion, freesurfer, anatomy = (
        f"nsddata_diffusion/ppdata/{sid}",
        f"nsddata/freesurfer/{sid}",
        f"nsddata/ppdata/{sid}/anat",
    )
    keys = [f"{diffusion}/run_{run}/dwi/dwi.{extension}" for extension in ("nii.gz", "bvals", "bvecs")] + [
        f"{anatomy}/aseg_0pt8.nii.gz"
    ]
    for hemi in FS.values():
        keys += [
            f"{diffusion}/run_{run}/cortexmap/surf/{hemi}.cras.{surface}.surf.gii" for surface in ("white", "pial")
        ]
        keys += [
            f"{freesurfer}/label/{hemi}.{label}.mgz"
            for label in ("prfangle", "prfeccentricity", "prfR2", "prf-visualrois")
        ]
        keys += [f"{freesurfer}/surf/{hemi}.sphere.reg"]
    return keys


def bridge_file(cfg, hemi):
    """HCP's standard sphere that places the fsLR 32k mesh on the fsaverage sphere (downloaded once, about 0.7 MB)."""
    return http_get(
        BRIDGE.format(h=hemi),
        cfg.P.osf / "fsaverage_bridge" / f"fs_LR-deformed_to-fsaverage.{hemi}.sphere.32k_fs_LR.surf.gii",
    )


def template_prf(cfg, sid, hemi):
    """HCP group-average retinotopy on this subject's vertices via the fsaverage sphere. Returns {ang, ecc, r2}."""
    from .subject import load_prf

    sphere = nib.freesurfer.read_geometry(cfg.P.nsd / f"nsddata/freesurfer/{sid}/surf/{FS[hemi]}.sphere.reg")[0]
    fslr_sphere = nib.load(bridge_file(cfg, hemi)).darrays[0].data
    nearest = cKDTree(fslr_sphere).query(sphere)[1]
    group = load_prf(cfg, "999999")[hemi]
    return {quantity: group[quantity][nearest] for quantity in ("ang", "ecc", "r2")}


def nsd_download(cfg, sid, run=1, s3=None):
    s3 = s3 or anon_client()
    return [s3_get(s3, BUCKET, key, cfg.P.nsd / key) for key in nsd_keys(sid, run)]


def _mgz(cfg, sid, hemi, name):
    return np.asarray(nib.load(cfg.P.nsd / f"nsddata/freesurfer/{sid}/label/{FS[hemi]}.{name}.mgz").dataobj).ravel()


def build_anat(cfg, sid, run=1):
    """subject.build_anat for NSD (one pRF fit; fits 1 and 2 NaN) plus tpl_* keys; vertices outside the box dropped."""
    hemis = "".join(cfg.hemis)
    folder = cfg.P.nsd / f"nsddata_diffusion/ppdata/{sid}/run_{run}"
    image = nib.load(folder / "dwi/dwi.nii.gz")
    vertices, normals, hemi_index, areas, dorsal, prf, template, faces, offset = (
        [],
        [],
        [],
        [],
        [],
        {quantity: [] for quantity in ("ang", "ecc", "r2")},
        {quantity: [] for quantity in ("ang", "ecc", "r2")},
        [],
        0,
    )
    for hemi_number, hemi in enumerate(hemis):
        white_gifti = nib.load(folder / f"cortexmap/surf/{FS[hemi]}.cras.white.surf.gii")
        white = white_gifti.darrays[0].data.astype(np.float32)
        faces.append(white_gifti.darrays[1].data.astype(np.int64) + offset)
        offset += len(white)
        for quantity, values in template_prf(cfg, sid, hemi).items():
            template[quantity].append(values)
        pial = nib.load(folder / f"cortexmap/surf/{FS[hemi]}.cras.pial.surf.gii").darrays[0].data.astype(np.float32)
        normal = pial - white
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8
        labels = _mgz(cfg, sid, hemi, "prf-visualrois").astype(int)
        vertices.append(white)
        normals.append(normal)
        hemi_index.append(np.full(len(white), hemi_number, np.int8))
        areas.append(np.vectorize(lambda label: ROI_TO_AREA.get(label, 0))(labels).astype(np.int8))
        dorsal.append(np.isin(labels, [2, 4, 6]))
        for quantity, name in (("ang", "prfangle"), ("ecc", "prfeccentricity"), ("r2", "prfR2")):
            values = _mgz(cfg, sid, hemi, name).astype(np.float32)
            prf[quantity].append(
                np.stack([values, np.full_like(values, np.nan), np.full_like(values, np.nan)], 1)
            )  # no split-half fits in NSD
    vertices, normals, hemi_index, areas, dorsal = (
        np.concatenate(parts) for parts in (vertices, normals, hemi_index, areas, dorsal)
    )
    prf = {quantity: np.concatenate(values) for quantity, values in prf.items()}
    template = {quantity: np.concatenate(values) for quantity, values in template.items()}
    faces = np.concatenate(faces)
    lower, upper = crop_box(vertices[areas > 0], image.affine, image.shape, cfg.crop_margin_mm)
    shape_fine = tuple(int(length) for length in upper - lower)
    affine_fine = shifted_affine(image.affine, lower)
    _, affine = resample(
        np.zeros(shape_fine, np.float32), affine_fine, cfg.ref_voxel, lowpass=False
    )  # 1.25 mm grid, same box
    shape = tuple(max(2, int(round(length * abs(affine_fine[0, 0]) / cfg.ref_voxel))) for length in shape_fine)
    pve, sdist = tissue_maps(nib.load(cfg.P.nsd / f"nsddata/ppdata/{sid}/anat/aseg_0pt8.nii.gz"), affine, shape)
    to_voxel = np.linalg.inv(affine)
    vertices_voxel = nib.affines.apply_affine(to_voxel, vertices)
    inside = np.all((vertices_voxel > -1) & (vertices_voxel < np.array(shape)), axis=1)  # only vertices inside the box
    rotation = to_voxel[:3, :3] * np.abs(np.diag(affine)[:3])
    new_index = np.full(len(vertices), -1)
    new_index[inside] = np.arange(inside.sum())
    faces = new_index[faces]
    faces = faces[(faces >= 0).all(1)]  # mesh inside the box
    anat = dict(
        sid=str(sid),
        affine=affine.astype(np.float32),
        start=lower,
        shape=np.array(shape),
        shape08=np.array(shape_fine),
        pve=pve,
        sdist=sdist,
        verts=vertices[inside],
        normals=normals[inside],
        verts_vox=vertices_voxel[inside].astype(np.float32),
        normals_vox=(normals[inside] @ rotation.T).astype(np.float32),
        hemi=hemi_index[inside],
        area=areas[inside],
        dorsal=dorsal[inside],
        vert_id=np.flatnonzero(inside).astype(np.int32),
        faces=faces.astype(np.int32),
        hemis=hemis,
        **{f"prf_{quantity}": values[inside] for quantity, values in prf.items()},
        **{f"tpl_{quantity}": values[inside] for quantity, values in template.items()},
    )
    grid = np.stack(np.meshgrid(*[np.arange(length) for length in shape], indexing="ij"), -1).reshape(-1, 3)
    distance, nearest = cKDTree(anat["verts_vox"]).query(grid, distance_upper_bound=5.0 / cfg.ref_voxel)
    normal_field = np.zeros((len(grid), 3), np.float32)
    near = np.isfinite(distance)
    normal_field[near] = -anat["normals_vox"][nearest[near]]
    anat["normal_field"] = normal_field.reshape(*shape, 3).transpose(3, 0, 1, 2)
    return anat


def load_dwi_crop(cfg, sid, anat, run=1):
    """Diffusion crop from 0.8 mm to 1.25 mm without low-pass (acquired at 1.5 mm); b-values rounded to 500."""
    folder = cfg.P.nsd / f"nsddata_diffusion/ppdata/{sid}/run_{run}/dwi"
    start, shape = anat["start"], anat["shape08"]
    slices = tuple(slice(int(offset), int(offset + size)) for offset, size in zip(start, shape))
    image = nib.load(folder / "dwi.nii.gz")
    dwi = np.asarray(image.dataobj[slices + (slice(None),)], dtype=np.float32)
    dwi, _ = resample(dwi, shifted_affine(image.affine, start), cfg.ref_voxel, lowpass=False)
    bvals = (np.round(np.loadtxt(folder / "dwi.bvals") / 500) * 500).astype(int)
    bvecs = np.loadtxt(folder / "dwi.bvecs").T.astype(np.float32)
    if cfg.nsd.flip_x:
        bvecs[:, 0] *= -1
    return dwi, bvals, bvecs, dwi[..., bvals == 0].mean(-1) > 0
