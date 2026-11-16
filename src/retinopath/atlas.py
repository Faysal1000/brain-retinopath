"""Anatomical retinotopy atlas (Benson et al. 2014, v4.0 from neuropythy): a key that needs no functional scan."""

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from .download import http_get
from .nsd import FS, bridge_file

MAPS = "https://raw.githubusercontent.com/noahbenson/neuropythy/master/neuropythy/lib/data/fsaverage/surf/{fs}.benson14_{q}.v4_0.mgz"
SPHERE = (
    "https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/"
    "standard_mesh_atlases/resample_fsaverage/fsaverage_std_sphere.{h}.164k_fsavg_{h}.surf.gii"
)


def atlas_files(cfg, hemi):
    """Download once (about 6 MB per hemisphere). Returns {angle, eccen, varea, sphere: path}."""
    folder = cfg.P.osf / "benson14"
    folder.mkdir(parents=True, exist_ok=True)
    urls = {quantity: MAPS.format(fs=FS[hemi], q=quantity) for quantity in ("angle", "eccen", "varea")} | {
        "sphere": SPHERE.format(h=hemi)
    }
    out = {}
    for quantity, url in urls.items():
        out[quantity] = folder / url.rsplit("/", 1)[1]
        http_get(url, out[quantity])
    return out


def fsaverage_maps(cfg, hemi):
    """fsaverage sphere, polar angle (pRF convention: 0 = right horizontal, 90 = upper vertical), eccentricity and
    area of one hemisphere. The atlas angle runs from 0 (upper vertical) to 180 (lower vertical) in each hemisphere."""
    files = atlas_files(cfg, hemi)
    angle, eccentricity, area = (
        np.asarray(nib.load(files[quantity]).dataobj).ravel().astype(np.float32)
        for quantity in ("angle", "eccen", "varea")
    )
    angle = np.mod(90 - angle, 360) if hemi == "L" else 90 + angle
    return nib.load(files["sphere"]).darrays[0].data, angle, eccentricity, area


def atlas_key_source(cfg, anat, session):
    """Atlas values at this session's vertices (nearest fsaverage vertex on the registered sphere), in the layout
    metrics.get_key expects. r2 is 100 where the atlas places the vertex in V1-V3 and 0 elsewhere."""
    out, offset = {quantity: [] for quantity in ("ang", "ecc", "r2")}, 0
    for hemi_number, hemi in enumerate(anat["hemis"]):
        sphere, angle, eccentricity, area = fsaverage_maps(cfg, hemi)
        if session["kind"].startswith("nsd"):  # native mesh: its own registration to fsaverage
            registered = nib.freesurfer.read_geometry(
                cfg.P.nsd / f"nsddata/freesurfer/{session['sid']}/surf/{FS[hemi]}.sphere.reg"
            )[0]
            points = registered[anat["vert_id"][anat["hemi"] == hemi_number] - offset]
            offset += len(registered)
        else:  # HCP: fsLR 32k mesh placed on the fsaverage sphere
            points = nib.load(bridge_file(cfg, hemi)).darrays[0].data
        nearest = cKDTree(sphere).query(points)[1]
        for quantity, values in (
            ("ang", angle[nearest]),
            ("ecc", eccentricity[nearest]),
            ("r2", 100.0 * np.isin(area[nearest], (1, 2, 3))),
        ):
            out[quantity].append(values)
    return {f"prf_{quantity}": np.concatenate(values).astype(np.float32)[:, None] for quantity, values in out.items()}
