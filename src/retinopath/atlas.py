"""Anatomical retinotopy atlas (Benson et al. 2014, template version 4.0 distributed with neuropythy): polar angle and
eccentricity predicted from cortical folding alone, defined on fsaverage. Used as a key that needs no functional
scan, which is what a clinical dataset would have."""
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree
from .nsd import FS, bridge_file

MAPS = "https://raw.githubusercontent.com/noahbenson/neuropythy/master/neuropythy/lib/data/fsaverage/surf/{fs}.benson14_{q}.v4_0.mgz"
SPHERE = ("https://raw.githubusercontent.com/Washington-University/HCPpipelines/master/global/templates/"
          "standard_mesh_atlases/resample_fsaverage/fsaverage_std_sphere.{h}.164k_fsavg_{h}.surf.gii")


def atlas_files(cfg, h):
    """Download once (about 6 MB per hemisphere). Returns {angle, eccen, varea, sphere: path}."""
    import requests
    d = cfg.P.osf / "benson14"; d.mkdir(parents=True, exist_ok=True)
    urls = {q: MAPS.format(fs=FS[h], q=q) for q in ("angle", "eccen", "varea")} | {"sphere": SPHERE.format(h=h)}
    out = {}
    for q, u in urls.items():
        out[q] = d / u.rsplit("/", 1)[1]
        if not out[q].exists():
            r = requests.get(u, timeout=300); r.raise_for_status()
            tmp = out[q].with_name(out[q].name + ".part"); tmp.write_bytes(r.content); tmp.replace(out[q])
    return out


def fsaverage_maps(cfg, h):
    """Atlas on the fsaverage mesh of one hemisphere: sphere coordinates, polar angle in the pRF convention
    (0 = right horizontal meridian, 90 = upper vertical, counter-clockwise), eccentricity (deg), area label.
    The atlas stores angle as 0 (upper vertical) to 180 (lower vertical) for both hemispheres; the left hemisphere
    sees the right visual field."""
    f = atlas_files(cfg, h)
    ang, ecc, area = (np.asarray(nib.load(f[q]).dataobj).ravel().astype(np.float32) for q in ("angle", "eccen", "varea"))
    ang = np.mod(90 - ang, 360) if h == "L" else 90 + ang
    return nib.load(f["sphere"]).darrays[0].data, ang, ecc, area


def atlas_key_source(cfg, anat, ses):
    """Atlas values at this session's vertices (nearest fsaverage vertex on the registered sphere), in the layout
    metrics.get_key expects. r2 is 100 where the atlas places the vertex in V1-V3 and 0 elsewhere."""
    out, off = {q: [] for q in ("ang", "ecc", "r2")}, 0
    for hi, h in enumerate(anat["hemis"]):
        sph, ang, ecc, area = fsaverage_maps(cfg, h)
        if ses["kind"].startswith("nsd"):                        # native mesh: its own registration to fsaverage
            reg = nib.freesurfer.read_geometry(cfg.P.nsd / f"nsddata/freesurfer/{ses['sid']}/surf/{FS[h]}.sphere.reg")[0]
            pts = reg[anat["vert_id"][anat["hemi"] == hi] - off]; off += len(reg)
        else:                                                    # HCP: fsLR 32k mesh placed on the fsaverage sphere
            pts = nib.load(bridge_file(cfg, h)).darrays[0].data
        nn = cKDTree(sph).query(pts)[1]
        for q, v in (("ang", ang[nn]), ("ecc", ecc[nn]), ("r2", 100.0 * np.isin(area[nn], (1, 2, 3)))):
            out[q].append(v)
    return {f"prf_{q}": np.concatenate(v).astype(np.float32)[:, None] for q, v in out.items()}
