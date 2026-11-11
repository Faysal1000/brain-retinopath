"""Resumable downloads: HCP (S3, needs keys in .env), HCP 7T retinotopy results (OSF, open), NSD (S3, open)."""
import os, threading
from pathlib import Path
import boto3, requests
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from botocore.config import Config
from .core import save_json, load_json

HCP_BUCKET = "hcp-openaccess"
OSF_FILES = {"prfresults.mat": "https://osf.io/download/yus6t/",
             "atlas.mat": "https://osf.io/download/523df/"}


def read_env(path):
    """KEY=value lines of a .env file as a dict; quotes stripped, # comment lines skipped."""
    env = {}
    for line in open(path):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1); env[k.strip()] = v.strip().strip("\"'")
    return env


def hcp_client(cfg):
    if not cfg.P.env_file.exists():
        raise FileNotFoundError(f"{cfg.P.env_file} not found. Copy .env.example to .env and put your HCP S3 keys in it.")
    env = read_env(cfg.P.env_file)
    if not env.get("S3ACCESS_KEY") or not env.get("S3SECRET_KEY"):
        raise KeyError("S3ACCESS_KEY / S3SECRET_KEY missing or empty in .env (see .env.example)")
    return boto3.client("s3", aws_access_key_id=env["S3ACCESS_KEY"],
                        aws_secret_access_key=env["S3SECRET_KEY"], region_name="us-east-1")


def anon_client(region="us-east-2"):
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name=region)


def s3_get(s3, bucket, key, dest):
    """Download one object unless a file of the same size is already there. Returns (key, size, etag)."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    head = s3.head_object(Bucket=bucket, Key=key)
    size, etag = head["ContentLength"], head["ETag"].strip('"')
    if not (dest.exists() and dest.stat().st_size == size):
        # temporary name unique to this thread: two downloads of the same file must not share it
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
        s3.download_file(bucket, key, str(tmp)); tmp.replace(dest)
    return key, size, etag


def s3_dirs(s3, bucket, prefix):
    """Names of the immediate sub-folders under `prefix` (all pages of the listing)."""
    out, tok = [], None
    while True:
        kw = dict(Bucket=bucket, Prefix=prefix, Delimiter="/")
        if tok: kw["ContinuationToken"] = tok
        r = s3.list_objects_v2(**kw)
        out += [p["Prefix"][len(prefix):].strip("/") for p in r.get("CommonPrefixes", [])]
        tok = r.get("NextContinuationToken")
        if not tok: return out


def s3_exists(s3, bucket, key):
    """False only when the object is missing; wrong keys or a network error raise."""
    try:
        s3.head_object(Bucket=bucket, Key=key); return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def hcp_keys(sid, kind="3T", release="HCP_1200"):
    """S3 keys needed for one subject. kind: 3T | 7T (diffusion only; anatomy comes with 3T)."""
    b = f"{release}/{sid}/T1w"
    dif = "Diffusion" if kind == "3T" else "Diffusion_7T"
    keys = [f"{b}/{dif}/{f}" for f in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz")]
    if kind == "3T":
        keys += [f"{b}/aparc+aseg.nii.gz"]
        keys += [f"{b}/fsaverage_LR32k/{sid}.{h}.{s}_MSMAll.32k_fs_LR.surf.gii"
                 for h in "LR" for s in ("white", "pial")]
    return keys


def hcp_download(cfg, sid, kind="3T", release="HCP_1200", s3=None):
    s3 = s3 or hcp_client(cfg)
    return [s3_get(s3, HCP_BUCKET, k, cfg.P.hcp / k) for k in hcp_keys(sid, kind, release)]


def osf_download(cfg, names=tuple(OSF_FILES)):
    for n in names:
        dest = cfg.P.osf / n
        if dest.exists() and dest.stat().st_size > 0 and not dest.with_suffix(".part").exists():
            continue
        with requests.get(OSF_FILES[n], stream=True, timeout=60) as r:
            r.raise_for_status()
            tmp = dest.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20): f.write(chunk)
            tmp.rename(dest)


def update_manifest(cfg, rows):
    """rows: (key, size, etag). Single-part S3 ETags are MD5 sums; multipart ones are recorded as-is."""
    p = cfg.P.data / "manifest.json"
    man = load_json(p) if p.exists() else {}
    for key, size, etag in rows:
        man[key] = {"bytes": size, "etag": etag}
    save_json(man, p)
    return man
