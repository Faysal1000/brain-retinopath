"""Resumable downloads: HCP (S3, needs keys in .env), HCP 7T retinotopy results (OSF, open), NSD (S3, open)."""
import os, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import boto3, requests
from boto3.s3.transfer import TransferConfig
from tqdm.auto import tqdm
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from botocore.config import Config
from .core import save_json, load_json

HCP_BUCKET = "hcp-openaccess"
OSF_FILES = {"prfresults.mat": "https://osf.io/download/yus6t/",
             "atlas.mat": "https://osf.io/download/523df/"}
# One connection to these servers gives 1-2 MB/s whatever the line speed, so large files are fetched as many byte
# ranges at once. S3 retries time-outs and dropped connections itself. The connection pool is shared by all download
# threads and must hold workers x PARTS_AT_ONCE connections, or the extra threads wait for a free one.
PART_BYTES, PARTS_AT_ONCE = 8 << 20, 16
S3_CONFIG = dict(retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=30, read_timeout=120,
                 max_pool_connections=512)
S3_TRANSFER = TransferConfig(multipart_threshold=PART_BYTES, multipart_chunksize=PART_BYTES, max_concurrency=PARTS_AT_ONCE)
# no compression: sizes and byte ranges then count the bytes of the file itself (GitHub gzips by default)
PLAIN = {"Accept-Encoding": "identity"}
RETRY = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


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
                        aws_secret_access_key=env["S3SECRET_KEY"], region_name="us-east-1",
                        config=Config(**S3_CONFIG))


def anon_client(region="us-east-2"):
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, **S3_CONFIG), region_name=region)


def s3_get(s3, bucket, key, dest, progress=None):
    """Download one object unless a file of the same size is already there. Returns (key, size, etag).
    progress: called with each number of bytes received (a shared progress bar)."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    head = s3.head_object(Bucket=bucket, Key=key)
    size, etag = head["ContentLength"], head["ETag"].strip('"')
    if not (dest.exists() and dest.stat().st_size == size):
        # temporary name unique to this thread: two downloads of the same file must not share it
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
        s3.download_file(bucket, key, str(tmp), Callback=progress, Config=S3_TRANSFER); tmp.replace(dest)
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


def s3_size(s3, bucket, key, dest):
    """(bytes of the object, bytes still to download, etag): 0 to download when a file of the same size is on disk."""
    head = s3.head_object(Bucket=bucket, Key=key); size = head["ContentLength"]
    dest = Path(dest)
    return size, 0 if dest.exists() and dest.stat().st_size == size else size, head["ETag"].strip('"')


def _retry(fn, what, tries=8):
    """fn() retried after a time-out, a dropped connection or a server-side error (HTTP 5xx, 429), waiting 10 s,
    20 s, ... up to 5 min between tries. Other HTTP errors (404, 403) are raised at once."""
    for i in range(tries):
        try:
            return fn()
        except (*RETRY, requests.HTTPError) as e:
            code = e.response.status_code if isinstance(e, requests.HTTPError) and e.response is not None else None
            if i == tries - 1 or (isinstance(e, requests.HTTPError) and not (code == 429 or (code or 0) >= 500)):
                raise
            wait = min(10 * 2 ** i, 300)
            print(f"{what}: {type(e).__name__}, retrying in {wait} s ({i + 1}/{tries - 1})", flush=True)
            time.sleep(wait)


def _size(url, timeout):
    """Size of the file at `url` if the server serves byte ranges, else None."""
    with requests.get(url, headers={**PLAIN, "Range": "bytes=0-0"}, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        return int(cr.rsplit("/", 1)[1]) if r.status_code == 206 and cr.rsplit("/", 1)[-1].isdigit() else None


def _fetch(url, f, a, b, timeout, progress):
    """Bytes a..b (inclusive; b None = to the end) of `url` appended to file `f`, continuing what `f` already holds."""
    have = f.stat().st_size if f.exists() else 0
    if b is not None and have == b - a + 1:
        return
    if b is not None and have > b - a + 1:                            # left by an older, broken run
        progress(-have); f.unlink(); have = 0
    hdr = {"Range": f"bytes={a + have}-{'' if b is None else b}"} if have or b is not None else {}
    with requests.get(url, headers={**PLAIN, **hdr}, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        if hdr and r.status_code != 206:                              # server ignored Range: start again
            have = 0; progress(-f.stat().st_size if f.exists() else 0)
        with open(f, "ab" if have else "wb") as o:
            for chunk in r.iter_content(1 << 20):
                o.write(chunk); progress(len(chunk))
    if b is not None and f.stat().st_size != b - a + 1:
        raise requests.exceptions.ChunkedEncodingError(f"{f.name}: connection closed early")


def http_get(url, dest, tries=8, timeout=60):
    """Download a URL to `dest` with a progress bar; skipped when `dest` exists. If the server serves byte ranges the
    file is fetched as parts of PART_BYTES, PARTS_AT_ONCE at a time, each kept in `dest`.parts/ until all are there;
    otherwise as one stream into `dest`.part. Time-outs are retried and a rerun continues from the bytes on disk."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = _retry(lambda: _size(url, timeout), dest.name, tries)
    lock = threading.Lock()
    with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name) as bar:
        def update(n):
            with lock: bar.update(n)
        if size is None:
            tmp = dest.with_name(dest.name + ".part")
            if tmp.exists(): update(tmp.stat().st_size)
            _retry(lambda: _fetch(url, tmp, 0, None, timeout, update), dest.name, tries)
        else:
            parts = dest.with_name(dest.name + ".parts"); parts.mkdir(exist_ok=True)
            ranges = [(parts / f"{i:06d}", a, min(a + PART_BYTES, size) - 1) for i, a in enumerate(range(0, size, PART_BYTES))]
            update(sum(f.stat().st_size for f, _, _ in ranges if f.exists()))
            with ThreadPoolExecutor(PARTS_AT_ONCE) as ex:
                list(ex.map(lambda r: _retry(lambda: _fetch(url, *r, timeout, update), dest.name, tries), ranges))
            tmp = dest.with_name(dest.name + ".part")
            with open(tmp, "wb") as o:
                for f, _, _ in ranges:
                    with open(f, "rb") as i: shutil.copyfileobj(i, o, 1 << 24)
            shutil.rmtree(parts)
    tmp.replace(dest)
    return dest


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


def hcp_figure_keys(sid, release="HCP_1200"):
    """Very inflated surfaces of both hemispheres (same vertices and faces as the white surface); figures only."""
    return [f"{release}/{sid}/T1w/fsaverage_LR32k/{sid}.{h}.very_inflated_MSMAll.32k_fs_LR.surf.gii" for h in "LR"]


def hcp_download(cfg, sid, kind="3T", release="HCP_1200", s3=None):
    s3 = s3 or hcp_client(cfg)
    return [s3_get(s3, HCP_BUCKET, k, cfg.P.hcp / k) for k in hcp_keys(sid, kind, release)]


def osf_download(cfg, names=tuple(OSF_FILES)):
    for n in names:
        http_get(OSF_FILES[n], cfg.P.osf / n)


def update_manifest(cfg, rows):
    """rows: (key, size, etag). Single-part S3 ETags are MD5 sums; multipart ones are recorded as-is."""
    p = cfg.P.data / "manifest.json"
    man = load_json(p) if p.exists() else {}
    for key, size, etag in rows:
        man[key] = {"bytes": size, "etag": etag}
    save_json(man, p)
    return man
