"""Resumable downloads: HCP (S3, needs keys in .env), HCP 7T retinotopy results (OSF, open), NSD (S3, open)."""

import os, shutil, threading, time, uuid
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
OSF_FILES = {"prfresults.mat": "https://osf.io/download/yus6t/", "atlas.mat": "https://osf.io/download/523df/"}
# one connection gives only 1-2 MB/s, so files are fetched as many byte ranges at once; a connection without data
# for 30 s is retried
PART_BYTES, PARTS_AT_ONCE = 8 << 20, 16
S3_CONFIG = dict(
    retries={"max_attempts": 10, "mode": "standard"}, connect_timeout=15, read_timeout=30, max_pool_connections=4096
)
S3_TRANSFER = TransferConfig(
    multipart_threshold=PART_BYTES, multipart_chunksize=PART_BYTES, max_concurrency=PARTS_AT_ONCE
)
# no compression: sizes and byte ranges then count the bytes of the file itself (GitHub gzips by default)
PLAIN = {"Accept-Encoding": "identity"}
RETRY = (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError)


def read_env(path):
    """KEY=value lines of a .env file as a dict; quotes stripped, # comment lines skipped."""
    env = {}
    for line in open(path):
        if "=" in line and not line.strip().startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")
    return env


def hcp_client(cfg):
    if not cfg.P.env_file.exists():
        raise FileNotFoundError(
            f"{cfg.P.env_file} not found. Copy .env.example to .env and put your HCP S3 keys in it."
        )
    env = read_env(cfg.P.env_file)
    if not env.get("S3ACCESS_KEY") or not env.get("S3SECRET_KEY"):
        raise KeyError("S3ACCESS_KEY / S3SECRET_KEY missing or empty in .env (see .env.example)")
    return boto3.client(
        "s3",
        aws_access_key_id=env["S3ACCESS_KEY"],
        aws_secret_access_key=env["S3SECRET_KEY"],
        region_name="us-east-1",
        config=Config(**S3_CONFIG),
    )


def anon_client(region="us-east-2"):
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, **S3_CONFIG), region_name=region)


def s3_get(s3, bucket, key, dest, progress=None):
    """Download one object unless a file of the same size is already there. Returns (key, size, etag).
    progress: called with each number of bytes received (a shared progress bar)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    head = s3.head_object(Bucket=bucket, Key=key)
    size, etag = head["ContentLength"], head["ETag"].strip('"')
    if not (dest.exists() and dest.stat().st_size == size):
        # temporary name unique to this thread: two downloads of the same file must not share it
        partial = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
        s3.download_file(bucket, key, str(partial), Callback=progress, Config=S3_TRANSFER)
        partial.replace(dest)
    return key, size, etag


def s3_get_many(items, connections, progress=None, on_done=None):
    """Download items (s3, bucket, key, dest, size, etag) as PART_BYTES ranges through one pool of `connections`.
    Each file goes to <dest>.<pid>.<call id>.part and is renamed when complete; failed parts are retried 10 times.
    progress(bytes), on_done((key, size, etag)). Returns (key, size, etag) rows in the order of `items`."""
    from concurrent.futures import ThreadPoolExecutor, FIRST_EXCEPTION, wait

    stop, lock = threading.Event(), threading.Lock()
    progress = progress or (lambda count: None)
    parts_left, partials, tasks, call_id = {}, {}, [], uuid.uuid4().hex[:8]
    for index, (s3, bucket, key, dest, size, etag) in enumerate(items):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partials[index] = dest.with_name(f"{dest.name}.{os.getpid()}.{call_id}.part")
        with open(partials[index], "wb") as handle:
            handle.truncate(size)
        ranges = [(first, min(first + PART_BYTES, size) - 1) for first in range(0, size, PART_BYTES)]
        parts_left[index] = len(ranges)
        tasks += [(index, first, last) for first, last in ranges]
        if not ranges:
            partials[index].replace(dest)
            on_done and on_done((key, size, etag))

    def part(index, first, last):
        s3, bucket, key, dest, size, etag = items[index]
        for attempt in range(10):
            if stop.is_set():
                return
            received = 0
            try:
                body = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={first}-{last}")["Body"]
                descriptor = os.open(partials[index], os.O_WRONLY)
                try:
                    for chunk in body.iter_chunks(1 << 20):
                        if stop.is_set():
                            return
                        os.pwrite(descriptor, chunk, first + received)
                        received += len(chunk)
                        progress(len(chunk))
                finally:
                    os.close(descriptor)
                if received != last - first + 1:
                    raise IOError(f"{key}: {received} of {last - first + 1} bytes")
                break
            except ClientError as error:
                progress(-received)
                code = error.response["Error"]["Code"]
                if attempt == 9 or not (
                    code in ("SlowDown", "RequestTimeout", "InternalError", "ServiceUnavailable")
                    or str(error.response["ResponseMetadata"].get("HTTPStatusCode", 0)).startswith("5")
                ):
                    raise
            except Exception:
                progress(-received)
                if attempt == 9 or stop.is_set():
                    raise
            if stop.wait(min(10 * 2**attempt, 300)):
                return
        with lock:
            parts_left[index] -= 1
            last = parts_left[index] == 0
        if last and not stop.is_set():  # a stopped call leaves its file unfinished
            partials[index].replace(Path(dest))
            on_done and on_done((key, size, etag))

    executor = ThreadPoolExecutor(connections)
    try:
        futures = [executor.submit(part, *task) for task in tasks]
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        for future in done:
            future.result()
    finally:  # Ctrl-C or a failed part: stop every connection at once
        stop.set()
        executor.shutdown(wait=True, cancel_futures=True)
    return [(key, size, etag) for _, _, key, _, size, etag in items]


def remove_stale_parts(root):
    """Delete the .part files of s3_get and s3_get_many left by stopped runs (<file>.<pid>.<id>.part, the process gone)."""
    for path in Path(root).rglob("*.part"):
        pid = path.name.rsplit(".", 3)[-3] if path.name.count(".") >= 3 else ""
        if pid.isdigit() and not _running(int(pid)):
            path.unlink(missing_ok=True)


def _running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def s3_dirs(s3, bucket, prefix):
    """Names of the immediate sub-folders under `prefix` (all pages of the listing)."""
    names, token = [], None
    while True:
        request = dict(Bucket=bucket, Prefix=prefix, Delimiter="/")
        if token:
            request["ContinuationToken"] = token
        response = s3.list_objects_v2(**request)
        names += [entry["Prefix"][len(prefix) :].strip("/") for entry in response.get("CommonPrefixes", [])]
        token = response.get("NextContinuationToken")
        if not token:
            return names


def s3_exists(s3, bucket, key):
    """False only when the object is missing; wrong keys or a network error raise."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_size(s3, bucket, key, dest):
    """(bytes of the object, bytes still to download, etag): 0 to download when a file of the same size is on disk."""
    header = s3.head_object(Bucket=bucket, Key=key)
    size = header["ContentLength"]
    dest = Path(dest)
    return size, 0 if dest.exists() and dest.stat().st_size == size else size, header["ETag"].strip('"')


def _retry(function, what, tries=8):
    """function() retried after a time-out, a dropped connection or a server-side error (HTTP 5xx, 429), waiting 10 s,
    20 s, ... up to 5 min between tries. Other HTTP errors (404, 403) are raised at once."""
    for attempt in range(tries):
        try:
            return function()
        except (*RETRY, requests.HTTPError) as error:
            code = (
                error.response.status_code
                if isinstance(error, requests.HTTPError) and error.response is not None
                else None
            )
            if attempt == tries - 1 or (
                isinstance(error, requests.HTTPError) and not (code == 429 or (code or 0) >= 500)
            ):
                raise
            wait = min(10 * 2**attempt, 300)
            print(f"{what}: {type(error).__name__}, retrying in {wait} s ({attempt + 1}/{tries - 1})", flush=True)
            time.sleep(wait)


def _size(url, timeout):
    """Size of the file at `url` if the server serves byte ranges, else None."""
    with requests.get(url, headers={**PLAIN, "Range": "bytes=0-0"}, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_range = response.headers.get("Content-Range", "")
        return (
            int(content_range.rsplit("/", 1)[1])
            if response.status_code == 206 and content_range.rsplit("/", 1)[-1].isdigit()
            else None
        )


def _fetch(url, path, first, last, timeout, progress):
    """Bytes a..b (inclusive; b None = to the end) of `url` appended to file `f`, continuing what `f` already holds."""
    present = path.stat().st_size if path.exists() else 0
    if last is not None and present == last - first + 1:
        return
    if last is not None and present > last - first + 1:  # left by an older, broken run
        progress(-present)
        path.unlink()
        present = 0
    headers = (
        {"Range": f"bytes={first + present}-{'' if last is None else last}"} if present or last is not None else {}
    )
    with requests.get(url, headers={**PLAIN, **headers}, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if headers and response.status_code != 206:  # server ignored Range: start again
            present = 0
            progress(-path.stat().st_size if path.exists() else 0)
        with open(path, "ab" if present else "wb") as handle:
            for chunk in response.iter_content(1 << 20):
                handle.write(chunk)
                progress(len(chunk))
    if last is not None and path.stat().st_size != last - first + 1:
        raise requests.exceptions.ChunkedEncodingError(f"{path.name}: connection closed early")


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

        def update(count):
            with lock:
                bar.update(count)

        if size is None:
            partial = dest.with_name(dest.name + ".part")
            if partial.exists():
                update(partial.stat().st_size)
            _retry(lambda: _fetch(url, partial, 0, None, timeout, update), dest.name, tries)
        else:
            parts_folder = dest.with_name(dest.name + ".parts")
            parts_folder.mkdir(exist_ok=True)
            ranges = [
                (parts_folder / f"{index:06d}", first, min(first + PART_BYTES, size) - 1)
                for index, first in enumerate(range(0, size, PART_BYTES))
            ]
            update(sum(part.stat().st_size for part, _, _ in ranges if part.exists()))
            with ThreadPoolExecutor(PARTS_AT_ONCE) as executor:
                list(
                    executor.map(
                        lambda part_range: _retry(lambda: _fetch(url, *part_range, timeout, update), dest.name, tries),
                        ranges,
                    )
                )
            partial = dest.with_name(dest.name + ".part")
            with open(partial, "wb") as handle:
                for part, _, _ in ranges:
                    with open(part, "rb") as source:
                        shutil.copyfileobj(source, handle, 1 << 24)
            partial.replace(dest)
            shutil.rmtree(parts_folder)  # parts deleted only once the file is in place
            return dest
    partial.replace(dest)
    return dest


def hcp_keys(sid, kind="3T", release="HCP_1200"):
    """S3 keys needed for one subject. kind: 3T | 7T (diffusion only; anatomy comes with 3T)."""
    base = f"{release}/{sid}/T1w"
    diffusion = "Diffusion" if kind == "3T" else "Diffusion_7T"
    keys = [f"{base}/{diffusion}/{name}" for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz")]
    if kind == "3T":
        keys += [f"{base}/aparc+aseg.nii.gz"]
        keys += [
            f"{base}/fsaverage_LR32k/{sid}.{hemi}.{surface}_MSMAll.32k_fs_LR.surf.gii"
            for hemi in "LR"
            for surface in ("white", "pial")
        ]
    return keys


def hcp_figure_keys(sid, release="HCP_1200"):
    """Very inflated surfaces of both hemispheres (same vertices and faces as the white surface); figures only."""
    return [
        f"{release}/{sid}/T1w/fsaverage_LR32k/{sid}.{hemi}.very_inflated_MSMAll.32k_fs_LR.surf.gii" for hemi in "LR"
    ]


def hcp_download(cfg, sid, kind="3T", release="HCP_1200", s3=None):
    s3 = s3 or hcp_client(cfg)
    return [s3_get(s3, HCP_BUCKET, key, cfg.P.hcp / key) for key in hcp_keys(sid, kind, release)]


def osf_download(cfg, names=tuple(OSF_FILES)):
    for name in names:
        http_get(OSF_FILES[name], cfg.P.osf / name)


def update_manifest(cfg, rows):
    """rows: (key, size, etag). Single-part S3 ETags are MD5 sums; multipart ones are recorded as-is."""
    path = cfg.P.data / "manifest.json"
    manifest = load_json(path) if path.exists() else {}
    for key, size, etag in rows:
        manifest[key] = {"bytes": size, "etag": etag}
    save_json(manifest, path)
    return manifest
