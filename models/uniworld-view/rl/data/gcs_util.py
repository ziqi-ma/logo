# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object-store staging helper for the RL loop.

Stages the rollout store / checkpoints / adapters / scene inputs. Supports:

  hf://      a file in a HuggingFace repo (public model weights)
  s3://      S3 or any S3-compatible store (boto3, ambient AWS_* creds;
             set S3_ENDPOINT_URL for a non-AWS endpoint)

    python -m <module> download_prefix s3://bucket/prefix/ /dest
    python -m <module> download_file   s3://bucket/prefix/f /dest/f
    python -m <module> upload_file     /local/f  s3://bucket/prefix/f
    python -m <module> upload_prefix   /local/d  s3://bucket/prefix/
"""

import os
import shutil
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or None

def _is_local(uri: str) -> bool:
    """A plain filesystem path, i.e. no ``scheme://``. Local paths let the eval run
    entirely off a HuggingFace checkout with no object store."""
    return "://" not in uri


def _split(uri: str):
    scheme, _, rest = uri.partition("://")
    if scheme != "s3":
        raise ValueError(f"only s3:// URIs are supported, got {uri!r}")
    bucket, _, prefix = rest.partition("/")
    return scheme, bucket, prefix

def _s3_client():
    import boto3

    # boto3>=1.36 defaults to an aws-chunked request checksum. S3-compatible stores
    # (some S3-compatible stores) mishandle it, and on a retry boto3 must rewind the wrapped stream
    # -> UnseekableStreamError -> upload fails (silently, for best-effort callers).
    # Force checksums to "when_required" so uploads use a plain seekable body.
    cfg = None
    try:
        from botocore.config import Config
        cfg = Config(request_checksum_calculation="when_required",
                     response_checksum_validation="when_required")
    except Exception:  # noqa: BLE001 -- older botocore without these knobs; env fallback below
        os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
        os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

    # s3:// -> ambient AWS_* creds; S3_ENDPOINT_URL selects a non-AWS endpoint.
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, config=cfg)


def _hf_download(uri: str, dest: str) -> None:
    """``hf://<repo_id>/<path/in/repo>`` -> a local file, via huggingface_hub.

    Public model weights are fetched from their own release rather than mirrored, so the
    only things this repo keeps on S3 are the ones that are not published elsewhere.
    """
    from huggingface_hub import hf_hub_download

    rest = uri[len("hf://"):]
    parts = rest.split("/")
    if len(parts) < 3:
        raise ValueError(f"hf:// needs <org>/<repo>/<path>, got {uri!r}")
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    src = hf_hub_download(repo_id=repo_id, filename=filename,
                          revision=os.environ.get("HF_REVISION") or None)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    shutil.copyfile(src, dest)

def download_file(uri: str, dest: str) -> None:
    if uri.startswith("hf://"):
        return _hf_download(uri, dest)
    if _is_local(uri):
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        return shutil.copyfile(uri, dest)
    scheme, bucket, key = _split(uri)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    _s3_client().download_file(bucket, key, dest)

def upload_file(local: str, uri: str, attempts: int = 4) -> None:
    if _is_local(uri):
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        return shutil.copyfile(local, uri)
    """Upload with retries. Some S3-compatible stores intermittently lose a multipart session and
    fail CompleteMultipartUpload with NoSuchUpload ("the upload may have been aborted"); boto's
    own `retries` config does not cover it, because the whole multipart sequence has to restart.
    One such hiccup on a single clip takes down a whole gang eval, since
    the rank exited non-zero and fate sharing killed the healthy pods with it."""
    scheme, bucket, key = _split(uri)
    last = None
    for i in range(max(1, attempts)):
        try:
            _s3_client().upload_file(local, bucket, key)
            return
        except Exception as e:  # noqa: BLE001 - retry the whole multipart sequence
            last = e
            if i + 1 < attempts:
                time.sleep(2 ** i)
                print(f"[gcs_util] upload retry {i+1}/{attempts-1} for {uri}: "
                      f"{type(e).__name__}", flush=True)
    raise last

def exists(uri: str) -> bool:
    if _is_local(uri):
        return os.path.isfile(uri)
    """HEAD a single object. Never use a LIST to answer this: on some S3-compatible stores LIST on fresh
    prefixes intermittently returns empty while GET/HEAD are correct."""
    scheme, bucket, key = _split(uri)
    try:
        _s3_client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001 - absent, or transient; caller re-does the work
        return False

def delete_file(uri: str) -> None:
    """Delete a single object at ``scheme://bucket/key``. Best-effort (no error if absent)."""
    scheme, bucket, key = _split(uri)
    _s3_client().delete_object(Bucket=bucket, Key=key)

def list_keys(uri: str):
    if _is_local(uri):
        root = uri.rstrip("/")
        out = []
        for d, _sub, files in os.walk(root):
            out += [os.path.join(d, f) for f in files]
        return out
    """List object keys under a ``scheme://bucket/prefix`` URI (no dir markers)."""
    scheme, bucket, prefix = _split(uri)
    return _list_keys(bucket, prefix)

def _list_keys(bucket: str, prefix: str):
    client = _s3_client()
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", []) if not o["Key"].endswith("/")]
        if not resp.get("IsTruncated"):
            return keys
        token = resp["NextContinuationToken"]

def _list_sizes(bucket: str, prefix: str):
    """{key: size} for a prefix. Used only to SKIP re-uploading unchanged objects, so a
    stale/incomplete listing is safe in one direction: a missing entry re-uploads."""
    client = _s3_client()
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            if not o["Key"].endswith("/"):
                out[o["Key"]] = o["Size"]
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]

def download_prefix(uri: str, dest: str) -> None:
    scheme, bucket, prefix = _split(uri)
    prefix = (prefix.rstrip("/") + "/") if prefix else ""
    keys = _list_keys(bucket, prefix)

    def _dl(key):
        p = os.path.join(dest, key[len(prefix):])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        download_file(f"{scheme}://{bucket}/{key}", p)

    # 16 was hardcoded; a training pod stages 205 GB (147.7 model + 57.3 cond) and the tail is
    # set by a handful of ~10 GB objects, so concurrency directly shortens the 46-min startup
    # Override with GCS_DL_WORKERS.
    nw = int(os.environ.get("GCS_DL_WORKERS", "32"))
    with ThreadPoolExecutor(max_workers=nw) as ex:
        list(ex.map(_dl, keys))
    print(f"download_prefix: {len(keys)} objs ({nw} workers) {uri} -> {dest}", flush=True)

def upload_prefix(local_dir: str, uri: str) -> None:
    scheme, bucket, prefix = _split(uri)
    prefix = (prefix.rstrip("/") + "/") if prefix else ""
    files = [os.path.join(r, f) for r, _, fs in os.walk(local_dir) for f in fs]

    # INCREMENTAL. This used to re-upload the whole directory every cycle. The mirror runs
    # every 5 min and a training pod accumulates one 204MB adapter + one 614MB trainstate per
    # step, so by step 150 a cycle was re-pushing ~92GB and could not finish inside its own
    # interval -- the newest adapter sits LAST (trainstates go first, see below), so fresh
    # checkpoints showed up one or two cycles late and the lag grew without bound: ~650GB/cycle
    # by step 800: a step's objects can still be absent remotely while
    # the pod had already written them locally.
    #
    # Skip only when the remote object exists AND matches byte-for-byte in size. The listing is
    # eventually consistent, but that is safe HERE: an entry missing from a stale listing simply
    # gets re-uploaded. As extra insurance the newest _KEEP_FRESH steps are always re-uploaded,
    # so the commit-marker invariant cannot be broken by a bad listing.
    _KEEP_FRESH = 2
    _steps = []
    for _p in files:
        _m = re.search(r"step([0-9]+)", os.path.basename(_p))
        if _m:
            _steps.append(int(_m.group(1)))
    _always = set(sorted(set(_steps), reverse=True)[:_KEEP_FRESH])
    try:
        _remote = _list_sizes(bucket, prefix)
    except Exception:                       # listing unavailable -> fall back to uploading all
        _remote = {}

    def _unchanged(p):
        _m = re.search(r"step([0-9]+)", os.path.basename(p))
        if _m and int(_m.group(1)) in _always:
            return False
        rel = os.path.relpath(p, local_dir)
        try:
            return _remote.get(prefix + rel) == os.path.getsize(p)
        except OSError:
            return False

    _skipped = [p for p in files if _unchanged(p)]
    files = [p for p in files if p not in set(_skipped)]

    def _ul(p):
        upload_file(p, f"{scheme}://{bucket}/{prefix}{os.path.relpath(p, local_dir)}")

    # ORDER MATTERS: trainstates before adapters, so the adapter is the COMMIT MARKER for a step.
    # This mirror is periodic and non-atomic; killed mid-upload it used to leave the small adapter
    # (204MB) present and the large trainstate (614MB) missing, and try_resume would then pick that
    # adapter and silently resume with a cold optimizer (see rl/nft_loop.py try_resume). Uploading
    # the trainstate first makes the torn state harmless: if adapter N is visible, trainstate N was
    # already uploaded, so "newest adapter" and "newest complete pair" can never disagree.
    ts = [p for p in files if "nft_trainstate_step" in os.path.basename(p)]
    rest = [p for p in files if p not in set(ts)]
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_ul, ts))
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_ul, rest))
    print(f"upload_prefix: {len(files)} files ({len(ts)} trainstate first), "
          f"{len(_skipped)} already-current skipped {local_dir} -> {uri}", flush=True)

def main() -> None:
    cmd = sys.argv[1]
    if cmd == "download_prefix":
        download_prefix(sys.argv[2], sys.argv[3])
    elif cmd == "download_file":
        download_file(sys.argv[2], sys.argv[3])
    elif cmd == "upload_file":
        upload_file(sys.argv[2], sys.argv[3])
    elif cmd == "upload_prefix":
        upload_prefix(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown cmd {cmd}")

if __name__ == "__main__":
    main()
