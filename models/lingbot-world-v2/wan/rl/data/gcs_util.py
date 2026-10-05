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
import time
import sys
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

def upload_file(local: str, uri: str) -> None:
    if _is_local(uri):
        os.makedirs(os.path.dirname(uri) or ".", exist_ok=True)
        return shutil.copyfile(local, uri)
    scheme, bucket, key = _split(uri)
    _s3_client().upload_file(local, bucket, key)

def object_size(uri: str):
    """Remote size in bytes, or None if the object is absent/unreadable."""
    if _is_local(uri):
        return os.path.getsize(uri) if os.path.isfile(uri) else None
    scheme, bucket, key = _split(uri)
    try:
        return int(_s3_client().head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception:  # noqa: BLE001 -- absent or transient; caller decides
        return None

def upload_file_verified(local: str, uri: str, attempts: int = 5) -> None:
    """upload_file + retry with backoff + read-back size check.

    Plain ``upload_file`` is a single shot. A transient object-store error therefore
    loses the object for good, and on lingbot that silently cost checkpoints 9, 15, 27
    and 42 of one 45-step run (ckpt 42's loss turned a 2-step resume into a 5-step one).
    S3-compatible stores fail here in a specific way: boto3>=1.36's aws-chunked request checksum
    makes an internal retry rewind a wrapped stream -> UnseekableStreamError, which the
    best-effort caller swallowed. Verifying the remote size catches a "succeeded" upload
    that did not actually land.

    Raises the last error if every attempt fails -- callers that want best-effort must
    catch it explicitly, so the failure can never be invisible.
    """
    want = os.path.getsize(local)
    last = None
    for i in range(max(1, attempts)):
        try:
            upload_file(local, uri)
            got = object_size(uri)
            if got == want:
                return
            last = RuntimeError(f"size mismatch after upload: local={want} remote={got}")
        except Exception as e:  # noqa: BLE001 -- retry transient store errors
            last = e
        if i + 1 < attempts:
            time.sleep(min(30, 3 * (i + 1)))
    raise RuntimeError(f"upload_file_verified failed for {uri} after {attempts} attempts: {last}")

def delete_file(uri: str) -> None:
    """Delete a single object at ``scheme://bucket/key``. Best-effort (no error if absent)."""
    scheme, bucket, key = _split(uri)
    _s3_client().delete_object(Bucket=bucket, Key=key)

def list_keys(uri: str):
    """List object keys under a ``scheme://bucket/prefix`` URI (no dir markers).
    For a local path, the paths under it, relative to the same root the S3 branch
    reports keys against."""
    if _is_local(uri):
        root = uri.rstrip("/")
        out = []
        for d, _sub, files in os.walk(root):
            out += [os.path.join(d, f) for f in files]
        return out
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

def download_prefix(uri: str, dest: str) -> None:
    scheme, bucket, prefix = _split(uri)
    prefix = (prefix.rstrip("/") + "/") if prefix else ""
    keys = _list_keys(bucket, prefix)

    def _dl(key):
        p = os.path.join(dest, key[len(prefix):])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        download_file(f"{scheme}://{bucket}/{key}", p)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_dl, keys))
    print(f"download_prefix: {len(keys)} objs {uri} -> {dest}", flush=True)

def upload_prefix(local_dir: str, uri: str) -> None:
    scheme, bucket, prefix = _split(uri)
    prefix = (prefix.rstrip("/") + "/") if prefix else ""
    files = [os.path.join(r, f) for r, _, fs in os.walk(local_dir) for f in fs]

    def _ul(p):
        upload_file(p, f"{scheme}://{bucket}/{prefix}{os.path.relpath(p, local_dir)}")

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_ul, files))
    print(f"upload_prefix: {len(files)} files {local_dir} -> {uri}", flush=True)

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
