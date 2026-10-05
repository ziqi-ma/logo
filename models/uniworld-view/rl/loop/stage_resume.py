"""Stage the newest COMPLETE adapter/trainstate pair from an object-store prefix.

Pulls only the latest usable pair (~0.8 GB) instead of the whole prefix, which at
200 checkpoints is ~33 GB, and verifies both files structurally before trusting them.

    python -m rl.loop.stage_resume <uri-prefix> <dest-dir>
    python -m rl.loop.stage_resume s3://bucket/runs/myrun/adapters /outputs/adapters

Two failures this exists to prevent, both observed in production:

  * A whole-prefix download with `|| true` swallowed every error, so a resume
    silently fell back to an older step (12 steps lost once, then 6).
  * Object-store LIST is eventually consistent on some backends: a pod listed
    adapters<=34 / trainstates<=35 while both existed through 41. HEAD is strongly
    consistent, so this probes upward past whatever the listing claims.

A mid-write adapter generates silently-wrong video rather than failing, so both
files are structurally verified: the safetensors payload end must equal the file
size, and the trainstate must carry the PK magic plus a zip end-of-central-directory
record. Failures raise instead of being swallowed.
"""
import argparse

def _args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("uri", help="object-store prefix holding nft_new_step*.safetensors")
    ap.add_argument("dest", help="local directory to stage into")
    a = ap.parse_args()
    return a.uri, a.dest

import os, re, sys
from rl.data import gcs_util as g
uri, dest = _args()
os.makedirs(dest, exist_ok=True)
scheme, bucket, prefix = g._split(uri)
prefix = (prefix.rstrip("/") + "/") if prefix else ""
keys = set(g._list_keys(bucket, prefix))
ad = {int(m.group(1)) for k in keys if (m := re.search(r"nft_new_step([0-9]+)[.]safetensors$", k))}
ts = {int(m.group(1)) for k in keys if (m := re.search(r"nft_trainstate_step([0-9]+)[.]pt$", k))}
# the object store LIST is EVENTUALLY CONSISTENT: a pod can list this prefix and get
# adapters<=34 / trainstates<=35 while both actually existed through 41, so the resume
# silently gave up 6 steps. HEAD is strongly consistent, so probe upward past whatever the
# listing claims and trust the probe. Cheap: a couple of HEADs per candidate step.
def _exists(name):
    return g.exists(f"{scheme}://{bucket}/{prefix}{name}")
_probe_from = (max(ad | ts) if (ad or ts) else 0) + 1
_found = []
for _c in range(_probe_from, _probe_from + 40):        # 40 steps of slack past the listing
    if _exists(f"nft_new_step{_c:04d}.safetensors") and _exists(f"nft_trainstate_step{_c:04d}.pt"):
        _found.append(_c); ad.add(_c); ts.add(_c)
    elif _c > _probe_from + 2:                          # allow a small gap, then stop
        break
if _found:
    print(f"[resume-stage] LIST was stale: HEAD found additional complete pairs {_found}", flush=True)

paired = sorted(ad & ts)
if not paired:
    print(f"[resume-stage] no complete pair among {len(ad)} adapters / {len(ts)} trainstates"
          f" -- starting from scratch", flush=True); sys.exit(0)
n = paired[-1]
print(f"[resume-stage] newest complete pair = step{n:04d} "
      f"(adapters up to {max(ad) if ad else '-'}, trainstates up to {max(ts) if ts else '-'})",
      flush=True)
want = [f"nft_new_step{n:04d}.safetensors", f"nft_trainstate_step{n:04d}.pt"]
meta = f"nft_new_step{n:04d}.meta.json"
if prefix + meta in keys: want.append(meta)
for name in want:
    g.download_file(f"{scheme}://{bucket}/{prefix}{name}", os.path.join(dest, name))
# Structural verification, not just size: a short adapter still has a parseable header
# whose payload-end will not equal the file size, and a truncated torch save loses its
# zip end-of-central-directory record. Both are cheap and catch a partial download.
import json as _j, struct as _st
ap = os.path.join(dest, want[0]); tp = os.path.join(dest, want[1])
with open(ap, "rb") as f:
    hn = _st.unpack("<Q", f.read(8))[0]
    hdr = _j.loads(f.read(hn))
    tens = {k: v for k, v in hdr.items() if k != "__metadata__"}
    endoff = 8 + hn + max(v["data_offsets"][1] for v in tens.values())
asz = os.path.getsize(ap)
if endoff != asz:
    sys.exit(f"[resume-stage] FATAL adapter truncated: payload ends {endoff}, file is {asz}")
tsz = os.path.getsize(tp)
with open(tp, "rb") as f:
    if f.read(2) != b"PK":
        sys.exit("[resume-stage] FATAL trainstate is not a zip (torch save incomplete)")
    f.seek(max(0, tsz - 128)); tail = f.read()
if bytes([0x50, 0x4b, 0x05, 0x06]) not in tail:
    sys.exit("[resume-stage] FATAL trainstate missing zip EOCD (truncated)")
print(f"[resume-stage] verified step{n:04d}: adapter {asz}B ({len(tens)} tensors, "
      f"payload==size), trainstate {tsz}B (zip intact)", flush=True)
