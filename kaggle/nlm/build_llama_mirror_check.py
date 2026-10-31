"""Build ntl-llama-mirror-check: is alexchilton's public Kaggle model alexchilton/llama-3-1-8b-base byte-identical to the
Llama-3.1-8B base our forks loaded from avazaku's (now deleted) Verified Llama Asset Cache?

    python kaggle/nlm/build_llama_mirror_check.py

avazaku's cache was deleted on 2026-10-29, so new commits of the nlm-v27 / V31 forks lose ll8v5 (the 0.651 recipe
without Llama scored 0.643). alexchilton's 0.660 notebook now reads Llama from his public model instead. The cache's
PROTOCOL recorded the files of NousResearch/Meta-Llama-3.1-8B (revision 1f47e50, matching meta-llama/Llama-3.1-8B
revision d04e592): the sha256 of the four weight shards and the git blob sha1 of every file. This CPU kernel hashes
the mirror's files against them (no model import or execution). Shard sizes alone cannot tell the base model from
Instruct, whose shards have the same sizes. For the shards the recorded git blob sha1 is that of the Git LFS pointer
file, not of the weights, so only their sha256 is compared (version 1 compared both and reported FAIL on all four
shards although each sha256 matched; 2026-10-29 12:15 UTC: every file matches, so the mirror is byte-identical).
Output: kaggle/nlm/ntl-llama-mirror-check/ (notebook + kernel-metadata.json).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = "ntl-llama-mirror-check"
MIRROR = "alexchilton/llama-3-1-8b-base/Transformers/default/1"
# path: (bytes, sha256, git blob sha1), from avazaku's PROTOCOL (checked_at_utc 2026-10-25T22:05:09Z)
EXPECTED = {
    "LICENSE": (None, None, "a7c3ca16cee30425ed6ad841a809590f2bcbf290"),
    "USE_POLICY.md": (None, None, "81ebb55902285e8dd5804ccf423d17ffb2a622ee"),
    "config.json": (None, None, "cccf055d6f8f210387a248c91dc40e0c7a4bafab"),
    "generation_config.json": (None, None, "fc9506438b7b55383dc04c0816561442324846c3"),
    "model-00001-of-00004.safetensors": (4976698672, "f8b9704ab09cdeb097aa4a0a24bca96f906eec36bad63ab495bc21475058601b",
                                         "a70cf544ea82b645a9084d345203e3b855c474af"),
    "model-00002-of-00004.safetensors": (4999802720, "c28b25e7541751056ee126627e007f8d4288319733285e9f7b17b9ff6eb313f0",
                                         "440f54a28d3e777128cca95168dd93017e99d8ef"),
    "model-00003-of-00004.safetensors": (4915916176, "d8e9504dd4e4a146d484c52a97584ec14dac92237c46b064934af67a85e7d383",
                                         "b42588cf2677760fc6f6c39b19e55a31fca75e1c"),
    "model-00004-of-00004.safetensors": (1168138808, "e4486f35c040f683f7d790354f66c169c109eb9fa0954a4a35d7c458a108405d",
                                         "e614ce32dfec70990b7029fb392a318df1622054"),
    "model.safetensors.index.json": (None, None, "0fd8120f1c6acddc268ebc2583058efaf699a771"),
    "special_tokens_map.json": (None, None, "d8cd5076496dbe4be2320312abc10adc43097b81"),
    "tokenizer.json": (None, None, "f916e71031fa08f3c6ef1680a590c15b52d3cdd9"),
    "tokenizer_config.json": (None, None, "cb9ec25536e44d86778b10509d3e5bdca459a5cf"),
}
CHECK = r'''# Only file hashing: no model import or execution.
import glob, hashlib, json, os, time
EXPECTED = %r
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
BL = [os.path.dirname(x) for x in cfgs if '/llama-3-1-8b-base/' in x.lower()]
print(BL, flush=True); assert len(BL) == 1, cfgs; BL = BL[0]
print(sorted(os.listdir(BL)), flush=True)
res = {}
for f, (n, sha, blob) in EXPECTED.items():
    p = os.path.join(BL, f)
    if not os.path.exists(p):
        res[f] = dict(ok=False, missing=True); print(f, 'MISSING', flush=True); continue
    size = os.path.getsize(p); t = time.time()
    h = hashlib.sha256(); g = hashlib.sha1(b'blob %%d\0' %% size)
    with open(p, 'rb') as fh:
        while True:
            b = fh.read(16 << 20)
            if not b: break
            h.update(b); g.update(b)
    # an LFS file's recorded blob sha1 is its pointer's, so a file with a sha256 is checked by that alone
    ok = (n is None or size == n) and (h.hexdigest() == sha if sha else g.hexdigest() == blob)
    res[f] = dict(bytes=size, sha256=h.hexdigest(), git_blob_sha1=g.hexdigest(), ok=ok, seconds=round(time.time() - t, 1))
    print(f, res[f], flush=True)
verdict = all(r['ok'] for r in res.values())
print('VERDICT', 'PASS' if verdict else 'FAIL', flush=True)
json.dump(dict(dir=BL, files=res, verdict=verdict), open('/kaggle/working/llama_mirror_check.json', 'w'), indent=2)
''' % (EXPECTED,)


def main():
    out = HERE / KERNEL
    out.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                     "source": CHECK.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "is_private": True, "keywords": [],
            "model_sources": [MIRROR]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
