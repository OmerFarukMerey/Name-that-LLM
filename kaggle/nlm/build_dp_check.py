"""Build ntl-l24-dp-check: exp018's data-parallel l24_predict against upstream's on Kaggle's T4 x2, before any
submission uses it.

    python kaggle/nlm/build_dp_check.py

For the three members that get "dp" (mn8v1, ll8v5, q8v5) and for q14v5 (no "dp": pooling on the GPU over upstream's
split copy) it runs two adapters each (a fold adapter and the full fit) over 160-400 train rows, the 48 longest among
them (peak memory), once with upstream's l24_predict and once with the patched one, in upstream's Docker image, with
the adapter files read once beforehand so both runs read them warm. It reports the time between the two adapters' log
lines (one adapter's merge and pass), peak memory per GPU, whether the copy per GPU fell back, and the largest
differences in plain / centred log-probabilities and in the Gaussian head's train features (q8v5, q14v5).
Output: kaggle/nlm/ntl-l24-dp-check/ (notebook + kernel-metadata.json).
"""
import hashlib
import json
from pathlib import Path

import build_nlm as B

KERNEL = "ntl-l24-dp-check"
SETUP = r'''import glob, os, json, shutil, time, gc
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]
    assert c, parts; return c[0]
def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True); assert h, name; return os.path.dirname(h[0])   # as nlm-v27's cell 0
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
B8, BL, B_MN8V1 = mdir('/qwen-3/', '/8b-base/'), mdir('/llama-3.1/', '/8b/'), mdir('/ministral-3/', '/ministral-3-8b-base-2512/')
B14 = mdir('/qwen-3/', '/14b-base/')
_c = time.time(); shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'
print('llama copied to local disk in', round(time.time() - _c), 's')
AQ8V5 = [adir(f'q8v5_f{k}') for k in range(5)] + [adir('q8v5')]
ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]
A_MN8V1 = [adir(f'mn8v1_f{k}') for k in range(5)] + [adir('mn8v1')]
AQ14V5 = [adir(f'q14v5_f{k}') for k in range(5)] + [adir('q14v5')]
print(B8, BL, B_MN8V1, B14, AQ8V5, ALL8V5, A_MN8V1, AQ14V5, sep='\n')
for _d in [AQ8V5[0], AQ8V5[-1], ALL8V5[0], ALL8V5[-1], A_MN8V1[0], A_MN8V1[-1], AQ14V5[0], AQ14V5[-1]]:   # page cache, so
    for _r, _, _fs in os.walk(_d):                                                                       # both runs read
        for _f in _fs: open(os.path.join(_r, _f), 'rb').read()                                            # adapters warm
print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())])
'''
CHECK = r'''LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # as infer.py
T0 = time.time(); LOG = []
def log(*x):
    LOG.append((time.time(), ' '.join(map(str, x)))); print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)
def load(src):
    ns = dict(np=np, torch=torch, os=os, json=json, gc=gc, AutoTokenizer=AutoTokenizer, AutoModel=AutoModel, LABELS=LABELS, log=log)
    exec(compile(src, 'l24_predict', 'exec'), ns); return ns['l24_predict']
up, dp = load(UP_SRC), load(DP_SRC)
train = pd.read_csv(DATA + '/train.csv')
rng = np.random.RandomState(0); lens = train.text.str.len().values
longest = np.argsort(-lens, kind='stable')[:48]
sel = np.concatenate([longest, rng.choice(np.setdiff1d(np.arange(len(train)), longest), 352, replace=False)])
other = rng.choice(np.setdiff1d(np.arange(len(train)), sel), 160, replace=False)
groups = pd.factorize(train.prompt_id.values[sel])[0]
L = [2, 4, 6, 8, 12, 16, 20, 24]
MEMBERS = [
    ({"name": "mn8v1", "type": "l24", "path": B_MN8V1, "adapters": [A_MN8V1[0], A_MN8V1[-1]], "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35}, 400, False),
    ({"name": "ll8v5", "type": "l24", "path": BL, "adapters": [ALL8V5[0], ALL8V5[-1]], "keep": 24, "layers": L, "w": 1.0, "bs": 16, "max_len": 384, "twin": 0.35}, 400, False),
    ({"name": "q8v5", "type": "l24", "path": B8, "adapters": [AQ8V5[0], AQ8V5[-1]], "keep": 24, "layers": L, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44}, 240, True),
    ({"name": "q14v5", "type": "l24", "path": B14, "adapters": [AQ14V5[0], AQ14V5[-1]], "keep": 27, "layers": L, "w": 3.0, "bs": 8, "max_len": 384, "twin": 0.35, "gauss": 0.70}, 160, True),
]
DP = {"mn8v1", "ll8v5", "q8v5"}   # q14v5 does not fit one T4: it checks pooling on the GPU over a split copy
SUMMARY = {}
for spec, n, gauss in MEMBERS:
    out = {}
    for mode, fn, sp in (('split', up, spec), ('dp', dp, dict(spec, dp=spec['name'] in DP))):
        for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
        k0 = len(LOG); t0 = time.time()
        r = fn(sp, train.text.values[sel[:n]], groups[:n], train_texts=train.text.values[other] if gauss else None)
        done = [t for t, msg in LOG[k0:] if 'adapter' in msg and ' done (' in msg]
        assert len(done) == 2, LOG[k0:]
        fell_back = any('one split copy' in msg for _, msg in LOG[k0:])
        peak = [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())]
        out[mode] = dict(r=r, total=round(time.time() - t0, 1), second_adapter=round(done[1] - done[0], 1), peak_gib=peak, fell_back=fell_back)
        log(f"{spec['name']} {mode}: {out[mode]['total']}s in all, second adapter {out[mode]['second_adapter']}s, peak GiB {peak}, fell back {fell_back}")
        del r; gc.collect(); torch.cuda.empty_cache()
    a, b = out['split']['r'], out['dp']['r']
    s = dict(rows=n, speedup=round(out['split']['second_adapter'] / out['dp']['second_adapter'], 2),
             **{f'{m}_{k}': out[m][k] for m in ('split', 'dp') for k in ('total', 'second_adapter', 'peak_gib', 'fell_back')},
             plain_maxdiff=float(np.abs(a[0] - b[0]).max()), centred_maxdiff=float(np.abs(a[1] - b[1]).max()),
             argmax_agreement=float((a[0].argmax(1) == b[0].argmax(1)).mean()))
    if gauss:
        s['feats_maxdiff'] = [float(np.abs(x - y).max()) for x, y in zip(a[2], b[2])]
    SUMMARY[spec['name']] = s; print(json.dumps(s), flush=True)
json.dump(SUMMARY, open('/kaggle/working/dp_check.json', 'w'), indent=1)
ok = all(not v['dp_fell_back'] and v['argmax_agreement'] >= 0.995 and v['plain_maxdiff'] < 0.05 for v in SUMMARY.values())
print('DP CHECK DONE', 'PASS' if ok else 'FAIL', json.dumps(SUMMARY), flush=True)
'''


def function_sources():
    """upstream's l24_predict and the exp018 patched one, cut from infer.py (notebook cell 2)."""
    raw = (B.HERE / B.BASE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == B.BASE_SHA256
    nb = json.loads(raw)
    code = [B.text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    infer = next(c for c in code if c.startswith("%%writefile infer.py"))
    patched = infer
    for old, new in B.L24_DP_FN:
        assert patched.count(old) == 1, old
        patched = patched.replace(old, new)

    def cut(s):
        i, j = s.index("@torch.no_grad()\ndef l24_predict"), s.index("\ndef gpu_mem")
        return s[i:j]
    return cut(infer), cut(patched)


def main():
    up, dp = function_sources()
    cells = [("markdown", f"# {KERNEL}\n\nexp018: the data-parallel `l24_predict` (a full copy of a 24-layer 8B member on each "
                          "T4, alternate batches in two threads) against upstream's (one copy split across both GPUs), "
                          "in nlm-v27's Docker image. Private check; no submission."),
             ("code", SETUP), ("code", f"UP_SRC = {up!r}\nDP_SRC = {dp!r}\n"), ("code", CHECK)]
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s, **({"execution_count": None, "outputs": []} if t == "code" else {})}
                    for t, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    meta_in = json.loads((B.HERE / "upstream_nlm_v27.meta.json").read_text(encoding="utf-8"))
    out = B.HERE / KERNEL
    out.mkdir(exist_ok=True)
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["dhruvtibarewal/nlm-l24-adapters-v5"],
            "kernel_sources": [B.LLAMA_CACHE, f"omerfarukmerey/{B.L24_MEMBERS['mn8v1'][2]}"],
            "competition_sources": ["name-that-llm"], "docker_image": meta_in["docker_image"],
            "machine_shape": meta_in["machine_shape"], "is_private": True, "keywords": [],
            "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1", "qwen-lm/qwen-3/Transformers/14b-base/1", B.L24_MEMBERS["mn8v1"][1]]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {KERNEL}: upstream l24_predict {len(up)} chars, patched {len(dp)} chars")


if __name__ == "__main__":
    main()
