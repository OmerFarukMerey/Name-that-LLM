"""Build ntl-atlasx-check: exp029's retrained ATLAS-8B (kernel ntl-atlas-full-ext) through the exact atlas_scores function
the -atlasx variants run, on Kaggle's T4 x2 in V31's Docker image, next to Joseph Adamski's bundle.

    python kaggle/nlm/build_atlasx_check.py

The -atlasx commits are fast commits (a placeholder on the 3-row test), so their ATLAS path first runs in the hidden
rerun, where a failure only logs ATLAS SKIPPED. On 72 random train rows (both bundles trained on them) it reports, per
bundle: rows scored, seconds, whether the scores are probabilities, accuracy, mean top score and top-2 margin.
Output: kaggle/nlm/ntl-atlasx-check/ (notebook + kernel-metadata.json).
"""
import json

import build_nlm as B

KERNEL = "ntl-atlasx-check"
VARIANT = "ntl-v31ac-a20-atlasx-p4"
CHECK = r'''import glob, os, sys, json, time, gc, types
import numpy as np, pandas as pd, torch
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
B8 = next(os.path.dirname(x) for x in cfgs if '/qwen-3/' in x.lower() and '/8b-base/' in x.lower())
bundles = glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True)
ATLAS = os.path.dirname(next(x for x in bundles if 'ntl-atlas-full-ext' in x))
JA = os.path.dirname(next(x for x in bundles if 'ntl-atlas-qwen3-8b-v1' in x))
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
print(B8, ATLAS, sorted(os.listdir(ATLAS)), JA, torch.__version__, torch.cuda.device_count(), sep='\n', flush=True)
print(json.load(open(os.path.join(ATLAS, 'atlas_bundle.json'))).get('exp029'), flush=True)
LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
train = pd.read_csv(DATA + '/train.csv')
test = train.iloc[np.random.RandomState(0).choice(len(train), 72, replace=False)].reset_index(drop=True)
y = test.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)
a = types.SimpleNamespace(atlas_base=B8, atlas_bs=4, atlas_chunk=3, atlas_probe=0)
ns = dict(np=np, pd=pd, torch=torch, os=os, sys=sys, json=json, time=time, gc=gc, a=a, test=test, T0=T0,
          LABELS=LABELS, log=log)
exec(compile(ATLAS_FN, 'atlas_scores', 'exec'), ns)
R = {}
for name, bd in (('full_ext', ATLAS), ('joseph', JA)):
    t = time.time(); A, scored = ns['atlas_scores'](bd, np.arange(len(test)), 1e9); s = np.sort(A, 1)
    R[name] = dict(seconds=round(time.time() - t, 1), scored=int(scored.sum()), finite=bool(np.isfinite(A).all()),
                   sums_ok=bool(np.allclose(A.sum(1), 1, atol=1e-3)), accuracy=float((A.argmax(1) == y).mean()),
                   mean_top=float(s[:, -1].mean()), mean_margin=float((s[:, -1] - s[:, -2]).mean()))
    np.save(f'/kaggle/working/atlas_{name}.npy', A)
    log(name, json.dumps(R[name]))
f = R['full_ext']
ok = f['scored'] == len(test) and f['finite'] and f['sums_ok'] and f['accuracy'] > 0.35
print('ATLASX CHECK DONE', 'PASS' if ok else 'FAIL', json.dumps(R), flush=True)
json.dump(R, open('/kaggle/working/atlasx_check.json', 'w'), indent=1)
'''


def atlas_fn():
    nb = json.loads((B.HERE / VARIANT / f"{VARIANT}.ipynb").read_text(encoding="utf-8"))
    src = next(B.text(c) for c in nb["cells"] if "def atlas_scores(" in B.text(c))
    i = src.index("def atlas_scores(")
    return src[i:src.index("\nCOMP_P = None", i)] + "\n"


def main():
    fn = atlas_fn()
    compile(fn, "atlas_scores", "exec"); compile(CHECK, KERNEL, "exec")
    cells = [("markdown", f"# {KERNEL}\n\nexp029: our retrained ATLAS-8B (ntl-atlas-full-ext) through the atlas_scores "
                          f"function of {VARIANT}, next to Joseph Adamski's bundle, on 72 train rows: rows scored, speed "
                          "on T4 x2, valid probabilities, accuracy and flatness. Private check; no submission."),
             ("code", f"ATLAS_FN = {fn!r}\n"), ("code", CHECK)]
    nb = {"cells": [{"cell_type": t, "metadata": {}, "source": s, **({"execution_count": None, "outputs": []} if t == "code" else {})}
                    for t, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    vmeta = json.loads((B.HERE / VARIANT / "kernel-metadata.json").read_text(encoding="utf-8"))
    out = B.HERE / KERNEL
    out.mkdir(exist_ok=True)
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["josephadamski91/ntl-atlas-qwen3-8b-v1"], "kernel_sources": [B.ATLAS_KERNEL],
            "competition_sources": ["name-that-llm"], "docker_image": vmeta["docker_image"],
            "machine_shape": vmeta["machine_shape"], "is_private": True, "keywords": [],
            "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {KERNEL} ({len(fn)} chars of atlas_scores from {VARIANT})")


if __name__ == "__main__":
    main()
