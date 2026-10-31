"""Build ntl-atlas2-check: exp019's second ATLAS bundle (harshitakumari256/ntl-atlas-bundle, Qwen3-4B-Base) through the
exact worker cell and parent function a variant would run, on Kaggle's T4 x2 in nlm-v27's Docker image.

    python kaggle/nlm/build_atlas2_check.py

On 400 train rows (the 48 longest + 352 random) it reports:
  - load time and seconds per text on T4 x2 (one worker process per GPU), for the hidden-run budget;
  - whether the bundle reproduces the bundle's own published out-of-fold scores (oof_all.parquet, shipped with it): if
    its mean log-likelihoods match, the bundle never trained on those rows and exp015's replica gain (+0.0074) is an
    out-of-fold estimate for this very model; if not, it was trained on them too (fold models made the OOF file);
  - a second pass with a deadline ~60 s after loading: only rows scored before it come back, equal to the first pass.
Output: kaggle/nlm/ntl-atlas2-check/ (notebook + kernel-metadata.json).
"""
import json

import build_nlm as B

KERNEL = "ntl-atlas2-check"
CHECK = r'''import glob, os, sys, json, time, types
import numpy as np, pandas as pd, torch
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
B4 = next(os.path.dirname(x) for x in cfgs if '/qwen-3/' in x.lower() and '/4b-base/' in x.lower())
ATLAS2 = [os.path.dirname(x) for x in glob.glob('/kaggle/input/**/atlas_config.json', recursive=True) if '/ntl-atlas-bundle/' in x]
assert len(ATLAS2) == 1, ATLAS2; ATLAS2 = ATLAS2[0]
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
print(B4, ATLAS2, sorted(os.listdir(ATLAS2)), torch.__version__, torch.cuda.device_count(), flush=True)
LABELS = ['claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash','glm-5.3','glm-5.3-flash',
 'gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra','kimi-k3','qwen3.8-max']
train = pd.read_csv(DATA + '/train.csv')
rng = np.random.RandomState(0); lens = train.text.str.len().values
longest = np.argsort(-lens, kind='stable')[:48]
sel = np.concatenate([longest, rng.choice(np.setdiff1d(np.arange(len(train)), longest), 352, replace=False)])
test = train.iloc[sel].reset_index(drop=True)
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)
a = types.SimpleNamespace(atlas2_dir=ATLAS2, atlas2_base=B4, cache='/kaggle/working/cache')
os.makedirs(a.cache, exist_ok=True)
ns = dict(np=np, pd=pd, torch=torch, os=os, sys=sys, json=json, time=time, a=a, test=test, T0=T0, LABELS=LABELS,
          log=log, __file__='/kaggle/working/infer.py')
exec(compile(ATLAS2_FN, 'atlas2_scores', 'exec'), ns)
cfg = json.load(open(os.path.join(ATLAS2, 'atlas_config.json')))

t = time.time(); pos, A = ns['atlas2_scores'](np.arange(len(test)), 1e9); full_s = time.time() - t
assert np.array_equal(pos, np.arange(len(test))), pos
z = [np.load(f'{a.cache}/atlas2_{k}.npz') for k in range(torch.cuda.device_count())]
rows = np.concatenate([q['rows'] for q in z]); LL = np.concatenate([q['LL'] for q in z])[np.argsort(rows)]
oof = pd.read_parquet(os.path.join(ATLAS2, 'oof_all.parquet')).set_index('id').loc[test.id.astype(str)]
LLo = oof[[f'LL_{l}' for l in cfg['labels']]].to_numpy(np.float64)
y = test.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
acc = lambda P: float((P.argmax(1) == y).mean())
Ao = np.exp(cfg['tau'] * (LLo - LLo.max(1, keepdims=True))); Ao /= Ao.sum(1, keepdims=True)
Ao = Ao[:, [cfg['labels'].index(l) for l in LABELS]]
R = dict(rows=len(test), seconds_total=round(full_s, 1),
         ll_maxdiff_vs_oof=float(np.abs(LL - LLo).max()), ll_meandiff_vs_oof=float(np.abs(LL - LLo).mean()),
         argmax_agreement_vs_oof=float((A.argmax(1) == Ao.argmax(1)).mean()),
         accuracy_bundle=acc(A), accuracy_oof=acc(Ao), accuracy_bundle_random352=acc(A[48:]), accuracy_oof_random352=acc(Ao[48:]))
log('full pass', json.dumps(R))

# deadline 40% of the full pass from now: the workers reload (from the page cache now), then score until it
t = time.time()
try:
    pos2, A2 = ns['atlas2_scores'](np.arange(len(test)), time.time() - T0 + 0.4 * full_s)
except Exception as e:   # no rows scored before the deadline: the variant logs ATLAS2 SKIPPED and keeps ATLAS-8B's mix
    pos2, A2 = np.zeros(0, int), np.zeros((0, len(LABELS))); R['deadline_error'] = repr(e)
R['deadline_pass_seconds'] = round(time.time() - t, 1)
R['deadline_rows'] = int(len(pos2)); R['deadline_maxdiff_vs_full'] = float(np.abs(A2 - A[pos2]).max()) if len(pos2) else None
ok = R['deadline_rows'] < len(test) and (R['deadline_maxdiff_vs_full'] or 0) < 1e-3 and R['argmax_agreement_vs_oof'] > 0 and R['accuracy_bundle'] > 0.3
print('ATLAS2 CHECK DONE', 'PASS' if ok else 'FAIL', json.dumps(R), flush=True)
json.dump(R, open('/kaggle/working/atlas2_check.json', 'w'), indent=1)
'''


def main():
    cells = [("markdown", f"# {KERNEL}\n\nexp019: the second ATLAS bundle (harshitakumari256/ntl-atlas-bundle, Qwen3-4B-Base) "
                          "through the worker cell and parent function a variant would run, on 400 train rows: speed on "
                          "T4 x2, agreement with the bundle's published out-of-fold scores, and the deadline stop. "
                          "Private check; no submission."),
             ("code", B.ATLAS2_WORKER), ("code", f"ATLAS2_FN = {B.ATLAS2_FN!r}\n"), ("code", CHECK)]
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
            "dataset_sources": [B.ATLAS2_DATASET], "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "docker_image": meta_in["docker_image"], "machine_shape": meta_in["machine_shape"], "is_private": True,
            "keywords": [], "model_sources": ["qwen-lm/qwen-3/Transformers/4b-base/1"]}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    compile(CHECK, KERNEL, "exec")
    compile(B.ATLAS2_WORKER.split("\n", 1)[1], "atlas2_worker.py", "exec")
    compile(B.ATLAS2_FN, "atlas2_scores", "exec")
    print(f"built {KERNEL}")


if __name__ == "__main__":
    main()
