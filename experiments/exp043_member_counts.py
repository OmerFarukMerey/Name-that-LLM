"""exp043: per member, out of fold on K's rows: predicted count per class (150 true each) and recall, plus the blend
before and after the balancing step.

    python -I experiments/exp043_member_counts.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1])
sys.path[:0] = [ROOT + "/src", ROOT + "/experiments"]
sys.argv.append("--plain")
import numpy as np, pandas as pd
import exp015_nlm_core_oof as E
import exp034_matched_extra as X
import exp036_svc_matched as M
from exp029_atlas_ext import soft_sinkhorn
from ntl.data import load_train

train = load_train()
nlm = np.load(E.NLM_OOF)
labels = np.array(E.NLM_LABELS)
y = train.model.map({l: i for i, l in enumerate(E.NLM_LABELS)}).to_numpy()
mem = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
z = np.load(E.ENC_DIR / "deberta_v3_large_pool" / "oof.npz")
mem["deb"] = E.lsm(z["logits"][:, [list(z["labels"]).index(l) for l in E.NLM_LABELS]].astype(np.float64))
mem["svc4"] = np.load(M.OUT / "m2_svc4.npy").astype(np.float64)
mem["distort"] = np.load(X.OUT / "m2_distort.npy")
mem["tfidf"] = np.load(X.OUT / "m2_tfidf.npy")
sty = np.load(f"{ROOT}/outputs/exp035_comp_extra/ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30_stylegram.npy")
mem["stylegram(cx)"] = sty
E.TERMS.update({"tfidf": 1, "distort": 1})
P = E.blend(mem, {**E.CORE, "svc4": 12.0, "deb": 1.0, "tfidf": 0.25, "distort": 0.10})
mem["blend"] = np.log(P)
mem["blend+sk30"] = soft_sinkhorn(P, alpha=0.30)
short = [l.replace("claude-", "c-").replace("gemini-", "g-").replace("gpt-", "").replace("-preview", "")[:11] for l in labels]
cnt = pd.DataFrame({k: np.bincount(v.argmax(1), minlength=18) for k, v in mem.items()}, index=short)
rec = pd.DataFrame({k: pd.Series(v.argmax(1) == y).groupby(y).mean().values for k, v in mem.items()}, index=short)
pd.set_option("display.width", 250)
print("predicted counts (true ~150 each):"); print(cnt.to_string())
print("\nrecall:"); print(rec.round(2).to_string())
