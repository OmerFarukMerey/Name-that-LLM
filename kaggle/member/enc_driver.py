# ---- driver cell (inserted by kaggle/member/build_member.py, which fills CONFIG)
# Two-stage encoder members for the nlm-v27 fork: one encoder per GPU (enc_worker.py), stage 1 on Dhruv Tibarewal's
# 120k pool (nlm_pool_v5.parquet, which already excludes every train text and near-duplicate), stage 2 on the official
# rows, once per outer fold of lightningv08's OOF bank (so ntl.oofbank.yardstick can score the out-of-fold logits)
# and once on all rows. Writes /kaggle/working/<name>/ in nlm-enc-v5's layout and enc_summary.json.
import glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/enc")
STARTED = time.time()


def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()


print(sh("nvidia-smi -L"), "\n", sh("df -h /tmp /kaggle/working | tail -2"), flush=True)
import torch, transformers
print(dict(torch=torch.__version__, transformers=transformers.__version__), flush=True)
assert torch.cuda.device_count() == 2, "expected T4 x2"

train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
train["label"] = train["model"].map({label: i for i, label in enumerate(CONFIG["labels"])})
train["fold"] = [int(c) for c in CONFIG["folds"]]
assert train.label.notna().all() and len(train) == 2699
pool_path = sorted(glob.glob("/kaggle/input/**/nlm_pool_v5.parquet", recursive=True))[0]
pool = pd.read_parquet(pool_path)
pool = pool[pool.model.isin(CONFIG["labels"])].copy()
pool["text"] = pool.text.astype(str)
overlap = pool.text.isin(set(train.text))
print(f"pool {pool_path}: {len(pool)} rows, {int(overlap.sum())} exact train texts dropped", flush=True)
pool = pool[~overlap]
pool["label"] = pool.model.map({label: i for i, label in enumerate(CONFIG["labels"])})
pool["w"] = pool.get("w_host", 1.0)
TMP.mkdir(parents=True, exist_ok=True)
pool[["text", "label", "w"]].to_parquet(TMP / "pool.parquet", index=False)
train[["id", "text", "label", "fold"]].to_parquet(TMP / "train.parquet", index=False)

LOGS = WORK / "logs"
LOGS.mkdir(exist_ok=True)
workers = []
for gpu, m in enumerate(CONFIG["models"]):
    if gpu:
        time.sleep(60)
    command = [sys.executable, "-u", str(WORK / "enc_worker.py"), "--model", m["hf"], "--pool", str(TMP / "pool.parquet"),
               "--train", str(TMP / "train.parquet"), "--labels", json.dumps(CONFIG["labels"]),
               "--out", str(WORK / m["name"]), *m["args"]]
    print("launch", " ".join(command[:6]), "...", " ".join(m["args"]), flush=True)
    log = open(LOGS / f"{m['name']}.log", "w")
    workers.append((m["name"], subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                                         PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")), log))
while any(p.poll() is None for _, p, _ in workers):
    time.sleep(300)
    for name, _, _ in workers:
        lines = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()
        print(f"[{(time.time() - STARTED) / 3600:.2f} h] {name}: {lines[-1] if lines else ''}", flush=True)
summary = {}
for name, p, log in workers:
    log.close()
    tail = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()
    print(f"==== {name} exit {p.returncode}\n" + "\n".join(tail[-12:]), flush=True)
    if p.returncode == 0:
        record = json.loads((WORK / name / "record.json").read_text())
        summary[name] = dict(oof_balanced_accuracy=record["oof_balanced_accuracy"],
                             folds={k: v.get("valid_balanced_accuracy") for k, v in record["folds"].items()},
                             stage1=record["stage1"], in_sample_check_40=record["in_sample_check_40"],
                             minutes=record["minutes"])
(WORK / "enc_summary.json").write_text(json.dumps(summary, indent=1))
assert summary, "no encoder finished"
print("ENC MEMBERS DONE", json.dumps({k: round(v["oof_balanced_accuracy"], 4) for k, v in summary.items()}),
      f"{(time.time() - STARTED) / 3600:.2f} h", flush=True)
