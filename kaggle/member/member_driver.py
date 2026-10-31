# ---- driver cell (inserted by kaggle/member/build_member.py, which fills CONFIG)
# Trains one LoRA classifier per outer fold of lightningv08's OOF bank, two GPUs in parallel, and writes
# /kaggle/working/oof.parquet (id, prompt_id, model, one probability column per label) and metrics.json.
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/member")
STARTED = time.time()
DEADLINE_HOURS = 11.3  # a batch session ends at 12 h; stop the workers first and keep finished folds


def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()


print(sh("nvidia-smi -L"), "\n", sh("df -h /tmp /kaggle/working | tail -2"), flush=True)
for module, package in (("peft", "peft"), ("bitsandbytes", "bitsandbytes")):
    try:
        __import__(module)
    except ImportError:
        print(f"installing {package}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
import torch, transformers, peft, bitsandbytes
print("torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__,
      "bitsandbytes", bitsandbytes.__version__, flush=True)
assert torch.cuda.device_count() == 2, "expected T4 x2"

if CONFIG["kaggle_model"]:
    configs = glob.glob(CONFIG["kaggle_model"] + "/**/config.json", recursive=True)
    if len(configs) != 1:
        print(sh("find /kaggle/input -maxdepth 6 | head -80"))
        raise SystemExit(f"expected one config.json under {CONFIG['kaggle_model']}: {configs}")
    BACKBONE = str(Path(configs[0]).parent)
else:
    from huggingface_hub import snapshot_download
    BACKBONE = snapshot_download(CONFIG["hf_repo"], revision=CONFIG["hf_revision"], local_dir=str(TMP / "backbone"),
                                 allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    from huggingface_hub import HfApi
    CONFIG["hf_commit"] = HfApi().model_info(CONFIG["hf_repo"], revision=CONFIG["hf_revision"]).sha
    print("hf commit", CONFIG["hf_commit"], flush=True)
print("backbone", BACKBONE, sorted(os.listdir(BACKBONE)), flush=True)

train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
train["fold"] = [int(c) for c in CONFIG["folds"]]
train["label"] = train["model"].map({label: i for i, label in enumerate(CONFIG["labels"])})
assert train.label.notna().all() and train.groupby("prompt_id").fold.nunique().max() == 1
TMP.mkdir(parents=True, exist_ok=True)
train[["id", "text", "label", "fold"]].to_parquet(TMP / "member_data.parquet", index=False)

FOLDS_OUT, LOGS = WORK / "folds", WORK / "logs"
FOLDS_OUT.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)
workers = []
for gpu, folds in enumerate(CONFIG["gpu_folds"]):
    if gpu:
        time.sleep(120)  # stagger the two model loads (host RAM)
    log = open(LOGS / f"gpu{gpu}.log", "w")
    command = [sys.executable, str(WORK / "member_worker.py"), "--data", str(TMP / "member_data.parquet"),
               "--backbone", BACKBONE, "--folds", folds, "--out", str(FOLDS_OUT), *CONFIG["worker_args"]]
    print("launch", " ".join(command), flush=True)
    workers.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))), log))

while any(p.poll() is None for p, _ in workers):
    time.sleep(120)
    hours = (time.time() - STARTED) / 3600
    for gpu in range(len(workers)):
        lines = (LOGS / f"gpu{gpu}.log").read_text(errors="replace").splitlines()
        print(f"[{hours:.2f} h] gpu{gpu}: {lines[-1] if lines else ''}", flush=True)
    if hours > DEADLINE_HOURS:
        print("deadline: stopping workers", flush=True)
        for p, _ in workers:
            p.terminate()
codes = [p.wait() for p, _ in workers]
for _, log in workers:
    log.close()
for gpu in range(len(workers)):
    print(f"==== gpu{gpu} log tail\n" + "\n".join((LOGS / f"gpu{gpu}.log").read_text(errors="replace").splitlines()[-15:]))

done = sorted(int(p.stem[4:]) for p in FOLDS_OUT.glob("fold*.npz"))
ids, logits = [], []
for k in done:
    with np.load(FOLDS_OUT / f"fold{k}.npz") as z:
        ids += list(z["ids"])
        logits.append(z["logits"].astype(np.float64))
logits = np.concatenate(logits) if logits else np.zeros((0, len(CONFIG["labels"])))
proba = np.exp(logits - logits.max(1, keepdims=True))
proba /= proba.sum(1, keepdims=True)
oof = pd.DataFrame(proba, columns=CONFIG["labels"])
oof.insert(0, "id", ids)
oof = train[["id", "prompt_id", "model", "fold"]].merge(oof, on="id", how="inner")
oof.to_parquet(WORK / "oof.parquet", index=False)


def balanced_accuracy(true, pred):
    frame = pd.DataFrame({"t": list(true), "p": list(pred)})
    return float((frame.t == frame.p).groupby(frame.t).mean().mean())


labels = np.array(CONFIG["labels"])
pred = labels[oof[CONFIG["labels"]].to_numpy().argmax(1)] if len(oof) else []
metrics = dict(config={k: v for k, v in CONFIG.items() if k != "folds"}, backbone_files=sorted(os.listdir(BACKBONE)),
               worker_exit_codes=codes, folds_done=done, rows=len(oof),
               balanced_accuracy=balanced_accuracy(oof.model, pred) if len(oof) else None,
               per_fold={int(k): balanced_accuracy(g.model, labels[g[CONFIG["labels"]].to_numpy().argmax(1)])
                         for k, g in oof.groupby("fold")},
               fold_records=[json.loads((FOLDS_OUT / f"fold{k}.json").read_text()) for k in done],
               hours=(time.time() - STARTED) / 3600)
(WORK / "metrics.json").write_text(json.dumps(metrics, indent=1))
print(json.dumps({k: metrics[k] for k in ("worker_exit_codes", "folds_done", "rows", "balanced_accuracy", "per_fold", "hours")}, indent=1))
shutil.rmtree(TMP, ignore_errors=True)
assert codes == [0] * len(workers) and len(done) == 5, "pilot incomplete: see the logs above"
