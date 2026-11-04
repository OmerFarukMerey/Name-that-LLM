# ---- driver cell (inserted by kaggle/member/build_tpu_member.py, which fills CONFIG)
# LoRA24 members trained end to end on the TPU v5e-8 (tpu_member.py), one after another in this session: stage 1 on
# crtproduction's open pool (every label, minus anything regenerated from a train prompt or equal to a train text), then
# stage 2 on the 2,699 official rows once per held-out fold (exp015's split, so the out-of-fold log-probs line up with
# the replica) and once on all. Each member's stage-1 time box is cut to what is left of CONFIG['session_hours'].
import glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/tpm")
STARTED = time.time()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout + r.stderr)[-3000:]}", flush=True)
    return r.returncode


PKGS = 'pip list 2>/dev/null | grep -iE "^(torch|torch[-_]xla|libtpu|transformers|peft|accelerate|tokenizers|safetensors) "'
sh(PKGS)
sh('python -c "import torch_xla.runtime as xr; print(xr.device_type(), xr.global_runtime_device_count())"')
if sh('python -c "import peft"'):
    import torch   # never let pip replace the TPU build of torch
    open("/tmp/constraints.txt", "w").write(f"torch=={torch.__version__}\n")
    sh("pip install -q peft -c /tmp/constraints.txt")
    sh(PKGS)
cfgs = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
TMP.mkdir(parents=True, exist_ok=True)
train["label"] = train.model.map({l: i for i, l in enumerate(CONFIG["labels"])}).astype(int)
train["fold"] = train.prompt_id.map(CONFIG["folds"]).astype(int)
train[["id", "text", "label", "fold", "prompt_id"]].to_parquet(TMP / "train.parquet", index=False)
op_path = sorted(glob.glob("/kaggle/input/**/ntl_open_pool.parquet", recursive=True))[0]
op = pd.read_parquet(op_path)
n0 = len(op)
op = op[op.model.isin(CONFIG["labels"]) & ~op.flag_train_prompt_regen.astype(bool)]
op = op[~op.text.astype(str).isin(set(train.text))].rename(columns={"prompt_group": "prompt_key"})
op["prompt_key"] = op.prompt_key.astype(str)
if CONFIG.get("max_group"):   # a whole source file shares one prompt_group: no same-prompt triplets from it
    big = op.prompt_key.map(op.prompt_key.value_counts()) > CONFIG["max_group"]
    op.loc[big, "prompt_key"] = "solo_" + pd.Series(range(int(big.sum())), index=op.index[big]).astype(str)
    print(f"{int(big.sum())} rows in groups of more than {CONFIG['max_group']} rows treated as singletons", flush=True)
op[["text", "model", "prompt_key", "match_weight"]].to_parquet(TMP / "pool.parquet", index=False)
print(f"open pool {op_path}: {n0} rows, {len(op)} kept", op.model.value_counts().to_dict(), flush=True)
results = {}
for mem in CONFIG["members"]:
    base = [os.path.dirname(c) for c in cfgs if all(p in c.lower() for p in mem["base_parts"])]
    assert base, (mem["tag"], cfgs[:20])
    left = CONFIG["session_hours"] - (time.time() - STARTED) / 3600 - mem["after_hours"]
    hours = min(mem["hours"], left)
    if hours < 0.5:
        print(f"SKIP {mem['tag']}: {left:.2f} h left in the session", flush=True)
        continue
    out = WORK / mem["tag"]
    args = [sys.executable, "-u", str(WORK / "tpu_member.py"), "--base", base[0], "--labels", json.dumps(CONFIG["labels"]),
            "--pool", str(TMP / "pool.parquet"), "--train", str(TMP / "train.parquet"), "--out", str(out),
            "--hours", f"{hours:.2f}", *mem["args"]]
    print(f"MEMBER {mem['tag']} on {base[0]}, stage-1 time box {hours:.2f} h", flush=True)
    rc = subprocess.run(args, env=dict(os.environ, PJRT_DEVICE="TPU", TOKENIZERS_PARALLELISM="false")).returncode
    results[mem["tag"]] = rc
    print("EXIT", mem["tag"], rc, f"{(time.time() - STARTED) / 3600:.2f} h", flush=True)
    for f in ("summary.json", "oof.npz", "s1_on_train.npy"):   # tpu_member.py writes these next to the member's dirs
        if (WORK / f).exists():
            (WORK / f).rename(WORK / f"{mem['tag']}_{f}")
    if (WORK / f"{mem['tag']}_summary.json").exists():
        print((WORK / f"{mem['tag']}_summary.json").read_text(), flush=True)
print("RESULTS", results, flush=True)
assert results and all(rc == 0 for rc in results.values()), "a member failed"
