# ---- driver cell (inserted by kaggle/member/build_member.py, which fills CONFIG)
# Full fit of the Mistral-7B member on all 2,699 official rows, one seed per GPU, for the V28 fork (Option A of the
# training track). Writes /kaggle/working/mistral_full/: full_seed<s>.safetensors (trainable weights), member.json
# and check.npz (in-sample logits of 256 fixed rows). Then mistral_infer.py, the peft-free loader the V28 fork runs,
# re-predicts those rows in a fresh process and must reproduce the training process's logits.
import glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/member")
OUT = WORK / "mistral_full"
STARTED = time.time()


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
versions = dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
                bitsandbytes=bitsandbytes.__version__)
print(versions, flush=True)
assert torch.cuda.device_count() == 2, "expected T4 x2"

configs = glob.glob(CONFIG["kaggle_model"] + "/**/config.json", recursive=True)
assert len(configs) == 1, configs
BACKBONE = str(Path(configs[0]).parent)
print("backbone", BACKBONE, sorted(os.listdir(BACKBONE)), flush=True)

train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
train["fold"] = 0  # unused: fold -1 trains on every row
train["label"] = train["model"].map({label: i for i, label in enumerate(CONFIG["labels"])})
assert train.label.notna().all() and len(train) == 2699
TMP.mkdir(parents=True, exist_ok=True)
train[["id", "text", "label", "fold"]].to_parquet(TMP / "member_data.parquet", index=False)

OUT.mkdir(exist_ok=True)
LOGS = WORK / "logs"
LOGS.mkdir(exist_ok=True)
workers = []
for gpu, seed in enumerate(CONFIG["seeds"]):
    if gpu:
        time.sleep(120)  # stagger the two model loads (host RAM)
    log = open(LOGS / f"train_seed{seed}.log", "w")
    command = [sys.executable, str(WORK / "member_worker.py"), "--data", str(TMP / "member_data.parquet"),
               "--backbone", BACKBONE, "--folds", "-1", "--seed", str(seed), "--out", str(OUT), *CONFIG["worker_args"]]
    print("launch", " ".join(command), flush=True)
    workers.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))), log, seed))
while any(p.poll() is None for p, _, _ in workers):
    time.sleep(120)
    for _, _, seed in workers:
        lines = (LOGS / f"train_seed{seed}.log").read_text(errors="replace").splitlines()
        print(f"[{(time.time() - STARTED) / 3600:.2f} h] seed {seed}: {lines[-1] if lines else ''}", flush=True)
codes = [p.wait() for p, _, _ in workers]
for _, log, seed in workers:
    log.close()
    print(f"==== seed {seed} log tail\n" + "\n".join((LOGS / f"train_seed{seed}.log").read_text(errors="replace").splitlines()[-8:]))
assert codes == [0] * len(workers), f"training failed: exit codes {codes}"

from safetensors import safe_open
names = {}
for seed in CONFIG["seeds"]:
    with safe_open(str(OUT / f"full_seed{seed}.safetensors"), "pt") as f:
        names[seed] = sorted(f.keys())
assert len({tuple(v) for v in names.values()}) == 1, "seeds saved different parameter sets"
lora = sorted({n.rsplit(".lora_", 1)[0] for n in names[CONFIG["seeds"][0]] if ".lora_" in n})
assert len(lora) == 32 * 7 and len(names[CONFIG["seeds"][0]]) == 2 * len(lora) + 1, (len(lora), len(names[0]))
args = dict(zip(CONFIG["worker_args"][::2], CONFIG["worker_args"][1::2]))
member = dict(labels=list(CONFIG["labels"]), rank=int(args["--rank"]), alpha=2 * int(args["--rank"]),
              max_len=int(args["--max-len"]), lora_modules=len(lora), seeds=CONFIG["seeds"],
              base_model=CONFIG["model_source"], train_sha256=CONFIG["train_sha256"], train_versions=versions,
              worker_args=CONFIG["worker_args"],
              sha256={f"full_seed{s}.safetensors": hashlib.sha256((OUT / f"full_seed{s}.safetensors").read_bytes()).hexdigest()
                      for s in CONFIG["seeds"]},
              records={s: json.loads((OUT / f"full_seed{s}.json").read_text()) for s in CONFIG["seeds"]})
(OUT / "member.json").write_text(json.dumps(member, indent=1))

# check rows: the same 256 for every seed (data.sample(random_state=0)); written as a query CSV for mistral_infer.py
check = {}
for seed in CONFIG["seeds"]:
    with np.load(OUT / f"full_seed{seed}.npz") as z:
        check[seed] = (z["ids"].astype(str), z["logits"])
ids = check[CONFIG["seeds"][0]][0]
assert all((v[0] == ids).all() for v in check.values())
truth = train.set_index("id").loc[ids]
np.savez(OUT / "check.npz", ids=ids, labels=np.asarray(CONFIG["labels"]), truth=truth.label.to_numpy(),
         **{f"logits_seed{s}": v[1] for s, v in check.items()})
truth.reset_index()[["id", "text"]].to_csv(TMP / "check.csv", index=False)
for seed in CONFIG["seeds"]:
    print(f"seed {seed}: in-sample accuracy on the check rows {(check[seed][1].argmax(1) == truth.label.to_numpy()).mean():.3f}")

runs = []
for gpu, seed in enumerate(CONFIG["seeds"]):
    command = [sys.executable, str(WORK / "mistral_infer.py"), "--backbone", BACKBONE, "--member", str(OUT),
               "--seed", str(seed), "--query", str(TMP / "check.csv"), "--out", str(TMP / f"reload_seed{seed}.npz"),
               "--start-delay", str(90 * gpu), "--deadline", str(time.monotonic() + 3600)]
    runs.append(subprocess.Popen(command, stdout=open(LOGS / f"reload_seed{seed}.log", "w"), stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))))
codes = [p.wait() for p in runs]
for seed in CONFIG["seeds"]:
    print(f"==== reload seed {seed}\n" + "\n".join((LOGS / f"reload_seed{seed}.log").read_text(errors="replace").splitlines()[-6:]))
assert codes == [0] * len(runs), f"mistral_infer.py failed: exit codes {codes}"
parity = {}
for seed in CONFIG["seeds"]:
    with np.load(TMP / f"reload_seed{seed}.npz") as z:
        assert (z["ids"].astype(str) == ids).all() and z["labels"].astype(str).tolist() == list(CONFIG["labels"])
        diff = np.abs(z["logits"] - check[seed][1])
        parity[seed] = dict(max_abs=float(diff.max()), mean_abs=float(diff.mean()),
                            argmax_agree=float((z["logits"].argmax(1) == check[seed][1].argmax(1)).mean()))
print("parity", json.dumps(parity), flush=True)
assert all(p["argmax_agree"] >= 0.99 and p["max_abs"] <= 0.1 for p in parity.values()), "reload does not reproduce training"
member["parity"] = parity
member["hours"] = (time.time() - STARTED) / 3600
(OUT / "member.json").write_text(json.dumps(member, indent=1))
shutil.copy(WORK / "mistral_infer.py", OUT / "mistral_infer.py")
for path in OUT.glob("full_seed*.npz"):
    path.unlink()  # copied into check.npz
shutil.rmtree(TMP, ignore_errors=True)
print("FULL FIT DONE", json.dumps({k: member[k] for k in ("lora_modules", "parity", "hours")}), sorted(os.listdir(OUT)))
