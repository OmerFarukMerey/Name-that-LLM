# ---- driver cell (inserted by kaggle/member/build_member.py, which fills CONFIG)
# Stage 1 of a LoRA24 member on the TPU v3-8 (tpu_s1.py, data-parallel over the 8 cores): continues the stage-1 output
# of an attached kernel (CONFIG['init_tag']) on Dhruv Tibarewal's 120k public pool, as l24_driver.py's GPU
# continuation does, up to CONFIG['epochs_total'] pool epochs or CONFIG['hours']. Output: /kaggle/working/<out_tag>/,
# which l24_driver.py's stage 2 (GPU) reads like a GPU stage-1 output.
import glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/l24")
STARTED = time.time()


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}\n{(r.stdout + r.stderr)[-3000:]}", flush=True)
    return r.returncode


PKGS = 'pip list 2>/dev/null | grep -iE "^(torch|torch[-_]xla|libtpu|transformers|peft|accelerate|tokenizers|safetensors) "'
sh(PKGS)
init = [os.path.dirname(f) for f in glob.glob(f"/kaggle/input/**/{CONFIG['init_tag']}/meta.json", recursive=True)
        if json.load(open(f)).get("stage") == 1]
assert len(init) == 1, init
init = init[0]
prev = json.load(open(os.path.join(init, "meta.json")))
base = prev["base"]   # the same Kaggle model mount path as in the GPU kernel that wrote it
assert os.path.exists(os.path.join(base, "config.json")), base
if sh('python -c "import torch_xla"'):
    import torch
    v = torch.__version__.split("+")[0]
    sh(f'pip install -q "torch=={v}" "torch_xla[tpu]=={v}" -f https://storage.googleapis.com/libtpu-releases/index.html')
if sh(f"python -c \"from transformers import AutoConfig; AutoConfig.from_pretrained('{base}'); import peft\""):
    import torch   # Ministral 3 needs a recent transformers; never let pip replace the TPU build of torch
    open("/tmp/constraints.txt", "w").write(f"torch=={torch.__version__}\n")
    sh("pip install -q -U transformers peft -c /tmp/constraints.txt")
sh(PKGS)

train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
TMP.mkdir(parents=True, exist_ok=True)
pool_path = sorted(glob.glob("/kaggle/input/**/nlm_pool_v5.parquet", recursive=True))[0]
pool = pd.read_parquet(pool_path)
pool = pool[pool.model.isin(CONFIG["labels"])].copy()
pool["text"] = pool.text.astype(str)
overlap = pool.text.isin(set(train.text))
print(f"pool {pool_path}: {len(pool)} rows, {int(overlap.sum())} exact train texts dropped", flush=True)
pool[~overlap][["text", "model", "prompt_key"]].to_parquet(TMP / "pool.parquet", index=False)

done_before = prev.get("epochs_total", prev["epochs_done"])
epochs = round(CONFIG["epochs_total"] - done_before, 3)
out = WORK / CONFIG["out_tag"]
args = [sys.executable, "-u", str(WORK / "tpu_s1.py"), "--stage", "1", "--base", base, "--labels", json.dumps(CONFIG["labels"]),
        "--pool", str(TMP / "pool.parquet"), *CONFIG["args"], "--epochs", str(epochs), "--init", init, "--seed", "1",
        "--hours", str(CONFIG["hours"]), "--class-weights", "--out", str(out)]
print(f"continuing {CONFIG['init_tag']} from {init} ({done_before} pool epochs done) for {epochs} more, at most "
      f"{CONFIG['hours']} h, into {out}", flush=True)
rc = subprocess.run(args, env=dict(os.environ, PJRT_DEVICE="TPU", TOKENIZERS_PARALLELISM="false")).returncode
meta = json.loads((out / "meta.json").read_text())
assert rc == 0 and (out / "adapter_model.safetensors").exists(), "stage 1 failed"
print("STAGE 1 DONE", CONFIG["out_tag"], json.dumps({k: meta[k] for k in ("steps", "total_steps", "planned_steps", "hours",
      "epochs_done", "epochs_total", "final_loss", "world", "device")}), f"{(time.time() - STARTED) / 3600:.2f} h", flush=True)
