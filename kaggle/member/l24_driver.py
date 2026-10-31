# ---- driver cell (inserted by kaggle/member/build_member.py, which fills CONFIG)
# Two-stage LoRA24 member (l24_worker.py), in the layout of Dhruv Tibarewal's nlm-l24-adapters-v5.
# Stage 1 (CONFIG['stage'] == 1): an fp16 health / memory / speed check of each candidate base in turn (the first that
# passes is trained), then the pool stage on Dhruv's 120k public pool (nlm_pool_v5.parquet, which already excludes
# every train text and near-duplicate), data-parallel over both T4s and time-boxed. Output: /kaggle/working/<tag>/.
# Stage 2 (CONFIG['stage'] == 2): from that output, the official rows once per held-out fold (the split inferred from
# nlm-v27's own fold adapters, so out-of-fold predictions line up with exp015's replica) and once on all rows, three
# jobs per GPU; then the out-of-fold log-probs (enc/<tag>/oof.npz) and a reload check of the full-data adapter through
# infer.py's own merge path (nlm_oof.py).
import glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

CONFIG = None
WORK, TMP = Path("/kaggle/working"), Path("/tmp/l24")
STARTED = time.time()


def sh(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip()


def hours():
    return f"{(time.time() - STARTED) / 3600:.2f} h"


print(sh("nvidia-smi -L"), "\n", sh("df -h /tmp /kaggle/working | tail -2"), flush=True)
import torch, transformers
try:
    import peft
except ImportError:
    print(sh(f"{sys.executable} -m pip install -q peft 2>&1 | tail -2"), flush=True)
    import peft
print(dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__), flush=True)
assert torch.cuda.device_count() == 2, "expected T4 x2"
cfgs = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))


def mdir(parts):
    hits = [os.path.dirname(c) for c in cfgs if all(p in c.lower() for p in parts)]
    assert hits, parts
    return hits[0]


train_csv = sorted(glob.glob("/kaggle/input/**/name-that-llm/train.csv", recursive=True))[0]
assert hashlib.sha256(Path(train_csv).read_bytes()).hexdigest() == CONFIG["train_sha256"], "train.csv changed"
train = pd.read_csv(train_csv, dtype=str, keep_default_na=False)
assert len(train) == 2699
TMP.mkdir(parents=True, exist_ok=True)
LOGS = WORK / "logs"
LOGS.mkdir(exist_ok=True)
labels = json.dumps(CONFIG["labels"])
worker = [sys.executable, "-u", str(WORK / "l24_worker.py")]
env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TOKENIZERS_PARALLELISM="false")


def follow(procs, every=600):
    """Wait for the processes, printing the last line of each log every `every` seconds."""
    while any(p.poll() is None for _, p in procs):
        time.sleep(every)
        for name, _ in procs:
            lines = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()
            print(f"[{hours()}] {name}: {lines[-1] if lines else ''}", flush=True)
    for name, p in procs:
        tail = (LOGS / f"{name}.log").read_text(errors="replace").splitlines()
        print(f"==== {name} exit {p.returncode}\n" + "\n".join(tail[-25:]), flush=True)
    return [p.returncode for _, p in procs]


if CONFIG["stage"] == 0:
    # probe: the check of every base (two at a time, one per GPU), then a few minutes of two-GPU stage 1 on a slice
    pool_path = sorted(glob.glob("/kaggle/input/**/nlm_pool_v5.parquet", recursive=True))[0]
    pool = pd.read_parquet(pool_path)
    pool = pool[pool.model.isin(CONFIG["labels"]) & ~pool.text.astype(str).isin(set(train.text))].copy()
    pool[["text", "model", "prompt_key"]].to_parquet(TMP / "pool.parquet", index=False)
    keys = pool.prompt_key.drop_duplicates().sample(400, random_state=0)
    pool[pool.prompt_key.isin(keys)][["text", "model", "prompt_key"]].to_parquet(TMP / "slice.parquet", index=False)
    results = {}
    for s in range(0, len(CONFIG["bases"]), 2):
        procs = []
        for gpu, b in enumerate(CONFIG["bases"][s:s + 2]):
            name = f"check_{b['tag']}"
            args = ["--base", mdir(b["parts"]), "--labels", labels, "--pool", str(TMP / "pool.parquet"), *b["args"]]
            procs.append((name, subprocess.Popen(worker + ["--stage", "0", "--out", str(WORK / name), *args],
                                                 stdout=open(LOGS / f"{name}.log", "w"), stderr=subprocess.STDOUT,
                                                 env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))))
        follow(procs, every=120)
        for name, p in procs:
            f = WORK / name / "check.json"
            results[name] = dict(rc=p.returncode, **(json.loads(f.read_text()) if f.exists() else {}))
    print("CHECKS", json.dumps(results, indent=1), flush=True)
    b = next(b for b in CONFIG["bases"] if results[f"check_{b['tag']}"].get("ok"))
    log = open(LOGS / "ddp.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                          str(WORK / "l24_worker.py"), "--stage", "1", "--out", str(WORK / f"ddp_{b['tag']}"),
                          "--hours", "0.05", "--class-weights", "--ckpt-min", "1", "--base", mdir(b["parts"]),
                          "--labels", labels, "--pool", str(TMP / "slice.parquet"), *b["args"]],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    rc = follow([("ddp", p)], every=60)
    print("PROBE DONE", b["tag"], rc, hours(), flush=True)

if CONFIG["stage"] == 1:
    pool_path = sorted(glob.glob("/kaggle/input/**/nlm_pool_v5.parquet", recursive=True))[0]
    pool = pd.read_parquet(pool_path)
    pool = pool[pool.model.isin(CONFIG["labels"])].copy()
    pool["text"] = pool.text.astype(str)
    overlap = pool.text.isin(set(train.text))
    print(f"pool {pool_path}: {len(pool)} rows, {int(overlap.sum())} exact train texts dropped", flush=True)
    pool[~overlap][["text", "model", "prompt_key"]].to_parquet(TMP / "pool.parquet", index=False)
    chosen = None
    if CONFIG.get("init_tag"):   # continue an earlier stage-1 run of this base (its kernel is attached), no new check
        hits = [os.path.dirname(f) for f in glob.glob(f"/kaggle/input/**/{CONFIG['init_tag']}/meta.json", recursive=True)
                if json.load(open(f)).get("stage") == 1]
        assert len(hits) == 1, hits
        prev = json.load(open(os.path.join(hits[0], "meta.json")))
        done_before = prev.get("epochs_total", prev["epochs_done"])
        b = next(b for b in CONFIG["bases"] if b["tag"] == CONFIG["init_tag"])
        assert os.path.exists(prev["base"]), prev["base"]
        args = ["--base", prev["base"], "--labels", labels, "--pool", str(TMP / "pool.parquet"), *b["args"]]
        for k in ("tri", "accum"):
            args[args.index(f"--{k}") + 1] = str(prev[k])
        args[args.index("--epochs") + 1] = str(round(CONFIG["epochs_total"] - done_before, 3))
        args += ["--init", hits[0], "--seed", "1"]
        print(f"continuing {b['tag']} from {hits[0]} ({done_before} pool epochs done)", flush=True)
        chosen = (b, prev["base"], args)
    for b in [] if chosen else CONFIG["bases"]:
        path = mdir(b["parts"])
        args = ["--base", path, "--labels", labels, "--pool", str(TMP / "pool.parquet"), *b["args"]]
        name = f"check_{b['tag']}"
        with open(LOGS / f"{name}.log", "w") as log:
            rc = subprocess.run(worker + ["--stage", "0", "--out", str(WORK / name), *args], stdout=log,
                                stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES="0")).returncode
        print(f"==== {name} exit {rc} ({hours()})\n" + "\n".join((LOGS / f"{name}.log").read_text(errors="replace").splitlines()[-12:]), flush=True)
        check = json.loads((WORK / name / "check.json").read_text()) if (WORK / name / "check.json").exists() else {}
        if rc == 0 and check.get("ok"):
            if check["tri"] < int(args[args.index("--tri") + 1]):   # smaller micro-batches, same optimizer batch
                f = int(args[args.index("--tri") + 1]) // check["tri"]
                args[args.index("--tri") + 1] = str(check["tri"])
                args[args.index("--accum") + 1] = str(int(args[args.index("--accum") + 1]) * f)
            chosen = (b, path, args); break
    assert chosen, "no base passed the fp16 / memory check"
    b, path, args = chosen
    print(f"training {b['tag']} from {path} for at most {CONFIG['hours']} h", flush=True)
    log = open(LOGS / f"{b['tag']}.log", "w")
    p = subprocess.Popen([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
                          str(WORK / "l24_worker.py"), "--stage", "1",
                          "--out", str(WORK / b["tag"]), "--hours", str(CONFIG["hours"]), "--class-weights", *args],
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    rc = follow([(b["tag"], p)])
    log.close()
    meta = json.loads((WORK / b["tag"] / "meta.json").read_text())
    assert rc == [0] and (WORK / b["tag"] / "adapter_model.safetensors").exists(), "stage 1 failed"
    print("STAGE 1 DONE", b["tag"], json.dumps({k: meta[k] for k in ("steps", "total_steps", "planned_steps", "hours",
          "epochs_done", "epochs_total", "final_loss", "overflow_microbatches", "skipped_steps")}), hours(), flush=True)

if CONFIG["stage"] == 2:
    s1 = sorted({os.path.dirname(m) for m in glob.glob(f"/kaggle/input/**/{CONFIG['tag']}/meta.json", recursive=True)
                 if json.load(open(m)).get("stage") == 1})
    assert len(s1) == 1, s1
    s1 = s1[0]
    s1_meta = json.load(open(os.path.join(s1, "meta.json")))
    print("stage 1:", s1, json.dumps({k: s1_meta.get(k) for k in ("base", "steps", "hours", "epochs_done", "final_loss")}), flush=True)
    path = s1_meta["base"]   # the base stage 1 chose, mounted at the same path here
    assert os.path.exists(os.path.join(path, "config.json")), path
    train["label"] = train.model.map({l: i for i, l in enumerate(CONFIG["labels"])})
    train["fold"] = [int(c) for c in CONFIG["folds"]]
    assert train.label.notna().all() and train.groupby("prompt_id").fold.nunique().max() == 1
    train[["id", "text", "label", "fold", "prompt_id"]].to_parquet(TMP / "train.parquet", index=False)
    procs = []
    for gpu, jobs in enumerate(CONFIG["gpu_jobs"]):
        if gpu:
            time.sleep(60)
        name = f"stage2_gpu{gpu}"
        command = worker + ["--stage", "2", "--base", path, "--labels", labels, "--train", str(TMP / "train.parquet"),
                            "--init", s1, "--jobs", jobs, "--out", str(WORK / CONFIG["tag"]), *CONFIG["args"]]
        print("launch", name, jobs, " ".join(CONFIG["args"]), flush=True)
        procs.append((name, subprocess.Popen(command, stdout=open(LOGS / f"{name}.log", "w"), stderr=subprocess.STDOUT,
                                             env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)))))
    rcs = follow(procs, every=300)
    folds = train.fold.values
    oof = np.zeros((len(train), len(CONFIG["labels"])), np.float32); done = []
    for k in range(5):
        f = WORK / f"{CONFIG['tag']}_f{k}" / "valid.npz"
        if f.exists():
            z = np.load(f)
            assert list(z["ids"]) == list(train.id.values[folds == k])
            oof[folds == k] = z["logits"]; done.append(k)
    summary = {"stage1": {k: s1_meta.get(k) for k in ("base", "steps", "hours", "epochs_done", "final_loss")}, "folds_done": done}
    for d in sorted(WORK.glob(f"{CONFIG['tag']}*/meta.json")):
        mm = json.loads(d.read_text())
        summary[d.parent.name] = {k: mm.get(k) for k in ("job", "steps", "minutes", "final_loss", "in_sample_check_40",
                                                           "valid_accuracy", "valid_balanced_accuracy", "overflow_microbatches")}
    if len(done) == 5:
        y = train.label.astype(int).values
        summary["oof_balanced_accuracy"] = float(np.mean([(oof[y == c].argmax(1) == c).mean() for c in range(len(CONFIG["labels"]))]))
        out = WORK / "enc" / CONFIG["tag"]
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "oof.npz", ids=np.array(train.id.tolist()), labels=np.array(CONFIG["labels"]), logits=oof)
    full = WORK / CONFIG["tag"]
    if (full / "heads.pt").exists():
        # reload check: infer.py's merge path (nlm_oof.py) on the rows the worker checked in-process
        sys.path.insert(0, str(WORK))
        import nlm_oof
        spec = dict(name=CONFIG["tag"], path=path, keep=s1_meta["keep"], layers=s1_meta["layers"], bs=8, max_len=s1_meta["max_len"])
        tok, m, mods, first = nlm_oof.load_base(spec)
        with nlm_oof.merged(mods, str(full)) as H:
            feats = nlm_oof.pooled(spec, tok, m, first, train.text.values[:40], spec["layers"], [0])
            lp = nlm_oof.head_lp(H, feats, spec["layers"])
        ref = np.load(full / "check40.npy")
        summary["reload_check"] = dict(max_abs_diff=float(np.abs(lp - ref).max()), argmax_agreement=float((lp.argmax(1) == ref.argmax(1)).mean()),
                                       accuracy_40=float((lp.argmax(1) == train.label.astype(int).values[:40]).mean()))
        print("reload check", summary["reload_check"], flush=True)
    (WORK / "l24_summary.json").write_text(json.dumps(summary, indent=1))
    print("STAGE 2 DONE", json.dumps(summary, indent=1), hours(), flush=True)
    assert rcs == [0, 0] and len(done) == 5 and summary.get("reload_check", {}).get("argmax_agreement", 0) >= 0.9, "stage 2 incomplete"
