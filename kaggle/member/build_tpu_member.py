"""Build the TPU member kernels: tpu_member.py (with l24_worker.py, which it imports) and tpu_member_driver.py.

    PYTHONPATH=src python kaggle/member/build_tpu_member.py [kernel ...]

Training kernels only (internet on for peft), never a submission. The folds are exp015's (the split inferred from
nlm-v27's fold adapters), so the out-of-fold log-probs (oof.npz) drop into the replica. Output: kaggle/member/<kernel>/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_member import HERE, filled, notebook
from ntl.oofbank import TRAIN_SHA256

ROOT = HERE.parents[1]
NLM_LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
              'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
              'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
QWEN8 = dict(base_parts=["/qwen-3/", "/8b-base/"], model="qwen-lm/qwen-3/Transformers/8b-base/1")
QWEN14 = dict(base_parts=["/qwen-3/", "/14b-base/"], model="qwen-lm/qwen-3/Transformers/14b-base/1")
LLAMA8 = dict(base_parts=["/llama-3-1-8b-base/"], model="alexchilton/llama-3-1-8b-base/Transformers/default/1")   # Llama 3.1 Community License
EPOCHS = ["--epochs", "2", "--epochs2", "2"]
# exp046: weighting the pool's rows by match_weight (per label, clip 4) lifts a pool-only TF-IDF model on the official
# rows from 0.4290 to 0.4460 (+0.0169 [+0.0078, +0.0267]); stage 1 of the later members uses it
PW = ["--pool-weight", "1.0", "--pool-weight-clip", "4"]
TPU = {   # kernel -> members trained one after another in one TPU session (one TPU session at a time per account)
    # Qwen3-8B-Base LoRA24 on the open pool (2 epochs), then the official rows (2 epochs per job)
    # (version 1 was pushed 2026-11-03 09:05 UTC with the single-member driver of that time: same trainer and settings)
    "ntl-tpu-op8": dict(session_hours=7.5, members=[dict(tag="op8v1", hours=2.5, after_hours=0.5, args=EPOCHS, **QWEN8)]),
    # Llama-3.1-8B (another family, as ll8v5), then Qwen3-14B-Base (q14v5's base, keep 27 as q14v5; too large to
    # replicate on one chip, so its weights are split 4 ways)
    # (max_group: prompt_groups of more than 24 rows are whole source files, not prompts; their rows go solo)
    "ntl-tpu-ll8q14": dict(session_hours=7.5, max_group=24, members=[
        dict(tag="ll8op1", hours=2.3, after_hours=0.5, args=EPOCHS + PW, **LLAMA8),
        dict(tag="q14op1", hours=3.0, after_hours=0.8, args=["--keep", "27", "--tp", "4"] + EPOCHS + PW, **QWEN14)]),
}


CHECKS = {   # GPU reload checks of a TPU kernel's adapters (tpu_check_driver.py)
    "ntl-tpu-op8-check": dict(kernel="omerfarukmerey/ntl-tpu-op8", members=[dict(keep=24, dirs=["op8v1", "op8v1_f0"], **QWEN8)]),
    "ntl-tpu-ll8q14-check": dict(kernel="omerfarukmerey/ntl-tpu-ll8q14", members=[
        dict(keep=24, dirs=["ll8op1", "ll8op1_f0"], **LLAMA8), dict(keep=27, dirs=["q14op1", "q14op1_f0"], **QWEN14)]),
}


def build_check(kernel):
    k = CHECKS[kernel]
    config = dict(members=[{x: m[x] for x in ("base_parts", "keep", "dirs")} for m in k["members"]])
    intro = f"# {kernel}\n\nReload check of {k['kernel']}'s adapters through infer.py's merge path on the T4. Not a submission."
    out = HERE / kernel
    out.mkdir(exist_ok=True)
    (out / f"{kernel}.ipynb").write_text(json.dumps(notebook(intro, [filled("tpu_check_driver.py", config)]), indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": [], "kernel_sources": [k["kernel"]],
            "competition_sources": ["name-that-llm"], "model_sources": sorted({m["model"] for m in k["members"]}),
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built", out)


def folds():
    """prompt_id -> exp015 fold (the folds stored with nlm_oof.npz, one per train.csv row)."""
    z = np.load(ROOT / "outputs" / "exp015_nlm_core_oof" / "nlm_oof.npz", allow_pickle=False)
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    m = pd.Series(z["folds"].astype(int)).groupby(train.prompt_id.values).agg(["min", "max"])
    assert (m["min"] == m["max"]).all() and len(m) == 150, "folds are not prompt-grouped"
    return {p: int(f) for p, f in m["min"].items()}


def build(kernel):
    k = TPU[kernel]
    members = [{x: m[x] for x in ("tag", "base_parts", "hours", "after_hours", "args")} for m in k["members"]]
    config = dict(members=members, session_hours=k["session_hours"], labels=NLM_LABELS, train_sha256=TRAIN_SHA256,
                  folds=folds(), max_group=k.get("max_group"))
    intro = (f"# {kernel}\n\nLoRA24 members (Lucas Boesen's recipe with Dhruv Tibarewal's two stages) trained on the "
             "TPU v5e-8 with torch_xla SPMD, one after another: stage 1 on crtproduction's open pool, stage 2 on the "
             "official rows per held-out fold and on all rows. Training kernel, not a submission.")
    cells = ["%%writefile /kaggle/working/l24_worker.py\n" + (HERE / "l24_worker.py").read_text(encoding="utf-8").rstrip("\n"),
             "%%writefile /kaggle/working/tpu_member.py\n" + (HERE / "tpu_member.py").read_text(encoding="utf-8").rstrip("\n"),
             filled("tpu_member_driver.py", config)]
    out = HERE / kernel
    out.mkdir(exist_ok=True)
    (out / f"{kernel}.ipynb").write_text(json.dumps(notebook(intro, cells), indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": True,
            "enable_internet": True, "keywords": [], "dataset_sources": ["crtproduction/name-that-llm-open-pool"],
            "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "model_sources": sorted({m["model"] for m in k["members"]}), "machine_shape": "TpuV5E8"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("built", out, [m["tag"] for m in members])


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for name in sys.argv[1:] or list(TPU):
        build_check(name) if name in CHECKS else build(name)
