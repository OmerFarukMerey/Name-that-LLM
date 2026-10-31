"""Build the phase-1 pilot kernels: a LoRA classifier trained on the official rows only, 5 folds, out-of-fold.

    PYTHONPATH=src python kaggle/member/build_member.py [kernel ...]      (default: every pilot and full fit)

Each kernel is a two-cell notebook: member_worker.py written to /kaggle/working by %%writefile, then
member_driver.py with CONFIG filled in. The folds are lightningv08's OOF-bank folds (ntl.oofbank.bank_folds),
so the result can be scored with ntl.oofbank.yardstick. Internet is on (pip, Hugging Face download); this is a
training kernel, never a submission. Output: kaggle/member/<kernel>/ (gitignored).

FULL holds the full-fit kernels (Option A of the training track, approved 2026-10-26): the pilot's recipe trained on
all 2,699 rows, one seed per GPU, by full_driver.py, with mistral_infer.py (the peft-free loader the V28 fork runs)
written next to the worker and checked against the training process's own logits.
"""
import json
import sys
from pathlib import Path

from ntl.data import LABELS
from ntl.oofbank import TRAIN_SHA256, bank_folds

HERE = Path(__file__).resolve().parent
COMMON = ["--epochs", "2", "--lr", "1e-4", "--rank", "16", "--max-len", "512", "--batch", "4", "--accum", "2"]
PILOTS = {
    "ntl-member-pilot-mistral7b": dict(kaggle_model="/kaggle/input/models/mistral-ai/mistral",
                                       model_sources=["mistral-ai/mistral/pyTorch/7b-v0.1-hf/1"],
                                       hf_repo=None, hf_revision=None),
    "ntl-member-pilot-phi4": dict(kaggle_model=None, model_sources=[], hf_repo="microsoft/phi-4", hf_revision="main"),
}


FULL = {
    "ntl-member-mistral7b-full": dict(kaggle_model="/kaggle/input/models/mistral-ai/mistral",
                                      model_source="mistral-ai/mistral/pyTorch/7b-v0.1-hf/1", seeds=[0, 1]),
}


ENC = {  # two-stage encoder members for the nlm-v27 fork (pool stage 1, official stage 2), one per GPU
    "ntl-member-enc-pool": [
        dict(name="roberta_large_pool", hf="FacebookAI/roberta-large",
             args=["--batch", "16", "--accum", "1", "--lr1", "2e-5", "--epochs1", "2", "--lr2", "1e-5", "--epochs2", "3",
                   "--max-len", "384", "--stage1-hours", "4.5"]),
        dict(name="deberta_v3_large_pool", hf="microsoft/deberta-v3-large",
             args=["--batch", "8", "--accum", "2", "--lr1", "2e-5", "--epochs1", "2", "--lr2", "1e-5", "--epochs2", "3",
                   "--max-len", "384", "--stage1-hours", "4.5"]),
    ],
    # replicas of nlm-v27's two encoders (Dhruv's ModernBERT-large, Nishant's DeBERTa-v3-large), for exp015's out-of-fold
    # replica; DeBERTa at batch 4 (batch 8 ran out of memory) and one pool epoch, to finish with the ModernBERT run
    "ntl-member-enc-rep": [
        dict(name="modernbert_large_pool", hf="answerdotai/ModernBERT-large",
             args=["--batch", "8", "--accum", "2", "--lr1", "2e-5", "--epochs1", "2", "--lr2", "1e-5", "--epochs2", "3",
                   "--max-len", "384", "--stage1-hours", "4.5"]),
        dict(name="deberta_v3_large_pool", hf="microsoft/deberta-v3-large",
             args=["--batch", "4", "--accum", "4", "--lr1", "2e-5", "--epochs1", "1", "--lr2", "1e-5", "--epochs2", "3",
                   "--max-len", "384", "--stage1-hours", "3.5"]),
    ],
}


# two-stage LoRA24 members of our own (phase 2 of the training track; the user cleared the GPU quota on 2026-10-26):
# Lucas Boesen's LoRA24 recipe with Dhruv Tibarewal's two stages, on a decoder family nlm-v27 lacks. Stage 1 tries
# the bases in order (fp16 health / memory check) and trains the first that passes; stage 2 is built from its output.
NLM_LABELS = sorted(LABELS)   # infer.py's head order
L24_S1_ARGS = ["--keep", "24", "--layers", "2,4,6,8,12,16,20,24", "--max-len", "384", "--tri", "2", "--accum", "2",
               "--lr", "1e-4", "--head-lr", "1e-3", "--warmup", "0.03", "--epochs", "2"]
L24_S2_ARGS = ["--keep", "24", "--layers", "2,4,6,8,12,16,20,24", "--max-len", "384", "--tri", "2", "--accum", "3",
               "--lr", "5e-5", "--head-lr", "5e-4", "--warmup", "0.1", "--epochs", "2"]
GLM4 = dict(tag="glm9v1", parts=["/glm4_9b_trunc25/"], args=L24_S1_ARGS, source=None, name="GLM-4-9B-0414",
            kernel="lucasboesen/name-that-llm-glm4-9b-trunc25")   # GLM-4-9B-0414 (MIT), Lucas Boesen's 25-layer copy
L24 = {
    "ntl-l24-probe": dict(stage=0, bases=[]),   # filled below: the next candidate bases
    "ntl-l24-mn8-s1": dict(stage=1, hours=10.0, bases=[
        dict(tag="mn8v1", parts=["/ministral-3/", "/ministral-3-8b-base-2512/"], args=L24_S1_ARGS,
             source="mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1"),
        dict(tag="ms7v1", parts=["/mistral-ai/mistral/", "/7b-v0.1-hf/"], args=L24_S1_ARGS,
             source="mistral-ai/mistral/pyTorch/7b-v0.1-hf/1")]),
}


def code_cell(source):
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None}


def notebook(intro, cells):
    return {"cells": [{"cell_type": "markdown", "source": intro, "metadata": {}}] + [code_cell(c) for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def write(kernel, nb, model_sources):
    out = HERE / kernel
    out.mkdir(exist_ok=True)
    (out / f"{kernel}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["name-that-llm"], "model_sources": model_sources,
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def filled(name, config):
    source = (HERE / name).read_text(encoding="utf-8").rstrip("\n")
    if source.count("CONFIG = None") != 1:
        raise SystemExit(f"{name} must contain 'CONFIG = None' once")
    return source.replace("CONFIG = None", "CONFIG = " + repr(config))  # Python literals (None, not null)


def build_full(kernel):
    fit = FULL[kernel]
    config = dict(kernel=kernel, kaggle_model=fit["kaggle_model"], model_source=fit["model_source"],
                  labels=list(LABELS), train_sha256=TRAIN_SHA256, seeds=fit["seeds"], worker_args=COMMON)
    intro = (f"# {kernel}\n\nFull fit of the phase-1 Mistral-7B member (LoRA r16 sequence classifier, 2 epochs) on "
             f"all 2,699 official rows, seeds {fit['seeds']} on the two GPUs, for a V28 fork that mixes its "
             f"probabilities in. Training kernel, not a submission.")
    worker = (HERE / "member_worker.py").read_text(encoding="utf-8").rstrip("\n")
    infer = (HERE / "mistral_infer.py").read_text(encoding="utf-8").rstrip("\n")
    write(kernel, notebook(intro, ["%%writefile /kaggle/working/member_worker.py\n" + worker,
                                   "%%writefile /kaggle/working/mistral_infer.py\n" + infer,
                                   filled("full_driver.py", config)]), [fit["model_source"]])
    print(f"built {kernel}: seeds {fit['seeds']}, worker args {' '.join(COMMON)}")


def build_enc(kernel):
    models = ENC[kernel]
    config = dict(kernel=kernel, labels=list(LABELS), train_sha256=TRAIN_SHA256,
                  folds="".join(map(str, bank_folds())), models=models)
    intro = (f"# {kernel}\n\nTwo-stage encoder members for a fork of nlm-v27 (Name That LLM): "
             + ", ".join(f"{m['hf']} ({m['args'][m['args'].index('--epochs1') + 1]} pool epochs)" for m in models)
             + ", one per GPU. Stage 1 on Dhruv Tibarewal's 120k "
             "public pool (nlm_pool_v5.parquet); stage 2: three epochs on the 2,699 official rows, once per outer fold "
             "of lightningv08's OOF bank (out-of-fold logits) and once on all rows (the saved model). Output in the "
             "layout of Dhruv's nlm-enc-v5 ModernBERT member. Training kernel, not a submission.")
    worker = (HERE / "enc_worker.py").read_text(encoding="utf-8").rstrip("\n")
    nb = notebook(intro, ["%%writefile /kaggle/working/enc_worker.py\n" + worker, filled("enc_driver.py", config)])
    write(kernel, nb, [])
    meta_path = HERE / kernel / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["dataset_sources"] = ["dhruvtibarewal/nlm-matched-clean"]
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: {[m['name'] for m in models]}")


L24["ntl-l24-mn8-s2"] = dict(stage=2, tag="mn8v1", s1="ntl-l24-mn8-s1",
                             source="mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1")
L24["ntl-l24-probe"]["bases"] = [GLM4, L24["ntl-l24-mn8-s1"]["bases"][1]]   # Ministral-3-8B passed in ntl-l24-mn8-s1
# GLM-4-9B passed ntl-l24-probe (7.5 h per pool epoch on T4 x2): two sessions reach the two pool epochs
L24["ntl-l24-glm9-s1"] = dict(stage=1, hours=10.0, bases=[GLM4])
L24["ntl-l24-glm9-s1b"] = dict(stage=1, hours=6.0, bases=[GLM4], init="ntl-l24-glm9-s1")
L24["ntl-l24-glm9-s2"] = dict(stage=2, tag="glm9v1", s1="ntl-l24-glm9-s1b", source=None, kernel=GLM4["kernel"])


def build_l24(kernel):
    spec = L24[kernel]
    config = dict(kernel=kernel, stage=spec["stage"], labels=NLM_LABELS, train_sha256=TRAIN_SHA256)
    if spec["stage"] == 0:
        config.update(bases=[{k: b[k] for k in ("tag", "parts", "args")} for b in spec["bases"]])
        sources = [b["source"] for b in spec["bases"] if b["source"]]
        intro = (f"# {kernel}\n\nProbe for two-stage LoRA24 members (Name That LLM): the fp16 health / memory / speed "
                 f"check of {', '.join(b['tag'] for b in spec['bases'])} on a T4, then a few minutes of two-GPU stage 1 "
                 "on a slice of Dhruv Tibarewal's 120k pool. Training kernel, not a submission.")
        datasets, kernels = ["dhruvtibarewal/nlm-matched-clean"], [b["kernel"] for b in spec["bases"] if b.get("kernel")]
    elif spec["stage"] == 1:
        config.update(hours=spec["hours"], bases=[{k: b[k] for k in ("tag", "parts", "args")} for b in spec["bases"]])
        sources = [b["source"] for b in spec["bases"] if b["source"]]
        names = " / ".join(b.get("name") or b["source"].split("/")[3] for b in spec["bases"])
        intro = (f"# {kernel}\n\nStage 1 of a two-stage LoRA24 member for a fork of nlm-v27 (Name That LLM): the first "
                 f"of {names} that passes an fp16 / memory check, truncated to 24 layers, LoRA r16 on every linear layer, "
                 f"two epochs (at most {spec['hours']} h) on Dhruv Tibarewal's 120k public pool (nlm_pool_v5.parquet) "
                 "with class weights, data-parallel over both T4s. Recipe: Lucas Boesen's public LoRA24 trainer with "
                 "Dhruv Tibarewal's two stages. Training kernel, not a submission.")
        datasets, kernels = ["dhruvtibarewal/nlm-matched-clean"], [b["kernel"] for b in spec["bases"] if b.get("kernel")]
        if spec.get("init"):   # a second session that continues an earlier stage-1 kernel to two pool epochs in all
            config.update(init_tag=spec["bases"][0]["tag"], epochs_total=2.0)
            kernels.append(f"omerfarukmerey/{spec['init']}")
            intro = (f"# {kernel}\n\nStage 1, continued: {spec['init']}'s {names} LoRA24 adapter and heads trained on "
                     f"Dhruv Tibarewal's 120k public pool up to two epochs in all (at most {spec['hours']} h, a new "
                     "triplet order and learning-rate cycle), data-parallel over both T4s. Training kernel, not a "
                     "submission.")
    else:
        config.update(tag=spec["tag"], folds="".join(map(str, inferred_folds())), gpu_jobs=["0,2,4", "1,3,full"],
                      args=L24_S2_ARGS)
        sources, datasets = [spec["source"]] if spec["source"] else [], []
        kernels = [f"omerfarukmerey/{spec['s1']}"] + ([spec["kernel"]] if spec.get("kernel") else [])
        intro = (f"# {kernel}\n\nStage 2 of the two-stage LoRA24 member {spec['tag']} (stage 1: {spec['s1']}): two epochs "
                 "on the 2,699 official rows, once per held-out fold of the split inferred from nlm-v27's own fold "
                 "adapters (out-of-fold predictions) and once on all rows, in the layout of Dhruv Tibarewal's "
                 "nlm-l24-adapters-v5. Training kernel, not a submission.")
    cells = ["%%writefile /kaggle/working/l24_worker.py\n" + (HERE / "l24_worker.py").read_text(encoding="utf-8").rstrip("\n")]
    if spec["stage"] == 2:
        nlm_oof = (HERE.parent / "nlm" / "nlm_oof.py").read_text(encoding="utf-8").rstrip("\n")
        cells.append("%%writefile /kaggle/working/nlm_oof.py\n" + nlm_oof)
    write(kernel, notebook(intro, cells + [filled("l24_driver.py", config)]), sources)
    meta_path = HERE / kernel / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    up = json.loads((HERE.parent / "nlm" / "upstream_nlm_v27.meta.json").read_text(encoding="utf-8"))
    meta.update(dataset_sources=datasets, kernel_sources=kernels, docker_image=up["docker_image"])  # infer.py's image
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: stage {spec['stage']}, models {sources}")


# TPU v3-8 probe (separate 20 h weekly quota): speed and memory of the LoRA24 recipe on Ministral-3-8B, see tpu_probe.py
TPU = {"ntl-tpu-probe": "mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1"}


def build_tpu(kernel):
    intro = (f"# {kernel}\n\nProbe: can Kaggle's TPU v3-8 (8 cores x 16 GB) train our two-stage LoRA24 members (Lucas "
             "Boesen's recipe on Ministral-3-8B-Base, 24 layers, LoRA r16, bf16 base) and how fast, one core and "
             "data-parallel over all eight? Random tokens at fixed shapes. Training-infrastructure kernel, not a submission.")
    probe = (HERE / "tpu_probe.py").read_text(encoding="utf-8").rstrip("\n")
    driver = (HERE / "tpu_driver.py").read_text(encoding="utf-8").rstrip("\n")
    write(kernel, notebook(intro, ["%%writefile /kaggle/working/tpu_probe.py\n" + probe, driver]), [TPU[kernel]])
    meta_path = HERE / kernel / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(enable_gpu=False, enable_tpu=True, machine_shape="Tpu1VmV38", competition_sources=[])
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: TPU v3-8, model {TPU[kernel]}")


# stage 1 on the TPU v3-8 (tpu_s1.py + tpu_l24_driver.py), continuing a GPU stage-1 output up to two pool epochs:
# 16 triplets per optimizer step over the 8 cores (the GPU run: 8, over two T4s with accumulation 2)
TPU_S1 = {"ntl-tpu-mn8-s1b": dict(init="ntl-l24-mn8-s1", init_tag="mn8v1", out_tag="mn8v2", hours=8.0,
                                  source="mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1")}
L24["ntl-l24-mn8-s2b"] = dict(stage=2, tag="mn8v2", s1="ntl-tpu-mn8-s1b",
                              source="mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1")


def build_tpu_s1(kernel):
    spec = TPU_S1[kernel]
    args = list(L24_S1_ARGS)
    del args[args.index("--epochs"):args.index("--epochs") + 2]   # the driver passes the epochs left to two
    args[args.index("--accum") + 1] = "1"
    config = dict(kernel=kernel, init_tag=spec["init_tag"], out_tag=spec["out_tag"], hours=spec["hours"],
                  epochs_total=2.0, args=args, labels=NLM_LABELS, train_sha256=TRAIN_SHA256)
    intro = (f"# {kernel}\n\nStage 1, continued on a TPU v3-8: {spec['init']}'s LoRA24 adapter and heads "
             f"({spec['init_tag']}) trained further on Dhruv Tibarewal's 120k public pool, up to two pool epochs in all "
             f"(at most {spec['hours']} h, a new triplet order and learning-rate cycle), data-parallel over the 8 TPU "
             f"cores, saved as {spec['out_tag']} for the GPU stage 2. Training kernel, not a submission.")
    cells = ["%%writefile /kaggle/working/l24_worker.py\n" + (HERE / "l24_worker.py").read_text(encoding="utf-8").rstrip("\n"),
             "%%writefile /kaggle/working/tpu_s1.py\n" + (HERE / "tpu_s1.py").read_text(encoding="utf-8").rstrip("\n"),
             filled("tpu_l24_driver.py", config)]
    write(kernel, notebook(intro, cells), [spec["source"]])
    meta_path = HERE / kernel / "kernel-metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(enable_gpu=False, enable_tpu=True, machine_shape="Tpu1VmV38",
                dataset_sources=["dhruvtibarewal/nlm-matched-clean"], kernel_sources=[f"omerfarukmerey/{spec['init']}"])
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: TPU v3-8, continues {spec['init']} into {spec['out_tag']}, args {' '.join(args)}")


def inferred_folds():
    """The fold split ntl-nlm-oof inferred from nlm-v27's fold adapters (exp015), in train.csv order."""
    import numpy as np
    from ntl.paths import REPO_ROOT
    return np.load(REPO_ROOT / "outputs" / "exp015_nlm_core_oof" / "nlm_oof.npz")["folds"]


def build(kernel):
    if kernel in TPU:
        return build_tpu(kernel)
    if kernel in TPU_S1:
        return build_tpu_s1(kernel)
    if kernel in L24:
        return build_l24(kernel)
    if kernel in FULL:
        return build_full(kernel)
    if kernel in ENC:
        return build_enc(kernel)
    pilot = PILOTS[kernel]
    folds = bank_folds()
    config = dict(kernel=kernel, kaggle_model=pilot["kaggle_model"], hf_repo=pilot["hf_repo"],
                  hf_revision=pilot["hf_revision"], labels=list(LABELS), train_sha256=TRAIN_SHA256,
                  folds="".join(map(str, folds)), gpu_folds=["0,2,4", "1,3"], worker_args=COMMON)
    driver = (HERE / "member_driver.py").read_text(encoding="utf-8").rstrip("\n")
    if driver.count("CONFIG = None") != 1:
        raise SystemExit("member_driver.py must contain 'CONFIG = None' once")
    driver = driver.replace("CONFIG = None", "CONFIG = " + repr(config))  # Python literals (None, not null)
    worker = (HERE / "member_worker.py").read_text(encoding="utf-8").rstrip("\n")
    intro = (f"# {kernel}\n\nPhase-1 pilot of a new ensemble member for Name That LLM: a LoRA sequence classifier "
             f"trained on the 2,699 official rows only, one model per outer fold of lightningv08's published OOF "
             f"bank, so its out-of-fold predictions can be scored against that bank. Training kernel, not a submission.")
    nb = {"cells": [{"cell_type": "markdown", "source": intro, "metadata": {}},
                    code_cell("%%writefile /kaggle/working/member_worker.py\n" + worker),
                    code_cell(driver)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = HERE / kernel
    out.mkdir(exist_ok=True)
    (out / f"{kernel}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": ["name-that-llm"], "model_sources": pilot["model_sources"],
            "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: folds {sorted(set(config['folds']))}, sizes {[config['folds'].count(str(k)) for k in range(5)]}")


if __name__ == "__main__":
    for name in sys.argv[1:] or [*PILOTS, *FULL, *ENC, *L24]:
        build(name)
