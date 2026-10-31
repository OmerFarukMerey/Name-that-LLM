"""Build the out-of-fold kernels for fold-averaged LoRA24 members, which experiments/exp015_nlm_core_oof.py reads.

    PYTHONPATH=src python kaggle/nlm/build_oof.py [ntl-nlm-oof | ntl-nlm-oof-q25]

A two-cell notebook: nlm_oof.py written by %%writefile, then a cell that finds the inputs the way nlm-v27's cell 0
does and runs it. Private, internet off, upstream nlm-v27's Docker image, so the features match the submission's.

- ntl-nlm-oof: nlm-v27's three members (Dhruv's v5 adapters, Kaggle's Qwen3 bases, Llama-3.1-8B from avazaku's public
  cache as in ntl-nlm27-mb). The GroupKFold split of train.csv by prompt_id (sklearn's, no shuffle) is passed as a
  hypothesis for the unpublished fold split; the kernel infers the split from the adapters themselves and reports how
  well the two agree. Output: nlm_oof.npz. About 45 GPU minutes (it took 1.5 h).
- ntl-nlm-oof-q25: Dhruv's newer Qwen2.5-14B member (dataset nlm-q25-v5, five fold adapters and a full fit, LoRA on 32
  layers) on the split ntl-nlm-oof inferred, which is passed in. Its 8 pooled depths are not published, so the kernel
  finds them from the full-data adapter first. Output: nlm_oof_q25.npz, enc/q25_14v5/oof.npz, the depth search.
"""
import json
import sys

import numpy as np
from sklearn.model_selection import GroupKFold

from ntl.data import load_train
from ntl.paths import REPO_ROOT

HERE = REPO_ROOT / "kaggle" / "nlm"
SHARDS = {"model-00001-of-00004.safetensors": 4976698672, "model-00002-of-00004.safetensors": 4999802720,
          "model-00003-of-00004.safetensors": 4915916176, "model-00004-of-00004.safetensors": 1168138808}
INFERRED = REPO_ROOT / "outputs" / "exp015_nlm_core_oof" / "nlm_oof.npz"   # ntl-nlm-oof's output (folds)
PRELUDE = """import glob, json, os
import numpy as np
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
def mdir(*parts):
    c = [os.path.dirname(x) for x in cfgs if all(p in x.lower() for p in parts)]
    assert c, parts; return c[0]
def adir(name):
    h = glob.glob(f'/kaggle/input/**/{name}/heads.pt', recursive=True); assert h, name; return os.path.dirname(h[0])
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
L = [2, 4, 6, 8, 12, 16, 20, 24]
"""


def digits(folds):
    return "".join(map(str, folds))


def nlm_oof():
    train = load_train()
    folds = np.zeros(len(train), int)
    for k, (_, valid) in enumerate(GroupKFold(5).split(train, train.model, train.prompt_id)):
        folds[valid] = k
    run = PRELUDE + f"""B8, B14, BL = mdir('/qwen-3/', '/8b-base/'), mdir('/qwen-3/', '/14b-base/'), mdir('/llama-3.1/', '/8b/')
SHARDS = {SHARDS!r}
assert 'name-that-llm-verified-llama-asset-cache' in BL and all(os.path.getsize(os.path.join(BL, f)) == n for f, n in SHARDS.items()), BL
SPECS = [dict(name='q8v5', path=B8, adapters=[adir(f'q8v5_f{{k}}') for k in range(5)], keep=24, layers=L, bs=16, max_len=384),
         dict(name='q14v5', path=B14, adapters=[adir(f'q14v5_f{{k}}') for k in range(5)], keep=27, layers=L, bs=8, max_len=384),
         dict(name='ll8v5', path=BL, adapters=[adir(f'll8v5_f{{k}}') for k in range(5)], keep=24, layers=L, bs=16, max_len=384)]
print(B8, B14, BL, sep='\\n')
np.save('/kaggle/working/gkf_folds.npy', np.array([int(c) for c in '{digits(folds)}']))
!python nlm_oof.py --data {{DATA}} --specs '{{json.dumps(SPECS)}}' --out /kaggle/working/nlm_oof.npz --hypothesis /kaggle/working/gkf_folds.npy
"""
    intro = ("Out-of-fold predictions of the three fold-averaged LoRA24 members of "
             "[nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) (Dhruv Tibarewal) on train.csv, to "
             "score additions to that ensemble offline. Built with Llama (Llama 3.1 Community License).")
    sources = dict(dataset_sources=["dhruvtibarewal/nlm-l24-adapters-v5"],
                   kernel_sources=["avazaku/name-that-llm-verified-llama-asset-cache"],
                   model_sources=["qwen-lm/qwen-3/Transformers/8b-base/1", "qwen-lm/qwen-3/Transformers/14b-base/1"])
    return run, intro, sources, f"hypothesis fold sizes {np.bincount(folds).tolist()}"


def nlm_oof_q25():
    z = np.load(INFERRED)
    folds = z["folds"]
    assert np.array_equal(z["y"], load_train().model.map({l: i for i, l in enumerate(z["labels"])}).to_numpy())
    run = PRELUDE + f"""B25 = mdir('/qwen2.5/', '/14b/')
SPECS = [dict(name='q25_14v5', path=B25, adapters=[adir(f'q25_14v5_f{{k}}') for k in range(5)], full=adir('q25_14v5'),
              keep=32, search=True, bs=8, max_len=384)]
print(B25)
np.save('/kaggle/working/folds.npy', np.array([int(c) for c in '{digits(folds)}']))
!python nlm_oof.py --data {{DATA}} --specs '{{json.dumps(SPECS)}}' --out /kaggle/working/nlm_oof_q25.npz --folds /kaggle/working/folds.npy --enc_dir /kaggle/working/enc
"""
    intro = ("Out-of-fold predictions of Dhruv Tibarewal's two-stage Qwen2.5-14B LoRA24 member "
             "([nlm-q25-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-q25-v5)) on train.csv, on the fold split "
             "inferred from his other fold adapters, to score it as an addition to "
             "[nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) offline.")
    sources = dict(dataset_sources=["dhruvtibarewal/nlm-q25-v5"], kernel_sources=[],
                   model_sources=["qwen-lm/qwen2.5/Transformers/14b/1"])
    return run, intro, sources, f"folds from {INFERRED.name}: sizes {np.bincount(folds).tolist()}"


KERNELS = {"ntl-nlm-oof": nlm_oof, "ntl-nlm-oof-q25": nlm_oof_q25}


def main():
    kernel = sys.argv[1] if len(sys.argv) > 1 else "ntl-nlm-oof"
    run, intro, sources, note = KERNELS[kernel]()
    worker = (HERE / "nlm_oof.py").read_text(encoding="utf-8").rstrip("\n")
    cell = lambda s: {"cell_type": "code", "source": s, "metadata": {}, "outputs": [], "execution_count": None}
    nb = {"cells": [{"cell_type": "markdown", "source": f"# {kernel}\n\n{intro}", "metadata": {}},
                    cell("%%writefile nlm_oof.py\n" + worker), cell(run)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    up = json.loads((HERE / "upstream_nlm_v27.meta.json").read_text(encoding="utf-8"))
    out = HERE / kernel
    out.mkdir(exist_ok=True)
    (out / f"{kernel}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], **sources, "competition_sources": ["name-that-llm"],
            "docker_image": up["docker_image"], "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}: {note}")


if __name__ == "__main__":
    main()
