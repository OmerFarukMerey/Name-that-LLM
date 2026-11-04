"""Build the exp029 kernels: ATLAS-8B adapters trained with kaggle/atlas/atlas_train.py on T4 x2 in mercurydust's V31
Docker image (the one infer.py runs ATLAS in), each holding out prompt fold 0 of the official train and scoring it.

    python kaggle/atlas/build_atlas_cv.py

ntl-atlas-cv-off: official rows only (Joseph Adamski's recipe, as far as his training_config describes it).
ntl-atlas-cv-ext: the same plus 500 passages per label from crtproduction/name-that-llm-open-pool (real-model public
passages, already filtered against the train prompts; gemini-3.7 rows flagged for API routing dropped). Same seed, so
the pair differs only in the data. ntl-atlas-full-ext: every official row plus the same external share (no hold-out),
the bundle an LB variant would load in place of Joseph's. ntl-atlas-cv-ext1000: fold 0 with 1,000 passages per label. ntl-atlas-cv-ext-f1: -ext on fold 1.
-s18: the same with seed 18 (another open-pool sample, initialisation and order), for a two-bundle ATLAS. Output: kaggle/atlas/<kernel>/ (notebook + kernel-metadata.json).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NLM = HERE.parent / "nlm"
SCRIPT = (HERE / "atlas_train.py").read_text(encoding="utf-8")
DRIVER = r'''import glob, os, subprocess, sys
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
BUNDLE = os.path.dirname(next(x for x in glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True) if 'ntl-atlas-qwen3-8b-v1' in x))
BASE = next(os.path.dirname(x) for x in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
            if '/qwen-3/' in x.lower() and '/8b-base/' in x.lower())
EXTRA = next(iter(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)), '')
print(DATA, BUNDLE, BASE, EXTRA, sep='\n')
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True)
cmd = [sys.executable, 'atlas_train.py', '--bundle', BUNDLE, '--base', BASE, '--train', DATA + '/train.csv',
       '--out', '/kaggle/working/atlas_out', '--holdout', '{holdout}', '--epochs', '5', '--seed', '{seed}', '--max_hours', '{max_hours}']
if {ext}:
    assert EXTRA, 'open pool not attached'
    cmd += ['--extra', EXTRA, '--extra_per_label', '{per_label}']
subprocess.run(cmd, check=True)
'''
KERNELS = {"ntl-atlas-cv-off": dict(ext=False, per_label=0, max_hours=9.5, holdout=0),
           "ntl-atlas-cv-ext": dict(ext=True, per_label=500, max_hours=9.5, holdout=0),
           # 2026-10-31: -ext was +0.065 alone on fold 0 (0.406 vs 0.341); does twice the external share add more?
           "ntl-atlas-cv-ext1000": dict(ext=True, per_label=1000, max_hours=9.5, holdout=0),
           # 2026-10-31: the power mix's +0.015 rests on fold 0's 540 rows; the 500/label run again on fold 1
           "ntl-atlas-cv-ext-f1": dict(ext=True, per_label=500, max_hours=9.5, holdout=1),
           # 2026-10-31: on fold 0 the mean of the 500 and 1,000/label retrains is 0.452 alone (0.406, 0.404 each; they
           # agree on 49% of argmaxes); the same recipe with another seed (so another open-pool sample) on fold 1 tests a
           # two-bundle ATLAS on a second fold, and -full-ext-s18 is that second bundle for the LB
           "ntl-atlas-cv-ext-f1-s18": dict(ext=True, per_label=500, max_hours=9.5, holdout=1, seed=18),
           # pushed only if -ext beats -off on fold 0: every official row plus the same external share, for the LB
           "ntl-atlas-full-ext": dict(ext=True, per_label=500, max_hours=10.5, holdout=-1),
           "ntl-atlas-full-ext-s18": dict(ext=True, per_label=500, max_hours=10.5, holdout=-1, seed=18)}


def main():
    meta_in = json.loads((NLM / "upstream_mercury_v31.meta.json").read_text(encoding="utf-8"))
    for kernel, kw in KERNELS.items():
        cells = ["%%writefile atlas_train.py\n" + SCRIPT, DRIVER.replace("{ext}", str(kw["ext"]))
                 .replace("{per_label}", str(kw["per_label"])).replace("{max_hours}", str(kw["max_hours"]))
                 .replace("{holdout}", str(kw["holdout"])).replace("{seed}", str(kw.get("seed", 17)))]
        nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 4}
        out = HERE / kernel
        out.mkdir(exist_ok=True)
        (out / f"{kernel}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
                "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["josephadamski91/ntl-atlas-qwen3-8b-v1", "crtproduction/name-that-llm-open-pool"],
                "kernel_sources": [], "competition_sources": ["name-that-llm"],
                "docker_image": meta_in["docker_image"], "machine_shape": "NvidiaTeslaT4", "is_private": True,
                "keywords": [], "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"]}
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
