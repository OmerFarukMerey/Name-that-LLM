"""Build the exp032 kernels: each loads one exp029 hold-out kernel's trained bundle (its output attached as a kernel
source) and rescores that fold with kaggle/atlas/atlas_rescore.py on T4 x2 in mercurydust's V31 Docker image, keeping
the token log-probabilities under every label adapter.

    python kaggle/atlas/build_atlas_rescore.py

Output: kaggle/atlas/<kernel>/ (notebook + kernel-metadata.json).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NLM = HERE.parent / "nlm"
SCRIPT = (HERE / "atlas_rescore.py").read_text(encoding="utf-8")
DRIVER = r'''import glob, os, subprocess, sys
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
BUNDLE = [os.path.dirname(x) for x in glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True)]
assert len(BUNDLE) == 1, BUNDLE
BASE = next(os.path.dirname(x) for x in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
            if '/qwen-3/' in x.lower() and '/8b-base/' in x.lower())
print(DATA, BUNDLE[0], BASE, sep='\n')
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True)
subprocess.run([sys.executable, 'atlas_rescore.py', '--bundle', BUNDLE[0], '--base', BASE, '--train', DATA + '/train.csv',
                '--out', '/kaggle/working/rescore'], check=True)
'''
KERNELS = {"ntl-atlas-rescore-f0": "omerfarukmerey/ntl-atlas-cv-ext",       # 500 open-pool passages per label, fold 0
           "ntl-atlas-rescore-f1": "omerfarukmerey/ntl-atlas-cv-ext-f1"}    # the same recipe, fold 1


def main():
    meta_in = json.loads((NLM / "upstream_mercury_v31.meta.json").read_text(encoding="utf-8"))
    for kernel, source in KERNELS.items():
        cells = ["%%writefile atlas_rescore.py\n" + SCRIPT, DRIVER]
        nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in cells],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 4}
        out = HERE / kernel
        out.mkdir(exist_ok=True)
        (out / f"{kernel}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb", "language": "python",
                "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": [], "kernel_sources": [source], "competition_sources": ["name-that-llm"],
                "docker_image": meta_in["docker_image"], "machine_shape": "NvidiaTeslaT4", "is_private": True,
                "keywords": [], "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1"]}
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("wrote", out)


if __name__ == "__main__":
    main()
