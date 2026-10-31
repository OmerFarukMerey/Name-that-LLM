"""Build ntl-gsq-check: exp027's GSQ stage (lightningv08's Qwen3.8-27B GSQ3 readout through build_nlm's GSQ_STAGE and
his five gsq/ files, verbatim) on 240 random train rows, on Kaggle's T4 x2 in mercurydust's V31 Docker image (his too).

    python kaggle/nlm/build_gsq_check.py

It reports the stage's wall time (wheel install, local copy of the 11.85 GB base, two workers' load and scoring) and the
workers' load times from their reports, for the hidden rerun's budget (~4,000 rows), plus the in-sample accuracy of the
plain head (his final weights saw all of train, so this only shows that labels and rows line up).
Output: kaggle/nlm/ntl-gsq-check/ (notebook + kernel-metadata.json).
"""
import json

import build_nlm as B

KERNEL = "ntl-gsq-check"
CHECK = r'''import json, re, time
import pandas as pd
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train = pd.read_csv(DATA + '/train.csv', dtype={'id': str})
rng = np.random.RandomState(0); sel = np.sort(rng.choice(len(train), 240, replace=False))
q = train.iloc[sel].reset_index(drop=True); q['text'] = q.text.fillna('').astype(str)
t = time.time(); R = run_gsq(q, '/kaggle/working/gsq_check.npz', 7200); R['wall'] = round(time.time() - t)
z = np.load('/kaggle/working/gsq_check.npz', allow_pickle=False)
assert z['ids'].astype(str).tolist() == q.id.tolist()
labs = z['labels'].astype(str).tolist(); R['in_sample_accuracy'] = float((np.array(labs)[z['plain_logits'].argmax(1)] == q.model.values).mean())
R['chars_mean'] = float(q.text.str.len().mean()); R['chars_test_mean'] = float(pd.read_csv(DATA + '/test.csv').text.str.len().mean())
log = open('/kaggle/working/gsq_stage.log').read()
R['load_seconds'] = [round(json.loads(l)['load_report']['seconds']) for l in log.splitlines() if l.startswith('{"worker"')]
print(log[-3000:]); print(json.dumps(R)); json.dump(R, open('/kaggle/working/gsq_check.json', 'w'), indent=2)
'''


def main():
    cells = [s for _, s in B.lv08_gsq_cells()] + [CHECK]
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    out = B.HERE / KERNEL
    out.mkdir(exist_ok=True)
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta_in = json.loads((B.HERE / "upstream_mercury_v31.meta.json").read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [B.LV08_DATASET], "kernel_sources": [], "competition_sources": ["name-that-llm"],
            "docker_image": meta_in["docker_image"], "machine_shape": "NvidiaTeslaT4", "is_private": True,
            "keywords": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", out, len(cells), "cells")


if __name__ == "__main__":
    main()
