"""End-to-end check of a build_graft.py variant's GLM stage on Kaggle (T4 x2), like build_stagecheck.py for the 0.694
forks: the variant's own V1 cells (config, the base64 payload with the patched v1_stage.py and our head, binary and
segment setup, GPU smoke test) and then v1_stage.py on the 120 train passages our stand-in stack ranks most GLM-like,
with the stand-in's out-of-fold logits as final-logits.npz. No time box (deadline 4 h away), every row scored, the
25% sanity guard off (these rows are GLM-heavy by construction); the predictions must match exp050's offline ones.

    python kaggle/glmll/build_stagecheck_graft.py <variant> [kernel name]   # default kaggle/glmll/ntl-stagecheck-graft/
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "experiments"), str(R / "kaggle" / "nlm"), str(R / "kaggle" / "glmll"), str(R / "src")]

import build_graft as G  # noqa: E402
import build_lv694 as B  # noqa: E402
import build_stagecheck as S  # noqa: E402
import exp050_glm_ll_head as X  # noqa: E402
from build_l681 import text  # noqa: E402

NAME = "ntl-stagecheck-graft"

CHECK = '''import json, subprocess, sys, time
import numpy as np, pandas as pd
from pathlib import Path
CHECK = json.loads({payload!r})
work = Path('/kaggle/temp/check'); work.mkdir(parents=True, exist_ok=True)
train = pd.read_csv(TRAIN, dtype={{'id': str}}).set_index('id').loc[CHECK['ids']].reset_index()
train[['id', 'text']].to_csv(work / 'test.csv', index=False)
np.savez(work / 'final-logits.npz', ids=np.array(CHECK['ids']), labels=np.array(CHECK['labels']),
         logits=np.array(CHECK['logits']))
cmd = [sys.executable, str(V1_DIR / 'v1_stage.py'), '--test-csv', str(work / 'test.csv'),
       '--final-logits', str(work / 'final-logits.npz'), '--out', '/kaggle/working/check.parquet',
       '--report', '/kaggle/working/report.json', '--bin', str(V1_BIN), '--pipeline', str(V1_DIR / 'run_pipeline.py'),
       '--flash-segs', str(V1_FLASH / '*.gguf'), '--glm53-segs', str(V1_GLM53 / '*.gguf'),
       '--model', str(V1_DIR / 'll_model.json'), '--work', str(work / 'v1-work'), '--outdir', '/kaggle/working/v1',
       '--deadline-mono', repr(time.monotonic() + 4 * 3600), '--seq-batch', str(V1_SEQ_BATCH),
       '--ubatch-div', str(V1_UBATCH_DIV), '--max-cover', '1.0', '--min-texts', '10', '--max-changed', '1.0']
for _k, _v in V1_COST.items():
    cmd += ['--' + _k.replace('_', '-'), repr(_v)]
subprocess.run(cmd, env=V1_ENV, check=True)
rep = json.loads(Path('/kaggle/working/report.json').read_text())
print({{k: v for k, v in rep.items() if not isinstance(v, dict)}})
got = pd.read_parquet('/kaggle/working/check.parquet').set_index('id').loc[CHECK['ids'], 'model'].tolist()
agree = float(np.mean([a == b for a, b in zip(got, CHECK['expected'])]))
print(f"stage vs offline predictions: {{agree:.3f}} agree on {{len(got)}} rows; status {{rep['status']}}", flush=True)
assert rep['status'] == 'applied' and agree >= 0.90
'''


def build(variant, NAME=NAME):
    head = G.VARIANTS[variant]
    model = json.loads((B.HEADS / f"{head}.json").read_text(encoding="utf-8"))
    nb = json.loads((G.HERE / variant / f"{variant}.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    pick = lambda start: next(c for c in cells if text(c).startswith(start))
    stage_cells = [pick("# ---- ours, V1: likelihood under GLM-5.3-Flash and GLM-5.3"), pick(G.SETUP)]
    sub_ids, z, expected = S.offline_expected(model)
    payload = json.dumps({"ids": sub_ids, "labels": X.LABELS, "logits": z.round(6).tolist(), "expected": expected})
    setup = ("import glob, os, time\nimport numpy as np, pandas as pd\nos.makedirs('/kaggle/temp', exist_ok=True)\n"
             "TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)\n"
             "DATA = os.path.dirname(TRAIN)   # the V1 cells smoke-test the first three rows of DATA/test.csv\n")
    out_cells = [{"cell_type": "markdown", "metadata": {}, "source":
                  f"# {NAME}\n\nThe GLM stage of `{variant}` (head `{head}`, feiyu26's graft) run end to end on {S.N} "
                  "train passages; its predictions must match the offline ones of exp050."},
                 {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": setup}]
    out_cells += [dict(c, outputs=[], execution_count=None) for c in stage_cells]
    out_cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                      "source": CHECK.format(payload=payload)})
    out = Path(__file__).resolve().parent / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(dict(nb, cells=out_cells), indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((G.HERE / "upstream_feiyu26_graft_1114.meta.json").read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [d for d in up["dataset_sources"] if "glm53" in d],
            "competition_sources": ["name-that-llm"],
            "kernel_sources": [k for k in up["kernel_sources"] if "glm-likelihood-train" in k],
            "model_sources": [], "docker_image": up["docker_image"], "machine_shape": up["machine_shape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {NAME} for {variant}: {len(out_cells)} cells, {S.N} rows, datasets {meta['dataset_sources']}, "
          f"kernels {meta['kernel_sources']}")


if __name__ == "__main__":
    build(*sys.argv[1:3])
