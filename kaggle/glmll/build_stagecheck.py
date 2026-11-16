"""End-to-end check of a build_lv694.py variant's GLM stage on Kaggle (T4 x2): the variant's own stage cells (head,
segio / features / run_pipeline, the patched v1_stage.py, extra_features.py, the segment and binary setup) run on the
120 train passages our stand-in stack ranks most GLM-like, with the stand-in's out-of-fold logits as final-logits.npz;
the predictions must match the ones exp050 computes offline from our train likelihoods for the same rows. The commit
of a variant never runs the stage (3-row preview), so this is its only full run before the hidden rerun.

    python kaggle/glmll/build_stagecheck.py <variant> [kernel name]   # default kaggle/glmll/ntl-stagecheck/
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R / "experiments"), str(R / "kaggle" / "nlm"), str(R / "src")]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import build_lv694 as B  # noqa: E402
import exp050_glm_ll_head as X  # noqa: E402
from build_l681 import text  # noqa: E402
from ntl.data import load_train  # noqa: E402

NAME = "ntl-stagecheck"
N = 120

CHECK = '''import json, subprocess, sys
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
       '--report', '/kaggle/working/report.json', '--bin', str(V1_BIN),
       '--pipeline', str(V1_DIR / 'run_pipeline.py'), '--flash-segs', json.dumps(V1_FLASH),
       '--glm53-segs', json.dumps(V1_GLM53), '--model', str(V1_DIR / 'll_model.json'), '--work', str(work / 'v1-work'),
       '--outdir', '/kaggle/working/v1', '--seq-batch', '8', '--ubatch-div', '1', '--max-cover', '1.0',
       '--min-texts', '10']
subprocess.run(cmd, env=V1_ENV, check=True)
rep = json.loads(Path('/kaggle/working/report.json').read_text())
print({{k: v for k, v in rep.items() if not isinstance(v, dict)}})
got = pd.read_parquet('/kaggle/working/check.parquet').set_index('id').loc[CHECK['ids'], 'model'].tolist()
agree = float(np.mean([a == b for a, b in zip(got, CHECK['expected'])]))
print(f"stage vs offline predictions: {{agree:.3f}} agree on {{len(got)}} rows; status {{rep['status']}}", flush=True)
assert rep['status'] == 'applied' and agree >= 0.95
'''


def build(variant, NAME=NAME):
    head = B.VARIANTS[variant][0]
    model = json.loads((B.HEADS / f"{head}.json").read_text(encoding="utf-8"))
    nb = json.loads((B.HERE / variant / f"{variant}.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    pick = lambda start: next(c for c in cells if text(c).startswith(start))
    stage_cells = [pick(B.LL_CELL)] + [pick(f"%%writefile /kaggle/temp/v1/{f}") for f in
                                       ("segio.py", "features.py", "run_pipeline.py", "v1_stage.py")]
    if model.get("extra"):
        stage_cells.append(pick("%%writefile /kaggle/temp/v1/extra_features.py"))
    stage_cells.append(pick("# GLM segment paths come from the pinned Kaggle input versions."))

    sub_ids, z, expected = offline_expected(model)
    payload = json.dumps({"ids": sub_ids, "labels": X.LABELS, "logits": z.round(6).tolist(), "expected": expected})

    setup = ("import glob, os, pandas as pd\nos.makedirs('/kaggle/temp', exist_ok=True)\nfrom pathlib import Path\n"
             "TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)\n"
             "DATA = TRAIN   # the setup cell tokenizes DATA's first three texts as a smoke test\n")
    out_cells = [{"cell_type": "markdown", "metadata": {}, "source":
                  f"# {NAME}\n\nThe GLM stage of `{variant}` (head `{head}`) run end to end on {N} train passages; its "
                  "predictions must match the offline ones of exp050."},
                 {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": setup}]
    out_cells += [dict(c, outputs=[], execution_count=None) for c in stage_cells]
    out_cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                      "source": CHECK.format(payload=payload)})
    write(NAME, dict(nb, cells=out_cells), variant, len(out_cells), z, expected)


def offline_expected(model):
    """The stand-in stack's out-of-fold logits on the N most GLM-like train rows and the predictions exp050's
    features of those rows give with this head (the stage must reproduce them)."""
    tr = load_train()
    ids, texts = tr.id.astype(str).tolist(), tr.text.fillna("").astype(str).tolist()
    E = X.E
    nlm = np.load(E.NLM_OOF)
    m = {k: nlm[f"oof_{k}"].astype(np.float64) for k in ("q14v5", "q8v5", "ll8v5")}
    m["svc4"] = np.load(E.OUT / "svc4_oof.npy")
    zl = np.log(np.clip(E.sinkhorn(E.blend(m, E.CORE)), 1e-12, None))
    rows = np.sort(np.argsort(-np.exp(zl[:, X.GLM]).sum(1), kind="stable")[:N])
    names = [c.split(":", 1)[1] for c in X.HIS["features"] if c.startswith("flash:")]
    sub_ids, sub_texts = [ids[i] for i in rows], [texts[i] for i in rows]
    sf, _ = X.load_scores("flash", sub_ids)
    sg, _ = X.load_scores("glm53", sub_ids)
    # every column the runtime can build, by name; the logistic head and the trees each pick theirs (as member_logp)
    Xo, cols = X.design(X.features(sf, sub_texts, 160), X.features(sg, sub_texts, 160), names, X.HIS["diff_features"])
    D = pd.DataFrame(Xo, columns=cols)
    ex = model.get("extra") or {}
    if ex.get("pair"):
        pr = X.pair_features(sf, sg, 160)
        D = D.join(pd.DataFrame(pr).add_prefix("pair:"))
    if "ref" in ex:
        tag = ex["ref"]["tag"]
        rr = pd.DataFrame(X.features(X.load_ref(tag, sub_ids), sub_texts, 160))[names]
        D = D.join(rr.add_prefix(f"{tag}:"))
        for c in model.get("ref_diff", []):
            D[f"r:{c}"] = D[f"glm53:{c}"] - D[f"{tag}:{c}"]
    A = D[model["features"]].to_numpy(np.float64)
    ll = X.log_softmax(((A - np.array(model["mean"])) / np.array(model["scale"])) @ np.array(model["coef"]).T
                       + np.array(model["intercept"]), 1)
    if model.get("gbm"):
        g = model["gbm"]
        Xg = (D[g["features"]].to_numpy(np.float64) - np.array(g["mean"])) / np.array(g["scale"])
        ll = X.mix(ll, X.gbm_logp(g, Xg), g["mix"])
    z = zl[rows]
    w = model["W"] / X.match_temperature(z, model["entropy_ref"])
    return sub_ids, z, np.asarray(X.LABELS)[(z + w * ll).argmax(1)].tolist()


def write(NAME, nb_out, variant, n_cells, z, expected):
    out = Path(__file__).resolve().parent / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((B.HERE / "upstream_lightningv08_694_v42.meta.json").read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [d for d in up["datasetDataSources"] if "glm53" in d],
            "competition_sources": ["name-that-llm"], "kernel_sources": ["estopadilla/ntl-glm-likelihood-train"],
            "model_sources": [s for s in up["modelDataSources"] if "base" in s], "docker_image": up["dockerImage"],
            "machine_shape": up["machineShape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {NAME} for {variant}: {n_cells} cells, {N} rows, expected changes vs the stack "
          f"{int((np.asarray(expected) != np.asarray(X.LABELS)[z.argmax(1)]).sum())}")


if __name__ == "__main__":
    build(*sys.argv[1:3])
