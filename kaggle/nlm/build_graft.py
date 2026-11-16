"""Forks of feiyu26's public NTL 0684 v1 graft (pulled 2026-11-14), the top of the score-sorted Code tab that day.

The graft = the public 0.684 stack (Atlas V9 clean LoRA + Qwen3-Embedding multidepth + GSQ-27B 3%, native share 0.40,
comp_w 0.15) followed by estopadilla's GLM-5.3 / GLM-5.3-Flash likelihood stage exactly as in his 0.689: the V1
payload (run_pipeline.py, segio.py, features.py, v1_stage.py, ll_model.json) is a base64 tar.gz byte-identical to his
(kaggle/glmll/v1src), time-boxed (8.8 h, at most 30% of the test).

Knobs:
    head  a likelihood head refit by experiments/exp050_glm_ll_head.py (outputs/exp050_glm_ll_head/heads/<head>.json)
          in place of his ll_model.json, with build_lv694.py's v1_stage.py patches (extra evidence, boosted trees, his
          head as the fallback) and kaggle/glmll/extra_features.py added to the payload
    cost  the stage's time-estimate constants (V1_COST, the config cell). The graft's 100 s + 0.0225 / 0.0442 s per
          token row overestimate ~40% at test scale; COST_FIT is ~20% above what three runs measured (our 160-cut
          train runs, 362,706 rows; ntl-stagecheck-graft, 18,794 rows; the graft author's 360-text run, 47,098 rows:
          Flash ~260 s + 0.0136 s / row, GLM-5.3 ~690 s + 0.022 s / row), so the stage covers ~25-30% instead of ~19%

    python kaggle/nlm/build_graft.py [variant ...]
"""
import base64
import hashlib
import io
import json
import re
import sys
import tarfile
from pathlib import Path

import build_lv694 as L
from build_l681 import text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_feiyu26_graft_1114.ipynb"
UPSTREAM_SHA = "00a0ee5b4695490c"
URL = "https://www.kaggle.com/code/feiyu26/ntl-0684-v1-graft"
KIRO_URL = "https://www.kaggle.com/code/lumibee/ntl-kiro-v44-apex-grandmaster"
UPSTREAMS = {   # name: (notebook, sha256 prefix, meta file, page header)
    "graft": (UPSTREAM, UPSTREAM_SHA, "upstream_feiyu26_graft_1114.meta.json",
              f"feiyu26's public [NTL 0684 v1 graft]({URL}) (pulled 2026-11-14)"),
    "kiro44": (HERE / "upstream_lumibee_kiro44_1115.ipynb", "652ed3efa9054c16", "upstream_lumibee_kiro44_1115.meta.json",
               f"lumibee's public [NTL Kiro v44 Apex Grandmaster]({KIRO_URL}) (pulled 2026-11-15; public 0.701 = the "
               "graft + path sorting, RUN_LIMIT 7.0 h, estopadilla's V1_COST, a post-stage sibling residual and "
               "typographic invariants)"),
}
PAYLOAD_SHA = "b0bb99ed84e417a8249600fb28f1f69c21641659510bc7ede15b3a73a05a785d"
SETUP = "# ---- ours, V1 setup and checks"
FILES = ("run_pipeline.py", "segio.py", "features.py", "v1_stage.py", "ll_model.json")

GCAP_URL = "https://www.kaggle.com/code/dhrubangtalukdar/transformer-ensemble-gda-score-0-680"
# dhrubangtalukdar's group cap (public 0.698 = the 0.684 stack + this, no GLM stage), his code: inside each test
# prompt group of 2..18 rows a class may claim at most one response; soft (strength 0.5), before the Sinkhorn step, so
# final-logits.npz (the GLM stage's and the post-stage cell's input) carries it
GCAP = [("p.add_argument('--group_sim', type=float, default=0.70)\n",
         "p.add_argument('--group_sim', type=float, default=0.70)\np.add_argument('--group_cap', type=float, default=0.5)\n"),
        ("def decide_scores(P):\n    Q = P.copy()\n",
         "def group_cap_probs(P, groups, strength, iters=25, lo=2, hi=18):\n"
         "    # A prompt has at most one response per model, so inside a group of 2..18 rows a class whose\n"
         "    # summed probability exceeds 1 is over-claimed: shrink it, renormalise rows, repeat. Soft (strength<1).\n"
         "    if strength <= 0:\n        return P\n"
         "    groups = np.asarray(groups)\n    size = np.bincount(groups)[groups]\n    Q = P.copy()\n"
         "    for g in np.unique(groups[(size >= lo) & (size <= hi)]):\n"
         "        ii = np.flatnonzero(groups == g)\n"
         "        q = np.clip(P[ii], 1e-12, None)\n        q = q / q.sum(1, keepdims=True)\n"
         "        for _ in range(iters):\n"
         "            q = q / np.maximum(1.0, q.sum(0, keepdims=True))\n            q = q / q.sum(1, keepdims=True)\n"
         "        Q[ii] = q\n"
         "    out = np.exp((1 - strength) * np.log(np.clip(P, 1e-12, None)) + strength * np.log(np.clip(Q, 1e-12, None)))\n"
         "    return out / out.sum(1, keepdims=True)\n\n\n"
         "def decide_scores(P):\n    P = group_cap_probs(P, Gte, a.group_cap)\n    Q = P.copy()\n")]
GCAP_CMD = ("'--sinkhorn', '0.30', '--sinkhorn_target_mix', '0.35',\n",
            "'--sinkhorn', '0.30', '--sinkhorn_target_mix', '0.35',\n       '--group_cap', '{}',\n")
# extra -> cap strength; 0.75: p72/gcap_sweep.py, better than 0.5 when test groups miss rows or are split, flat when
# 30% of their members are wrong (1.0 loses there)
GCAP_STRENGTH = {"gcap": "0.5", "gcap75": "0.75"}

VARIANTS = {   # name: head or (head, V1_COST[, upstream[, extras]])
    "ntl-graft-gbm": "gbmpair",
    "ntl-graft-gbm-cost": ("gbmpair", "COST_FIT"),
    "ntl-kiro-gbm": ("gbmpair", None, "kiro44"),
    "ntl-kiro-gbm-gcap": ("gbmpair", None, "kiro44", ("gcap",)),
    "ntl-graft-gbm-gcap": ("gbmpair", None, "graft", ("gcap",)),
    "ntl-kiro-gbm-gcap75": ("gbmpair", None, "kiro44", ("gcap75",)),
}
COST_LINE = "V1_COST = dict(fixed_s_flash=100.0, row_s_flash=0.0225, fixed_s_glm53=100.0, row_s_glm53=0.0442)\n"
COST_FIT = "V1_COST = dict(fixed_s_flash=400.0, row_s_flash=0.0165, fixed_s_glm53=900.0, row_s_glm53=0.0265)\n"
CONFIG = "# ---- ours, V1: likelihood under GLM-5.3-Flash and GLM-5.3"


def patched_stage(src):
    """build_lv694.py's v1_stage.py edits for a head with extra evidence and trees (same anchors in his file)."""
    for a, b in (("    blocks = [feats[t].add_prefix(f'{t}:') for t in ('flash', 'glm53')]\n",
                  "    blocks = [feats[t].add_prefix(f'{t}:') for t in feats]\n"),
                 ("    X = F[model['features']].to_numpy(np.float64)\n",
                  "    for c in model.get('ref_diff', []):   # exp050: GLM-5.3 minus the reference model\n"
                  "        F[f'r:{c}'] = F[f'glm53:{c}'] - F[model['extra']['ref']['tag'] + ':' + c]\n"
                  "    X = F[model['features']].to_numpy(np.float64)\n"),
                 ("    return log_softmax(z, 1)\n\n\ndef main():\n",
                  "    out = log_softmax(z, 1)\n"
                  "    if model.get('gbm'):   # exp050: averaged with boosted trees on the same and extra columns\n"
                  "        import extra_features as XF\n"
                  "        g = model['gbm']\n"
                  "        Xg = (F[g['features']].to_numpy(np.float64) - np.asarray(g['mean'])) / np.asarray(g['scale'])\n"
                  "        assert np.isfinite(Xg).all(), 'non-finite tree features'\n"
                  "        out = np.log((1 - g['mix']) * np.exp(out) + g['mix'] * np.exp(XF.gbm_logp(g, Xg)))\n"
                  "    return out\n\n\n" + L.EXTRA_FUNCS),
                 ("    ll = member_logp(model, feats, sel_ids)", L.EXTRA_CALL)):
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    return src


def tar_gz(files):
    """Deterministic tar.gz (fixed order, mtime, owner) of {name: bytes}."""
    buf = io.BytesIO()
    import gzip
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as t:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def build(name):
    v = (VARIANTS[name],) if isinstance(VARIANTS[name], str) else VARIANTS[name]
    head, cost, up_name, extras = v[0], (v[1:2] or (None,))[0], (v[2:3] or ("graft",))[0], (v[3:4] or ((),))[0]
    up_nb, up_sha, up_meta, up_text = UPSTREAMS[up_name]
    raw = up_nb.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == up_sha, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
    st = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith(SETUP)]
    assert len(st) == 1
    src = st[0]["source"]
    m = re.search(r'V1_PAYLOAD = "([A-Za-z0-9+/=]+)"', src)
    old = base64.b64decode(m.group(1))
    assert hashlib.sha256(old).hexdigest() == PAYLOAD_SHA and src.count(PAYLOAD_SHA) == 1
    with tarfile.open(fileobj=io.BytesIO(old), mode="r:gz") as t:
        files = {mm.name: t.extractfile(mm).read() for mm in t.getmembers()}
    assert tuple(files) == FILES, tuple(files)
    for f in FILES:   # the payload is estopadilla's V1, unchanged
        assert files[f] == (HERE.parents[0] / "glmll" / "v1src" / f).read_bytes(), f

    model = json.loads((L.HEADS / f"{head}.json").read_text(encoding="utf-8"))
    assert model["format"] == "ntl-v1-ll-model-1" and len(model["classes"]) == 18
    files["ll_model.json"] = json.dumps(model).encode()
    if model.get("extra"):
        assert model["fallback"]["format"] == "ntl-v1-ll-model-1"
        files["v1_stage.py"] = patched_stage(files["v1_stage.py"].decode("utf-8")).encode("utf-8")
        files["extra_features.py"] = L.EXTRA_SRC.read_bytes()
    new = tar_gz(files)
    src = src.replace(m.group(1), base64.b64encode(new).decode("ascii")).replace(
        PAYLOAD_SHA, hashlib.sha256(new).hexdigest())
    st[0]["source"] = src
    if cost:
        cf = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith(CONFIG)]
        assert len(cf) == 1 and cf[0]["source"].count(COST_LINE) == 1
        cf[0]["source"] = cf[0]["source"].replace(COST_LINE, globals()[cost])

    cap = next((GCAP_STRENGTH[e] for e in extras if e in GCAP_STRENGTH), None)
    if cap:
        inf = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith("%%writefile infer.py")]
        run = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith("LAYERS = [2, 4")]
        assert len(inf) == 1 and len(run) == 1
        for a, b in GCAP:
            assert inf[0]["source"].count(a) == 1, a
            inf[0]["source"] = inf[0]["source"].replace(a, b)
        assert run[0]["source"].count(GCAP_CMD[0]) == 1
        run[0]["source"] = run[0]["source"].replace(GCAP_CMD[0], GCAP_CMD[1].format(cap))
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\n{up_text}; the V1 payload's "
                     f"likelihood head replaced by {head} (exp050: refit on our train likelihoods, averaged with boosted "
                     "trees on 54 + 12 per-token Flash - GLM-5.3 features; his head as the fallback)."
                     + ("" if not cost else " The stage's time-estimate constants are refit to measured GLM runs (~20% "
                        "above them), so it scores ~25-30% of the test where the graft's constants allow ~19%.")
                     + ("" if not cap else f" Plus dhrubangtalukdar's [group cap]({GCAP_URL}) (strength "
                        f"{cap}) in infer.py's decide_scores, so the stack's final logits (and the GLM stage) carry it.")})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / up_meta).read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": up["dataset_sources"], "competition_sources": up["competition_sources"],
            "kernel_sources": up["kernel_sources"], "model_sources": up["model_sources"],
            "docker_image": up["docker_image"], "machine_shape": up["machine_shape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256("".join(text(c) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(cells)} cells, payload {len(old)} -> {len(new)} bytes, code sha256 {sha}")


if __name__ == "__main__":
    for v in sys.argv[1:] or VARIANTS:
        build(v)
