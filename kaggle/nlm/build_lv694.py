"""Forks of the public 0.694 (lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora, version 42).

The 0.694 = estopadilla's 0.689 (alexchilton's all-public stack + his GLM-5.3 / GLM-5.3-Flash likelihood stage) with
the stage inlined and given a fixed work allocation (the top texts by P(glm-5.3) + P(glm-5.3-flash) up to 130k token
rows, at most 20%), GSQ share 0.06 and Sinkhorn target mix 0.35. Its likelihood head (ll_model.json) is estopadilla's,
unchanged.

Knobs:
    head  a likelihood head refit by experiments/exp050_glm_ll_head.py on our own train likelihoods (kernels
          ntl-glmll-flash / ntl-glmll-glm53), outputs/exp050_glm_ll_head/heads/<head>.json, in place of his. A head
          may declare extra evidence (model["extra"]: per-token flash - glm53 contrasts, a Qwen3 base reference
          likelihood); the stage then computes it with kaggle/glmll/extra_features.py (written next to v1_stage.py)
          and falls back to his head (model["fallback"]) if that fails
    sib   the 0.683's sibling-style re-ranking on the stack's final logits (as build_g692.py)
    neg   the 0.683's typographic negative evidence (as build_g692.py)
    gsq   the GSQ-27B share (the 0.694's is 0.06; 0 skips that stage through the pipeline's own branch). The 11-14
          copies above the 0.693 lower it: heliosli 0.03, dhananjayk2609 0 (Areeb Ali's tests on the 0.684 code:
          3% 0.684, 6% 0.682, 12% 0.681); the GLM stage's work here is fixed, so only the stack blend changes

    python kaggle/nlm/build_lv694.py [variant ...]
"""
import hashlib
import json
import re
import sys
from pathlib import Path

from build_e683 import URL as E683_URL
from build_g692 import RUNTIME, e683_code
from build_l681 import patch, text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_lightningv08_694_v42.ipynb"
UPSTREAM_SHA = "719448ca19a27c34"
URL = "https://www.kaggle.com/code/lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora"
HEADS = HERE.parents[1] / "outputs" / "exp050_glm_ll_head" / "heads"
LL_CELL = "import json\nPath('/kaggle/temp/v1').mkdir(parents=True, exist_ok=True)\n"
STAGE = "%%writefile /kaggle/temp/v1/v1_stage.py\n"
EXTRA_SRC = HERE.parents[1] / "kaggle" / "glmll" / "extra_features.py"
EXTRA_FUNCS = """# exp050 (Name-that-LLM): extra evidence a refit head may declare in model['extra'], computed by extra_features.py
# (the code of the offline fit, written next to this file); any failure falls back to model['fallback'] (his head)
def token_scores(outdir, tag):
    z = np.load(outdir / f'{tag}_scores.npz', allow_pickle=False)   # its 'ids' is an object array: never read
    order = pd.read_csv(outdir / f'{tag}_features.csv', dtype={'id': str}, usecols=['id']).id.tolist()
    off, sc = z['offsets'], z['scores']
    assert len(off) == len(order) + 1
    return {k: sc[off[i]:off[i + 1]] for i, k in enumerate(order)}


def extra_blocks(spec, outdir, work, sub, sel_ids, cut):
    import extra_features as XF
    out = {}
    if spec.get('pair'):
        f, g = token_scores(outdir, 'flash'), token_scores(outdir, 'glm53')
        out['pair'] = pd.DataFrame(XF.pair_features([f[i] for i in sel_ids], [g[i] for i in sel_ids], cut),
                                   index=sel_ids)
    if spec.get('ref'):
        tag, size = spec['ref']['tag'], spec['ref']['size']
        cands = [f'/kaggle/input/models/qwen-lm/qwen-3/transformers/{size}/1', f'/kaggle/input/qwen-3/transformers/{size}/1']
        path = next((c for c in cands if os.path.isfile(os.path.join(c, 'config.json'))), None)
        path = path or os.path.dirname(sorted(glob.glob(f'/kaggle/input/**/{size}/**/config.json', recursive=True))[0])
        query, dest = work / 'ref_query.csv', work / f'{tag}_scores.npz'
        sub[['id', 'text']].to_csv(query, index=False)
        env = dict(os.environ)   # torch must not pick up ntl-segrun's CUDA 13 libraries
        env['LD_LIBRARY_PATH'] = ':'.join(d for d in env.get('LD_LIBRARY_PATH', '').split(':')
                                          if d and not os.path.isfile(os.path.join(d, 'libcublas.so.13')))
        with open(outdir / f'{tag}_ref.txt', 'w') as log:
            subprocess.run([sys.executable, str(Path(__file__).with_name('extra_features.py')), 'ref', path, str(query),
                            str(dest)], env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        with np.load(dest, allow_pickle=False) as z:
            rid, off, sc = z['ids'].astype(str).tolist(), z['offsets'], z['scores']
        assert rid == sub.id.tolist() and np.isfinite(sc).all()
        texts = sub.text.tolist()
        out[tag] = pd.DataFrame([XF.one(sc[off[i]:off[i + 1]], len(texts[i])) for i in range(len(rid))],
                                index=rid).loc[sel_ids]
    return out


def main():
"""
EXTRA_CALL = """    if model.get('extra'):
        try:
            feats.update(extra_blocks(model['extra'], Path(a.outdir), Path(a.work), frame.iloc[sel], sel_ids,
                                      a.max_text_tokens))
            member_logp(model, feats, sel_ids)   # every column the head reads is there and finite
            save(status='extra_done', blocks=sorted(feats))
        except Exception as e:   # the extra evidence failed: his head, which needs only the GLM features
            save(status='extra_failed', error=repr(e)[:1000])
            model = model['fallback']
    ll = member_logp(model, feats, sel_ids)"""

# V42 names ten datasets by the flat path /kaggle/input/<slug> (inputs lightningv08 attached long ago); a kernel pushed
# through the API mounts every input at /kaggle/input/datasets/<owner>/<slug> (its version 1 failed cell 8's input
# check), so those literals are mapped to the namespaced layout. Cell 16 already tries both for estopadilla's binary.
FLAT = {s: f"/kaggle/input/datasets/{o}/{s}" for o, s in (
    ("dhruvtibarewal", "nlm-l24-adapters-v5"), ("dhruvtibarewal", "nlm-enc-v5"), ("nishantdahal", "ntl-handwriting-blend"),
    ("josephadamski91", "ntl-atlas-qwen3-8b-v1"), ("dhruvtibarewal", "nlm-l24-adapters"), ("nuridudak", "ntl-q14-lora"),
    ("nuridudak", "ntl-q14-lora-v4"), ("kirilltym", "ntl-spectral-style-heads-paired-4404"),
    ("nuridudak", "ntl-svc-prefit-pool-v4n"), ("nuridudak", "ntl-two-stage-encoders"))}
FLAT_RE = re.compile(r"/kaggle/input/(" + "|".join(map(re.escape, sorted(FLAT, key=len, reverse=True)))
                     + r")(?![A-Za-z0-9_.\-])")

VARIANTS = {   # name: (head or None, sibling re-ranking, negative rules[, GSQ share; the 0.694's is 0.06])
    "ntl-lv694-w125": ("w125", False, False),
    "ntl-lv694-pair": ("pair150", False, False),
    "ntl-lv694-gbm": ("gbmpair", False, False),
    "ntl-lv694-gbm-sib": ("gbmpair", True, False),
    "ntl-lv694-gbm-neg": ("gbmpair", False, True),
    "ntl-lv694-gbm-negsib": ("gbmpair", True, True),
    "ntl-lv694-gbm-g3": ("gbmpair", False, False, 0.03),
    "ntl-lv694-gbm-g0": ("gbmpair", False, False, 0.0),
    "ntl-lv694-gbm-g0sib": ("gbmpair", True, False, 0.0),
}
GSQ = 0.06
GSQ_LAST = "CONFIG['gsq_share'] = 0.06\n"
GSQ_ASSERT = "    assert config['gsq_share'] == 0.06, 'Unexpected GSQ share'\n"


def build(name):
    head, sib, neg, *rest = VARIANTS[name]
    gsq = rest[0] if rest else GSQ
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["source"] = FLAT_RE.sub(lambda m: FLAT[m.group(1)], c["source"])
    left = [m.group(0) for c in cells if c["cell_type"] == "code"
            for m in re.finditer(r"/kaggle/input/(?!(?:datasets|models|notebooks|competitions)(?![A-Za-z0-9_.\-]))"
                                 r"[A-Za-z0-9_.\-]+", c["source"])]
    assert left == ["/kaggle/input/ntl-glm-likelihood-train"], left
    if head:
        model = json.loads((HEADS / f"{head}.json").read_text(encoding="utf-8"))
        assert model["format"] == "ntl-v1-ll-model-1" and len(model["classes"]) == 18
        hit = [c for c in cells if c["cell_type"] == "code" and text(c).startswith(LL_CELL)]
        assert len(hit) == 1
        hit[0]["source"] = (LL_CELL + f"# likelihood head refit on our own train likelihoods ({head}, exp050)\n"
                            + f"Path('/kaggle/temp/v1/ll_model.json').write_text(json.dumps({model!r}))\n")
        if model.get("extra"):
            assert model["fallback"]["format"] == "ntl-v1-ll-model-1"
            st = [c for c in cells if c["cell_type"] == "code" and text(c).startswith(STAGE)]
            assert len(st) == 1
            src = text(st[0])
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
                          "    return out\n\n\n" + EXTRA_FUNCS),
                         ("    ll = member_logp(model, feats, sel_ids)", EXTRA_CALL)):
                assert src.count(a) == 1, a
                src = src.replace(a, b)
            st[0]["source"] = src
            at = cells.index(st[0]) + 1
            cells[at:at] = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                             "source": "%%writefile /kaggle/temp/v1/extra_features.py\n" + EXTRA_SRC.read_text(encoding="utf-8")}]
    ecode = e683_code()
    if neg:
        ecm = next(c for c in ecode if c.startswith(RUNTIME + "core_math.py"))
        block = ecm[ecm.index("    # Soft evidence logit update"):ecm.index("    return res\n") + len("    return res\n")]
        patch(cells, RUNTIME + "core_math.py", "    return total + gate[:, None] * evidence\n", block)
    if sib:
        eens = next(c for c in ecode if c.startswith(RUNTIME + "ensemble.py"))
        i = eens.index("    # --- UPGRADE: SIBLING STYLE RESIDUAL RE-RANKING ---")
        block = eens[i:eens.index("\n\n", i) + 2]
        save = "    np.savez(work / 'final-logits.npz', ids=np.asarray(ids), labels=np.asarray(LABELS), logits=logits)\n"
        patch(cells, RUNTIME + "ensemble.py", save, block + save)
        module = next(c for c in ecode if c.startswith(RUNTIME + "sibling_style.py"))
        at = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(c).startswith(RUNTIME + "ensemble.py"))
        cells[at:at] = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": module}]
    if gsq != GSQ:
        patch(cells, RUNTIME + "ensemble.py", GSQ_ASSERT, GSQ_ASSERT.replace("0.06", repr(gsq)))
        patch(cells, GSQ_LAST, GSQ_LAST, f"CONFIG['gsq_share'] = {gsq!r}   # GSQ-27B share (the 0.694: 0.06)\n")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\nThe public [0.694]({URL}) (version 42); likelihood head {head or 'his'}, "
                     f"the [0.683]({E683_URL})'s sibling-style re-ranking {sib}, typographic negative rules {neg}"
                     + (f", GSQ share {gsq}." if gsq != GSQ else ".")})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_lightningv08_694_v42.meta.json").read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": up["datasetDataSources"], "competition_sources": up["competitionDataSources"],
            "kernel_sources": up["kernelDataSources"], "model_sources": up["modelDataSources"],
            "docker_image": up["dockerImage"], "machine_shape": up["machineShape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256("".join(text(c) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(cells)} cells, code sha256 {sha}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        build(name)
