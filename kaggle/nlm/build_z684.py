"""Forks of the public 0.684 (lbjyang/ntl-684-ozertuu-mirror version 1; krshs90's and wadigongren's 0.684s are the same
code byte for byte).

The 0.684 = ozertuu's NTL Atlas V9 Qwen Embed (version 2, public 0.679) with two changes: infer.py now applies a member's
`adapter_weights` (version 2 averaged its adapters equally), and the two-stage Qwen3-14B member (n14b, weight 2.5) reads
Nuri Dudak's q14 LoRA v4 at 0.72 + Nishant Dahal's two-stage Qwen3-14B LoRA at 0.28 instead of those two + q14 LoRA v1
at a third each: +0.005 on the public board. Both public versions read crtproduction's Open Pool v16.

Knobs:
    oq<NN>   q14 LoRA v4 share inside n14b (0.72 upstream; 100 drops the two-stage adapter)
    n14w<W>  n14b member weight (2.5 upstream)
    g<NN>    GSQ share (0.03 upstream; lightningv08's 0.03 -> 0.06 was 0.680 -> 0.681 on his line)
    neg      the public 0.683's typographic negative evidence on the final scores: -15 logits for the classes the
             official rows never show with a passage's spaced em dash (all but claude-opus-4-6 / -5, glm-5.3 / -flash),
             curly apostrophe or curly opening quote (claude* and glm*)
    sib      the public 0.683's sibling-style re-ranking (omkarpattanaik123's sibling_style.py, unchanged) on the final
             scores, cap 0.10, grouped by the official train prompts (on the 0.683 the cap at 0 or 0.05 lost ~0.005)
    typo     the public 0.683's per-fold typography step (core_math.finalize minus its balancing, which the 0.684 does
             itself): gated soft evidence from 37 typographic markers fitted on the official rows (V31 lineage;
             typography_evidence copied unchanged), then the -15 negative rules, on every native fold before the mean
    wemm<NN> CHEN Han's second native expert (Tencent WeMM-Embedding-4B, five folds) as the public 0.683 runs it: its
             wemm_embedding_worker.py and wemm_fusion.py unchanged, one worker per T4 (folds 0-2 / 3-4) after the
             Qwen3-Embedding expert, a monotonic deadline at the native deadline + 20 min (unscored folds fall back to
             the mean of the scored ones), mixed log-linearly per fold at the share; any failure keeps the 0.684 path

    python kaggle/nlm/build_z684.py [variant ...]
"""
import hashlib
import json
import sys
from pathlib import Path

from build_l681 import patch, text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_lbjyang_684_v1.ipynb"
UPSTREAM_SHA = "340d29bb90e4dc60"
URL = "https://www.kaggle.com/code/lbjyang/ntl-684-ozertuu-mirror"

VARIANTS = {   # name: (q14 LoRA v4 share in n14b, n14b member weight, GSQ share[, neg rules, sibling cap, typography, WeMM share])
    "ntl-z684-oq85": ("0.85", "2.5", "0.03"),
    "ntl-z684-oq100": ("1.0", "2.5", "0.03"),
    "ntl-z684-oq60": ("0.60", "2.5", "0.03"),
    "ntl-z684-n14w35": ("0.72", "3.5", "0.03"),
    "ntl-z684-g06": ("0.72", "2.5", "0.06"),
    "ntl-z684-neg": ("0.72", "2.5", "0.03", "1", "0.0"),
    "ntl-z684-sib": ("0.72", "2.5", "0.03", "0", "0.10"),
    "ntl-z684-neg-sib": ("0.72", "2.5", "0.03", "1", "0.10"),
    "ntl-z684-typo": ("0.72", "2.5", "0.03", "0", "0.0", "1"),
    "ntl-z684-typo-sib": ("0.72", "2.5", "0.03", "0", "0.10", "1"),
    "ntl-z684-wemm07": ("0.72", "2.5", "0.03", "0", "0.0", "0", "0.07"),
}
WEMM_POST = '''_WEMM = None


def run_wemm():
    """CHEN Han's WeMM-Embedding-4B expert as the public 0.683 runs it (worker and fusion code unchanged): one replica per T4,
    folds 0-2 and 3-4, the second loaded after the first is on its GPU. Any failure leaves the Qwen3-Embedding path alone."""
    global _WEMM
    _WEMM = None
    if a.wemm_share <= 0 or not a.wemm_dir:
        return
    try:
        import subprocess
        from wemm_fusion import load_wemm
        left = a.native_deadline + 1200 - time.time()
        if left < 1500:
            log('WEMM SKIPPED: not enough time left')
            return
        deadline = time.monotonic() + left
        here = os.path.dirname(os.path.abspath(__file__))
        work = os.path.join(a.native_work, 'wemm')
        os.makedirs(work, exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        procs, outs = [], []
        for gpu, fl in (('0', '0,1,2'), ('1', '3,4')):
            out, ready = os.path.join(work, f'wemm-gpu{gpu}.npz'), os.path.join(work, f'wemm-gpu{gpu}.ready')
            for f in (out, ready):
                if os.path.exists(f):
                    os.remove(f)
            env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUDA_VISIBLE_DEVICES=gpu)
            cmd = [sys.executable, '-u', os.path.join(here, 'wemm_embedding_worker.py'), '--base', a.wemm_base, '--assets', a.wemm_dir,
                   '--query', os.path.join(a.data, 'test.csv'), '--out', out, '--work', os.path.join(work, 'gpu' + gpu),
                   '--deadline', str(deadline), '--folds', fl, '--ready', ready]
            procs.append(subprocess.Popen(cmd, env=env))
            outs.append(out)
            while not os.path.exists(ready) and procs[-1].poll() is None:
                time.sleep(5)
        codes = [q.wait() for q in procs]
        archives = [o for o in outs if os.path.exists(o)]
        log('wemm worker exit codes', codes, 'archives', len(archives))
        if not archives:
            raise RuntimeError('WeMM produced nothing')
        W, wf = load_wemm(archives, a.wemm_dir, test['id'].astype(str).tolist(), LABELS, np.unique(Gte, return_inverse=True)[1])
        assert W.shape == (5, len(test), 18) and np.isfinite(W).all()
        agree = [float((W[k].argmax(1) == W.mean(0).argmax(1)).mean()) for k in range(5)]
        log(f'wemm folds {wf} at share {a.wemm_share}; per-fold agreement with the WeMM mean', [round(v, 3) for v in agree])
        _WEMM = W
    except Exception as _e:
        _WEMM = None
        log('WEMM SKIPPED:', repr(_e))


def mix_wemm(z, fold):
    return z if _WEMM is None else (1 - a.wemm_share) * z + a.wemm_share * _WEMM[fold]


'''
WEMM_MODEL = "dangkhoa2016/tencent-wemm-embedding-4b/Transformers/default/1"

E683 = HERE / "upstream_omkarpattanaik123_apex_v1.ipynb"
POST = '''NEG_RULES = [(' \u2014 ', {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}, True),   # spaced em dash: only these
             ('\u2019', {l for l in LABELS if l.startswith(('claude', 'glm'))}, False),   # curly apostrophe: never these
             ('\u201c', {l for l in LABELS if l.startswith(('claude', 'glm'))}, False)]   # curly opening quote: never these


def post_rules(probs):
    """Ported from the public 0.683 (omkarpattanaik123): typographic negative evidence (-15 logits) and the
    sibling-style re-ranking within model families, on the final fold-averaged scores."""
    L = np.log(np.maximum(probs, 1e-300))
    if a.neg_rules:
        for m, labs, only in NEG_RULES:
            rows = np.array([m in t for t in test.text.values])
            cols = [i for i, l in enumerate(LABELS) if (l not in labs) == only]
            if rows.any():
                L[np.ix_(rows, cols)] -= 15.0
            log(f'negative rule {m!r}: {int(rows.sum())} rows, {len(cols)} classes penalised')
    if a.sibling_alpha > 0:
        from sibling_style import apply_grouped_style_residual
        L = apply_grouped_style_residual(L, train.text.values, test.text.values, LABELS, y, Gtr, max_alpha=a.sibling_alpha,
                                         diagnostics_path='/kaggle/working/sibling-style-diagnostics.json')
        log('sibling-style re-ranking', open('/kaggle/working/sibling-style-diagnostics.json').read().replace(chr(10), ' '))
    return L


_TYPO = None


def fold_scores(S):
    """Ported from the public 0.683's core_math.finalize (its balancing aside, which decide_scores does here): gated
    soft typography evidence fitted on the official rows, then the typographic negative evidence (-15 logits)."""
    global _TYPO
    if not a.typo:
        return S
    if _TYPO is None:
        from ntl_typography import typography_evidence
        ev, Xte = typography_evidence(test.text.values, train.text.values, y)
        wc = np.array([len(re.findall(r'\\b\\w+\\b', str(t))) for t in test.text.values], dtype=np.float32)
        static = np.clip(np.sqrt(np.maximum(wc, 1.0) / 80.0), 0.40, 1.0) * np.clip(np.sqrt((Xte.sum(axis=1) + 1.0) / 4.0), 0.45, 1.0)
        rules = []
        for m, labs, only in NEG_RULES:
            rows = np.array([m in t for t in test.text.values])
            rules.append((rows, [i for i, l in enumerate(LABELS) if (l not in labs) == only]))
        _TYPO = (ev, static, rules)
        log('typography evidence', ev.shape, 'max |evidence|', round(float(abs(ev).max()), 3), 'rule rows', [int(r.sum()) for r, _ in rules])
    ev, static, rules = _TYPO
    o = np.sort(rowsoftmax(S), axis=1)
    gate = np.clip((0.60 - (o[:, -1] - o[:, -2])) / 0.60, 0., 1.) * static
    S = S + gate[:, None] * ev
    for rows, cols in rules:
        if rows.any():
            S[np.ix_(rows, cols)] -= 15.0
    return S


'''


def build(name):
    oq, w, gsq, *post = VARIANTS[name]
    neg, sib, typo, wemm = (list(post) + ["0", "0"])[:4] if post else ("0", "0.0", "0", "0")
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
    run = "LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]"
    if oq != "0.72":
        new = ('"adapters": [OQ4], "adapter_weights": [1.0]' if float(oq) == 1 else
               f'"adapters": [OQ4, N14B], "adapter_weights": [{oq}, {1 - float(oq):.2f}]')
        patch(cells, run, '"adapters": [OQ4, N14B], "adapter_weights": [0.72, 0.28]', new)
    if w != "2.5":
        patch(cells, run, '"w": 2.5,', f'"w": {w},')
    if gsq != "0.03":
        patch(cells, run, "'--gsq_share', '0.03'", f"'--gsq_share', '{gsq}'")
    if post:
        inf = "%%writefile infer.py"
        patch(cells, inf, "p.add_argument('--group_sim', type=float, default=0.70)\n",
              "p.add_argument('--group_sim', type=float, default=0.70)\np.add_argument('--neg_rules', type=int, default=0)\n"
              "p.add_argument('--sibling_alpha', type=float, default=0.0)\np.add_argument('--typo', type=int, default=0)\n")
        patch(cells, inf, "def rowsoftmax(z):\n", POST + "def rowsoftmax(z):\n")
        patch(cells, inf, "            write_pred(probs.argmax(1), 'final blend with native q8')\n",
              "            write_pred(post_rules(probs).argmax(1), 'final blend with native q8 + ported 0.683 steps')\n")
        patch(cells, run, "'--comp_w', '0.15', '--out', OUT]", f"'--comp_w', '0.15', '--out', OUT, '--neg_rules', '{neg}', '--sibling_alpha', '{sib}', '--typo', '{typo}']")
        patch(cells, inf, "        probs += rowsoftmax(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i]))) / len(folds)\n",
              "        probs += rowsoftmax(fold_scores(decide_scores(rowsoftmax((1 - share) * logP + share * lp[i])))) / len(folds)\n")
        if float(wemm) > 0:
            patch(cells, inf, "p.add_argument('--typo', type=int, default=0)\n", "p.add_argument('--typo', type=int, default=0)\n"
                  "p.add_argument('--wemm_base', default='')\np.add_argument('--wemm_dir', default='')\n"
                  "p.add_argument('--wemm_share', type=float, default=0.0)\n")
            patch(cells, inf, "\ndef rowsoftmax(z):\n", "\n" + WEMM_POST + "def rowsoftmax(z):\n")
            patch(cells, inf, "    probs = np.zeros_like(P)\n", "    run_wemm()\n    probs = np.zeros_like(P)\n")
            patch(cells, inf, "rowsoftmax((1 - share) * logP + share * lp[i])", "rowsoftmax(mix_wemm((1 - share) * logP + share * lp[i], folds[i]))")
            setup = "import glob\nimport json"
            patch(cells, setup, "NATIVE_BASE = model_dir('/qwen-3-embedding/', '/8b/')\n",
                  "NATIVE_BASE = model_dir('/qwen-3-embedding/', '/8b/')\nWEMM_BASE = model_dir('/tencent-wemm-embedding-4b/')\n"
                  "WEMM = os.path.dirname(os.path.dirname(first('/kaggle/input/**/ntl-native-wemm4-fivefold-0679/fold0/calibration.json')))\n"
                  "print('wemm', WEMM_BASE, WEMM)\n")
            patch(cells, run, "'--native_dir', NATIVE, '--native_base', NATIVE_BASE,",
                  f"'--native_dir', NATIVE, '--native_base', NATIVE_BASE, '--wemm_base', WEMM_BASE, '--wemm_dir', WEMM, '--wemm_share', '{wemm}',")
            e683 = [text(c) for c in json.loads(E683.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
            mods = []
            for mod in ("wemm_embedding_worker.py", "wemm_fusion.py"):
                src = next(c for c in e683 if c.startswith("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/" + mod))
                mods.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                             "source": f"%%writefile {mod}\n" + src.split("\n", 1)[1]})
            at = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(c).startswith("%%writefile infer.py"))
            cells[at:at] = [{"cell_type": "markdown", "metadata": {}, "source": "## WeMM-Embedding-4B expert (from the public 0.683)\n\n"
                             "CHEN Han's second native expert, with the 0.683's `wemm_embedding_worker.py` and `wemm_fusion.py` unchanged "
                             "(as wired in lightningv08's five-fold embeddings notebook)."}] + mods
        if typo == "1":   # the 0.683's typography_evidence and its marker list, unchanged, as a module next to infer.py
            cm = next(text(c) for c in json.loads(E683.read_text(encoding="utf-8"))["cells"]
                      if c["cell_type"] == "code" and text(c).startswith("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/core_math.py"))
            labels = cm[cm.index("LABELS = ["):cm.index("GUARDS = [")]
            evid = cm[cm.index("TYPOGRAPHY_MARKS = ["):cm.index("def balance_logits(")]
            at = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(c).startswith("%%writefile infer.py"))
            cells[at:at] = [{"cell_type": "markdown", "metadata": {}, "source": "## Typography evidence (from the public 0.683)\n\n"
                             "The 0.683's `typography_evidence` and its 37 markers, unchanged: a smoothed Bernoulli score per class "
                             "from which markers a passage contains, fitted on the official rows and clipped to ±0.30 logits."},
                            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                             "source": "%%writefile ntl_typography.py\nimport numpy as np\n\n" + labels + evid.rstrip() + "\n"}]
        if float(sib) > 0:   # omkarpattanaik123's module, unchanged, written next to infer.py
            src = next(text(c) for c in json.loads(E683.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code" and text(c).startswith("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/sibling_style.py"))
            body = src.split("\n", 1)[1]
            at = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(c).startswith("%%writefile infer.py"))
            cells[at:at] = [{"cell_type": "markdown", "metadata": {}, "source": "## Sibling-style re-ranking (from the public 0.683)\n\n"
                             "omkarpattanaik123's `sibling_style.py`, unchanged: within each model family, a small logistic model on "
                             "stylometric counts shifts at most 10% of a close call's probability between siblings."},
                            {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                             "source": "%%writefile sibling_style.py\n" + body}]
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\nThe public [0.684]({URL}) (version 1); n14b: q14 LoRA v4 share {oq}, member weight {w}; "
                     f"GSQ share {gsq}" + (f"; negative rules {neg}, sibling cap {sib}, typography {typo}, WeMM {wemm}." if post else ".")})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_lbjyang_684_v1.meta.json").read_text(encoding="utf-8"))
    models = up["modelDataSources"] + ([WEMM_MODEL] if post and float(wemm) > 0 else [])
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": up["datasetDataSources"], "competition_sources": up["competitionDataSources"],
            "kernel_sources": up["kernelDataSources"], "model_sources": models,
            "docker_image": up["dockerImage"], "machine_shape": up["machineShape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256("".join(text(c) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(cells)} cells, code sha256 {sha}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        build(name)
