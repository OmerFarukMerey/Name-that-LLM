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

VARIANTS = {   # name: (q14 LoRA v4 share in n14b, n14b member weight, GSQ share[, negative rules, sibling cap])
    "ntl-z684-oq85": ("0.85", "2.5", "0.03"),
    "ntl-z684-oq100": ("1.0", "2.5", "0.03"),
    "ntl-z684-oq60": ("0.60", "2.5", "0.03"),
    "ntl-z684-n14w35": ("0.72", "3.5", "0.03"),
    "ntl-z684-g06": ("0.72", "2.5", "0.06"),
    "ntl-z684-neg": ("0.72", "2.5", "0.03", "1", "0.0"),
    "ntl-z684-sib": ("0.72", "2.5", "0.03", "0", "0.10"),
    "ntl-z684-neg-sib": ("0.72", "2.5", "0.03", "1", "0.10"),
}
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


'''


def build(name):
    oq, w, gsq, *post = VARIANTS[name]
    neg, sib = post or ("0", "0.0")
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
              "p.add_argument('--sibling_alpha', type=float, default=0.0)\n")
        patch(cells, inf, "def rowsoftmax(z):\n", POST + "def rowsoftmax(z):\n")
        patch(cells, inf, "            write_pred(probs.argmax(1), 'final blend with native q8')\n",
              "            write_pred(post_rules(probs).argmax(1), 'final blend with native q8 + ported 0.683 steps')\n")
        patch(cells, run, "'--comp_w', '0.15', '--out', OUT]", f"'--comp_w', '0.15', '--out', OUT, '--neg_rules', '{neg}', '--sibling_alpha', '{sib}']")
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
                     f"GSQ share {gsq}" + (f"; negative rules {neg}, sibling cap {sib}." if post else ".")})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_lbjyang_684_v1.meta.json").read_text(encoding="utf-8"))
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
