"""Publication page for a scored build_ntl40.py fork of lightningv08's 0.704 with the 1.25 W GLM blend (ntl-lv704-w125,
ntl-lv704-w125-c22): our header, a note before the GLM stage, lightningv08's own section texts and credits, the scored
code cells byte for byte. Published only if the fork's LB score beats the Code tab's top (user, 2026-11-19).

    python kaggle/nlm/publish_lv704.py <variant> <score> <scored.ipynb> [--lb 116=0.704,117=0.704,...] [--title "..."]

--lb: our other LB reads on the 0.704 (submission number = score), shown in the A/B table when given.
Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_l681 import text
from build_ntl40 import BILL_URL, FABLE_URL, SOFT_URL

HERE = Path(__file__).resolve().parent
K = "https://www.kaggle.com"
ESTO = f"{K}/code/estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4"
ESTO_TRAIN = f"{K}/code/estopadilla/ntl-glm-likelihood-train"
APEX15 = f"{K}/code/lightningv08/0-702-apex-style-15-glm-qwen3-8-27b-lora"
OURS = {116: "the 0.704 + a boosted-tree GLM head + the 22% stage budget",
        117: "the 0.704 + a boosted-tree GLM head (our exp050 head; LB-neutral on three bases)",
        118: "the 0.704 + the 1.25 W blend (ntl-lv704-w125)"}
VARIANTS = {"ntl-lv704-w125": False, "ntl-lv704-w125-c22": True}


def build(variant, score, scored, lb=None, title=None):
    wide = VARIANTS[variant]
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / variant / f"{variant}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    title = title or (f"{score} | Fable Routing, 1.25 W Blend, 22% Stage" if wide
                      else f"{score} | GLM Fable Routing + Stronger GLM Blend")
    assert 5 <= len(title) <= 50, (title, len(title))
    rows = [f"| lightningv08's [0.702 Soft Typography]({SOFT_URL}): the shared base | 0.702 |",
            f"| bill123987's [ntl706w]({BILL_URL}): the 0.702 + the GLM blend at 1.25 W | 0.704 |",
            f"| lightningv08's [0.704 GLM Fable Routing]({FABLE_URL}): the 0.702 + Fable routing, post-likelihood "
            "probabilities for the typography check of GLM winners, penalty 0.15 | 0.704 |"]
    for k, v in sorted((lb or {}).items()):
        if k in OURS and not (k == 118 and not wide):
            rows.append(f"| ours: {OURS[k]} | {v} |")
    rows.append(f"| **this notebook**: the 0.704 + the 1.25 W blend{' + the 22% stage budget' if wide else ''} | "
                f"**{score}** |")
    table = "\n".join(rows)
    wide_note = (f"""
* **Stage budget: 22% of the test / 130,000 token positions** (the 0.704 keeps 15% / 90,000). The Fable routing ranks
  Fable-heavy passages into the stage, so at the old budget they push GLM passages out of it; the wider budget, which
  lightningv08's earlier [apex-style-15]({APEX15}) version ran within the 9-hour limit, scores both.""" if wide else "")
    head = f"""# {title}

**Public LB {score}** · lightningv08's 0.704 with a stronger GLM likelihood blend · T4 ×2 · internet off · every input
is public

**In short.** Two public notebooks took the same 0.702 base to 0.704 by changing different parts of the GLM likelihood
stage: lightningv08's [0.704]({FABLE_URL}) changes *which* passages the stage scores and how its winners pass the
typography check; bill123987's [ntl706w]({BILL_URL}) changes only *how strongly* the GLM likelihood evidence is blended
in (1.25 × the head's weight W). This notebook is the 0.704 with ntl706w's two-line change added, so it combines both.
Every other code cell is the 0.704's, and the code cells below are byte for byte the version that scored {score}.

## What changed (all in `v1_stage.py`, section 13)

* **Blend weight 1.25 W.** On the passages it scores, the stage adds `w · (head log-probabilities)` to the stack's
  log-probabilities, with `w = W / T'` (W = 1 from estopadilla's head, T' a temperature that matches the stack's
  confidence to the head's). Here `w = 1.25 · W / T'`.
* **Sanity guard 0.32** (was 0.25): the stage refuses its own result if it changes more than this share of all
  predictions. At a 15% budget it cannot trip either way; it is kept as in ntl706w.{wide_note}

## Why a stronger blend

* The likelihood head was fitted on the 2,699 train passages, where W = 1 was a sensible default, not a tuned value.
  Out of fold (5 folds grouped by prompt, the stage applied to the most GLM-like passages of a stand-in stack), W =
  1.25 beats W = 1 under both the plain GLM ranking and the 0.704's Fable routing: +0.0015 to +0.0025 balanced
  accuracy, with lower bounds around zero; W = 1.5 gains less. That matches ntl706w's +0.002 on the leaderboard.
* The 0.704's post-processing sends every stage passage whose new label is not GLM back to its pre-stage label, so in
  practice the blend weight decides how many passages move *into* glm-5.3 / glm-5.3-flash. A stronger blend moves
  more of the passages the GLM models find very likely.

## Leaderboard A/B

| Notebook | Public LB |
|---|---:|
{table}

Each row is one submission. The public board scores part of the test set, and identical code has scored ±0.001 apart
here, so read single 0.001 steps with care.

## How it works

1. **The stack** (lightningv08's pipeline): multi-depth LoRA read-outs of Qwen3-14B, Qwen3-8B, Llama-3.1-8B and
   Qwen2.5-14B, the Qwen3-14B adapters of nuridudak and Nishant Dahal, CHEN Han's five-fold Qwen3-Embedding-8B expert,
   lightningv08's Qwen3.8-27B classifier, a DeBERTa reader, soft Sinkhorn balancing.
2. **The GLM stage** (estopadilla's, with the 0.704's routing): passages ranked by GLM probability plus a capped share
   of claude-fable-5 probability are scored under the real GLM-5.3-Flash and GLM-5.3 (jamesliebel's IQ3_XXS GGUF
   segments, estopadilla's `ntl-segrun`), first 160 tokens. estopadilla's logistic head turns the two models'
   likelihood statistics into class log-probabilities, blended in at 1.25 W.
3. **Apex post-processing** (lumibee's sibling residual and typography rules, lightningv08's soft version): a
   within-family style residual, then three typographic habits that make some labels unlikely (penalty 0.15).

## Credits

* **lightningv08 (Vladislav Lavrentev)**: the [0.704]({FABLE_URL}) this notebook runs, and the pipeline behind it.
* **bill123987**: the 1.25 W blend, in [ntl706w]({BILL_URL}).
* **José Padilla (estopadilla)**: the [GLM likelihood stage]({ESTO}), `ntl-segrun` and the
  [likelihood head]({ESTO_TRAIN}). **jamesliebel**: the public GLM-5.3 / GLM-5.3-Flash GGUF segment mirrors.
* Everyone credited in lightningv08's credits section below; each input keeps its own licence. **Built with Llama.**
  Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)
  (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).

## Running it

Attach the inputs below, choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit
that version. The three visible test rows only run a fast format preview and the input checks; Kaggle's rerun on the
hidden rows runs everything.
"""
    stage = f"""## The GLM likelihood stage, with the 1.25 W blend

The next cells are the 0.704's GLM stage unchanged except in `v1_stage.py`: the line `w = 1.25 * model['W'] / tp`
(was `model['W'] / tp`) and the sanity guard default 0.32 (was 0.25){"; and the budget, 22% of the test / 130,000 token positions (was 15% / 90,000)" if wide else ""}.
"""
    cells = list(built["cells"])
    assert text(cells[0]).startswith(f"# {variant}")
    first = next(i for i, c in enumerate(cells) if text(c).startswith("from pathlib import Path\nPath('/kaggle/temp/v1')"))
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    out = [md(head)] + cells[1:first] + [md(stage)] + cells[first:]
    nb_out = dict(nb, cells=out)
    assert [text(c) for c in nb_out["cells"] if c["cell_type"] == "code"] == codes
    meta = json.loads((HERE / variant / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(title=title, is_private=False)
    dest = HERE / "publish" / variant
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{variant}.ipynb").write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"publish/{variant}: title {title!r}, {len(out)} cells, head {len(head)} chars, "
          f"{len(meta['dataset_sources'])} datasets")


if __name__ == "__main__":
    args = sys.argv[1:]
    opts = {}
    for flag in ("--title", "--lb"):
        if flag in args:
            i = args.index(flag); opts[flag] = args[i + 1]; del args[i:i + 2]
    lb = {int(k): v for k, v in (p.split("=") for p in opts["--lb"].split(","))} if "--lb" in opts else None
    build(args[0], args[1], args[2], lb, opts.get("--title"))
