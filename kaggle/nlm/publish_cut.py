"""Publication page for a scored build_ntl40.py cut_fork of lightningv08's 0.704 (ntl-lv704-l256, ntl-lv704-l320): the
GLM likelihood stage reads a longer cut with estopadilla's head refitted by his recipe. Our header, a note before the
GLM stage, lightningv08's own section texts and credits, the scored code cells byte for byte. Published only if the
fork's LB score beats the Code tab's top (user, 2026-11-19).

    python kaggle/nlm/publish_cut.py <variant> <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_l681 import text
from build_ntl40 import BILL_URL, FABLE_URL, SOFT_URL
from publish_lv704 import ESTO, ESTO_TRAIN

HERE = Path(__file__).resolve().parent
# exp060, out of fold: the head alone (18-class accuracy; on passages longer than 160 tokens) and after the stage
OOF = {256: ("0.215", "0.241", "+0.0015 [-0.0004, +0.0033]"), 320: ("0.222", "0.259", "+0.0004 [-0.0019, +0.0026]")}
# our LB reads on the 0.704 line (each one submission, 2026-11-19/20)
LB = [("the 0.704 + blend weight 1.25 W (bill123987's ntl706w change)", "0.704"),
      ("the 0.704 + blend weight 1.5 W", "0.704"),
      ("the 0.704 + 1.25 W + typography penalty 0.05", "0.704"),
      ("the 0.704 + 1.25 W without the Fable routing", "0.703"),
      ("the 0.704 + a boosted-tree GLM head", "0.702")]


def build(variant, score, scored, title=None, extra_lb=()):
    cut = int(variant.rsplit("-l", 1)[1])
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / variant / f"{variant}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    head_cell = next(c for c in codes if c.startswith("# ours (exp060)"))
    assert f"first {cut} tokens" in head_cell and f'"max_text_tokens": {cut}' in head_cell
    title = title or f"{score} | GLM Fable Routing + {cut}-Token Likelihood"
    assert 5 <= len(title) <= 50, (title, len(title))
    alone, alone_long, stage = OOF[cut]
    rows = [f"| lightningv08's [0.702 Soft Typography]({SOFT_URL}): the shared base | 0.702 |",
            f"| lightningv08's [0.704 GLM Fable Routing]({FABLE_URL}): the base of this notebook | 0.704 |",
            f"| bill123987's [ntl706w]({BILL_URL}): the 0.702 + the GLM blend at 1.25 W | 0.704 |"]
    rows += [f"| ours: {a} | {b} |" for a, b in list(LB) + list(extra_lb)]
    rows.append(f"| **this notebook**: the 0.704 + the GLM stage reading {cut} tokens | **{score}** |")
    table = "\n".join(rows)
    head = f"""# {title}

**Public LB {score}** · lightningv08's 0.704 with the GLM likelihood stage reading the first {cut} tokens of each
passage instead of 160 · T4 ×2 · internet off · every input is public

**In short.** lightningv08's [0.704]({FABLE_URL}) ends with estopadilla's GLM likelihood stage: the ~15% of test
passages that look most like GLM (plus a capped share of claude-fable-5 probability) are scored under the real
GLM-5.3-Flash and GLM-5.3, and a logistic head turns per-token statistics of the **first 160 tokens** into class
log-probabilities. This notebook changes only that cut: the head is refitted with estopadilla's exact recipe on the
first **{cut} tokens**, and the stage reads {cut} tokens per passage. Every other code cell is the 0.704's, and the code
cells below are byte for byte the version that scored {score}.

## What changed (two cells)

* **The head** (the cell that writes `ll_model.json`): estopadilla's recipe, unchanged: the same 54 likelihood features
  (per model: token log-probability mean, spread and quantiles, next-token entropy, Fast-DetectGPT curvature, rank
  statistics, bits per character, ...; plus six Flash − GLM-5.3 differences), StandardScaler + multinomial logistic
  regression, C = 0.05, blend weight W = 1. Fitted on the 2,699 train passages cut at {cut} tokens instead of 160, so
  `max_text_tokens` = {cut}. `v1_stage.py` passes that cut to the GLM runner and asserts that the test is cut exactly like
  the head's training features.
* **The token budget** (`v1_stage.py`): at most 130,000 scored token positions instead of 90,000, so that the 15% cover
  still binds with longer passages (about 100,000 positions expected on ~4,000 passages). lightningv08's earlier
  22% / 130,000 apex-style version ran within the time limit with the same budget.

## Why a longer cut

* 38% of the train passages are longer than 160 tokens (median 130; 11% are longer than 256), so the 160 cut drops the
  tail of more than a third of the passages the stage reads.
* Out of fold (5 folds grouped by prompt), the head alone gets better with each longer cut up to 320: 18-class accuracy 0.190 at
  160 tokens → 0.200 (192) → 0.210 (224) → 0.215 (256) → 0.222 (320); on passages longer than 160 tokens 0.204 →
  {alone_long} at {cut}.
* After the stage (rows chosen as in the 0.704, stage moves out of GLM reverted as its post-processing does), {cut}
  tokens scored {stage} balanced accuracy over 160 on a stand-in stack. Small, because the stage only decides a few
  hundred rows.
* The test-time GLM scores are noisier than any train run: the GGUF mixture-of-experts passes are batch-dependent, and
  the same text can get per-token log-probabilities several nats apart in two runs. Statistics averaged over more tokens
  are steadier, a plausible reason why the leaderboard gain (+0.002) is larger than the out-of-fold one.

## Leaderboard A/B

| Notebook | Public LB |
|---|---:|
{table}

Each row is one submission. On the 0.704 the stage's blend weight and the typography penalty did not move the public
score; the longer cut did. The public board scores part of the test set, and identical code has scored ±0.001 apart
here, so read single 0.001 steps with care.

## How the head was fitted

The train likelihoods come from running estopadilla's `ntl-segrun` (same prompt prefix, seq-batch 8) on all 2,699
train passages under both models, cut at 512 tokens, keeping the per-token scores. The head is then:

```python
from features import one            # the stage's own features.py (cell below)
f = [one(s[:{cut}], len(t)) for s, t in zip(flash_scores, texts)]    # likewise g for GLM-5.3
X = [flash:<24 features>, glm53:<24 features>, d:<6 differences>]   # the column layout of ll_model.json
sc = StandardScaler().fit(X)
lr = LogisticRegression(C=0.05, max_iter=5000).fit(sc.transform(X), y)
```

At 160 tokens the same code reproduces estopadilla's published head (coefficient correlation 0.994; the remaining gap
is the GLM runs' batch noise), so the cut is the only change.

## How it works

1. **The stack** (lightningv08's pipeline): multi-depth LoRA read-outs of Qwen3-14B, Qwen3-8B, Llama-3.1-8B and
   Qwen2.5-14B, the Qwen3-14B adapters of nuridudak and Nishant Dahal, CHEN Han's five-fold Qwen3-Embedding-8B expert,
   lightningv08's Qwen3.8-27B classifier, a DeBERTa reader, soft Sinkhorn balancing.
2. **The GLM stage** (estopadilla's, with the 0.704's routing): passages ranked by GLM probability plus a capped share
   of claude-fable-5 probability are scored under the real GLM-5.3-Flash and GLM-5.3 (jamesliebel's IQ3_XXS GGUF
   segments, estopadilla's `ntl-segrun`), **first {cut} tokens**. The refitted head turns the two models' likelihood
   statistics into class log-probabilities, blended in at W.
3. **Apex post-processing** (lumibee's sibling residual and typography rules, lightningv08's soft version and Fable
   routing): a within-family style residual, then three typographic habits that make some labels unlikely.

## Credits

* **lightningv08 (Vladislav Lavrentev)**: the [0.704]({FABLE_URL}) this notebook runs, and the pipeline behind it.
* **José Padilla (estopadilla)**: the [GLM likelihood stage]({ESTO}), `ntl-segrun`, the features and the
  [likelihood head recipe]({ESTO_TRAIN}) refitted here. **jamesliebel**: the public GLM-5.3 / GLM-5.3-Flash GGUF segment
  mirrors.
* **bill123987**: [ntl706w]({BILL_URL}), whose stage change (1.25 W) is in the A/B table above.
* Everyone credited in lightningv08's credits section below; each input keeps its own licence. **Built with Llama.**
  Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/)
  (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).

## Running it

Attach the inputs below, choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit
that version. The three visible test rows only run a fast format preview and the input checks; Kaggle's rerun on the
hidden rows runs everything.
"""
    stage_md = f"""## The GLM likelihood stage, at {cut} tokens

The next cells are the 0.704's GLM stage unchanged except two: the cell writing `ll_model.json` (estopadilla's recipe
refitted on the first {cut} tokens, `max_text_tokens` = {cut}) and, in `v1_stage.py`, the token budget (130,000 scored
positions, was 90,000).
"""
    cells = list(built["cells"])
    assert text(cells[0]).startswith(f"# {variant}")
    first = next(i for i, c in enumerate(cells) if text(c).startswith("from pathlib import Path\nPath('/kaggle/temp/v1')"))
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    out = [md(head)] + cells[1:first] + [md(stage_md)] + cells[first:]
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
    title = None
    if "--title" in args:
        i = args.index("--title"); title = args[i + 1]; del args[i:i + 2]
    build(args[0], args[1], args[2], title)
