"""Publication page for the scored build_graft.py fork ntl-graft-gbm: our header, a markdown note before the GLM stage
cells, the upstream section notes, the scored code cells byte for byte. kirilltym's paired-4404 dataset is attached to
the graft but never read (no cell names its files or matches them with a glob), so the page leaves it out.

    python kaggle/nlm/publish_graft.py <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/ntl-graft-gbm/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_graft import URL as GRAFT
from publish_e683 import ATLAS9, CHEN, K, OQ4, OURS682, Z684
from publish_g692 import ESTO, ESTO_TRAIN, JL
from publish_l681 import ASSETS, LV681, text
from publish_z684 import N14B

HERE = Path(__file__).resolve().parent
NAME = "ntl-graft-gbm"
UNREAD = "kirilltym/ntl-spectral-style-heads-paired-4404"
AREEB = f"{K}/code/areebbb/0-696-this-actually-works"
OZER = f"{K}/code/ozertuu/0-689-dual-svc-residual-refinement-apex"
LV694 = f"{K}/code/lightningv08/0-694-glm-5-3-likelihood-qwen3-8-27b-lora"


def build(score, scored, title=None):
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / NAME / f"{NAME}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    title = title or f"{score} | GLM-5.3 Likelihood + Boosted-Tree Head"
    assert 5 <= len(title) <= 50, (title, len(title))
    head = f"""# {title}

**Public LB {score}** · feiyu26's 0.696 graft with a stronger GLM likelihood head · T4 ×2 · internet off · every input is public

**In short.** This is feiyu26's [NTL 0684 v1 graft]({GRAFT}) (public 0.696): the public 0.684 stack, then estopadilla's
GLM-5.3 / GLM-5.3-Flash likelihood stage. One thing changed: the small head that turns the two GLM models' likelihood
statistics into class scores. estopadilla's logistic head is kept and averaged 50/50 with shallow boosted trees fitted on
the same statistics plus 12 token-by-token Flash-vs-GLM-5.3 contrasts. Everything else (the stack, which texts the stage
scores, its time budget, the blend weight) is the graft's, and the code cells below are byte for byte the version that
scored {score}.

## Why the head is worth improving

After the 0.684 stack, most of the remaining mistakes sit between glm-5.3, glm-5.3-flash and the Claude models, above
all claude-fable-5. On the text alone GLM-5.3 and claude-fable-5 are almost indistinguishable (a word / character
n-gram classifier separates them at AUC ~0.56 in our tests, even with 12k extra Open Pool passages), but under the real
GLM models' likelihood they separate at AUC ~0.82. The signal is the **contrast between the two sibling models**:
GLM-5.3's likelihood alone gives only about a quarter of the stage's gain. A linear head sees each statistic on its own;
depth-2 trees can use pairs of them (for example "Flash finds it much easier than GLM-5.3 *and* both are confident"),
and per-token contrasts keep information that the per-text averages lose.

## What changed

* **The head (`ll_model.json` inside the V1 payload).** estopadilla's head is unchanged: 54 features (24 statistics of
  each model's per-token scores over the first 160 tokens, such as log-probability quantiles, entropy, rank shares and
  bits per character, plus 6 Flash − GLM-5.3 differences), standardised, multinomial logistic regression, C = 0.05.
  Added: a histogram gradient-boosting classifier (max depth 2, 150 rounds, learning rate 0.1, L2 1.0) on those 54 plus
  12 per-token contrasts on the tokens both models score (spread and quantiles of the log-probability difference, the
  share of tokens where Flash is more likely, the mean absolute difference, the entropy difference, how often Flash
  ranks the true token higher or lower). The two heads' class probabilities are averaged 50/50; the log of that average
  replaces the head's log-probabilities. The trees are stored as plain arrays in the JSON and evaluated with numpy (no
  extra package); if anything in the added part fails, the stage falls back to estopadilla's head.
* **`v1_stage.py`** computes the 12 contrasts and evaluates the trees. Text selection, the time box, the blend weight
  and the temperature matching are unchanged.
* **Training data.** The head is fitted on the 2,699 official train passages, scored under the same two GGUF models
  with the same chat prefix and 160-token cut as at test time.

## Validation

* **Out of fold** (5 folds grouped by prompt, so no prompt is in both halves), with the stage applied to the 20% most
  GLM-like passages of a stand-in stack: estopadilla's head adds +0.027 balanced accuracy over the stack, the
  tree-averaged head **a further +0.006**. The gain holds when the GLM-5.3 scores are recomputed in shuffled batches:
  the IQ3_XXS mixture-of-experts model gives slightly different numbers for the same text depending on its batch
  neighbours, and the test run batches differently from the training run.
* **End to end on Kaggle T4 ×2:** the stage, run on 120 train passages, reproduced the offline predictions on 95% of
  them (the rest is the batch effect above).
* **What did not help** (all out of fold, so not in this notebook): refitting C, W, the token cut or the cover (small
  gains alone, nothing on top of the trees); training the head on extra Open Pool passages (−0.001 to −0.004: pool passages look different to the GLM models);
  a Qwen3-4B reference likelihood; self-training on the test's own confident rows (+0.001); more aggressive stage
  time-estimate constants (more cover, but a slower machine then loses the whole stage to its budget, so the graft's
  conservative constants are kept).

## Leaderboard A/B

| Notebook | Public LB |
|---|---:|
| [our 0.682]({OURS682}): lightningv08's 0.681 + GSQ 12% + Sinkhorn target mix 0.50 | 0.682 |
| [the 0.684]({Z684}) (lbjyang: ozertuu's [Atlas V9]({ATLAS9}) + q14 LoRA v4): the stack used here | 0.684 |
| [estopadilla's 0.689]({ESTO}): the GLM likelihood stage on our 0.682's stack | 0.689 |
| [lightningv08's 0.694]({LV694}): his stack with the GLM stage inlined at a fixed row budget | 0.694 |
| [feiyu26's graft]({GRAFT}) (the 0.684 stack + the stage) and its copies ([areebbb]({AREEB}), [ozertuu]({OZER})) | 0.696 |
| **this notebook**: the graft with the boosted-tree head | **{score}** |

Each row is one submission and the public board scores only part of the test set: the same stage code has scored
0.689-0.696 across reruns, so read single steps of 0.001-0.002 with care.

## How it works

1. **The stack** (the 0.684, as the graft runs it): multi-depth LoRA read-outs of Qwen3-14B (weight 3.0), Qwen3-8B (2.0),
   Llama-3.1-8B (1.0), the Qwen3-14B blend of nuridudak's [q14 LoRA v4]({OQ4}) 0.72 + Nishant Dahal's
   [two-stage LoRA]({N14B}) 0.28 (2.5) and Qwen2.5-14B (1.0); CHEN Han's five-fold Qwen3-Embedding-8B expert at a 40%
   share; lightningv08's Qwen3.8-27B GSQ classifier at 3%; the complementary DeBERTa + TF-IDF stack at 0.15; soft
   Sinkhorn balancing (0.30, target mix 0.35).
2. **The GLM stage** (estopadilla): the passages are ranked by the stack's glm-5.3 + glm-5.3-flash probability; as many
   top passages as fit before 8.8 h (at most 30%) are scored under the real GLM-5.3-Flash and GLM-5.3 (jamesliebel's
   public IQ3_XXS GGUF segments, run layer by layer on both T4s by estopadilla's `ntl-segrun`), each as an assistant
   turn, first 160 tokens.
3. **The head** (above) turns each scored passage's likelihood statistics into 18 class log-probabilities, which are
   added to the stack's log-probabilities (rescaled to a fixed confidence level) on those passages only; the rest keep
   the stack's answer.

## Credits and licences

* **feiyu26**: the [graft]({GRAFT}) this notebook runs. **estopadilla (José Padilla)**: the
  [GLM likelihood stage]({ESTO}), `ntl-segrun`, the logistic head kept here and its
  [training notebook]({ESTO_TRAIN}) (Apache 2.0). **jamesliebel**: the public
  [GLM-5.3 / GLM-5.3-Flash GGUF segment mirrors]({JL}).
* **lbjyang, ozertuu, krshs90, wadigongren**: [the 0.684]({Z684}); **nuridudak (Nuri Dudak)** and **Nishant Dahal**:
  the Qwen3-14B adapters; **lightningv08 (Vladislav Lavrentev)**: the pipeline ([0.681]({LV681})), the Qwen3.8-27B GSQ
  classifier and the [asset dataset]({ASSETS}); **CHEN Han**: the native expert ([0.679]({CHEN})); the model, adapter
  and data authors named in the cells below; crtproduction for the Open Pool (CC BY-SA 4.0).
* **Models.** GLM-5.3 / GLM-5.3-Flash: Z.ai, under their own licence. Qwen models: Alibaba Qwen team, under their own
  licences. **Built with Llama.** Llama 3.1 is licensed under the
  [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its
  [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/). Each input keeps its own licence.

Our earlier public notebook: [0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50]({OURS682}).

## Running it

Attach the inputs below (the graft's, without kirilltym's paired-4404 heads dataset, which the graft attaches but never
reads), choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit that version. The
three visible test rows only run a fast format preview and the input checks; Kaggle's rerun on the hidden rows runs
everything, about 8.8 hours.
"""
    stage = f"""## 13. GLM-5.3 likelihood stage (estopadilla's V1, with the boosted-tree head)

The three cells below are the graft's V1 stage. The first sets the time box (all stage work ends 8.8 h after the
notebook starts) and the graft's conservative time-estimate constants. The second unpacks the V1 payload (a base64
tar.gz: estopadilla's `run_pipeline.py`, `segio.py`, `features.py`, our `v1_stage.py` with the 12 per-token contrasts
and the tree evaluation, and `ll_model.json` = his logistic head + the boosted trees, his head kept as the fallback),
checks the `ntl-segrun` binary's sha256 and the 6 + 19 GGUF segments, and smoke-tests the GPU on three rows. The third
runs the stage on the stack's final log-probabilities: rank by GLM mass, score as many passages as the time left
allows, blend the head into those rows, and rewrite the submission.
"""
    cells = list(nb["cells"])
    # drop our build's one-line header and the 0.684's header (the new head credits the same people)
    assert text(cells[0]).startswith(f"# {NAME}")
    assert text(cells[1]).startswith("# 🏆 SOTA 0.684")
    first_v1 = next(i for i, c in enumerate(cells) if text(c).startswith("# ---- ours, V1: likelihood under GLM-5.3-Flash"))
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    out = [md(head)] + cells[2:first_v1] + [md(stage)] + cells[first_v1:]
    nb_out = dict(nb, cells=out)
    assert [text(c) for c in nb_out["cells"] if c["cell_type"] == "code"] == codes
    meta = json.loads((HERE / NAME / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert UNREAD in meta["dataset_sources"]
    unread_files = ("night24_readout_banks", "official_only.joblib", "synthetic4404", "readouts.py")
    assert not any(f in c for c in codes for f in unread_files)
    meta.update(title=title, is_private=False, dataset_sources=[d for d in meta["dataset_sources"] if d != UNREAD])
    dest = HERE / "publish" / NAME
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{NAME}.ipynb").write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"publish/{NAME}: title {title!r}, {len(out)} cells, head {len(head)} chars, "
          f"{len(meta['dataset_sources'])} datasets")


if __name__ == "__main__":
    args = sys.argv[1:]
    t = None
    if "--title" in args:
        i = args.index("--title"); t = args[i + 1]; del args[i:i + 2]
    build(args[0], args[1], t)
