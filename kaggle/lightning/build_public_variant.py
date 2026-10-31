"""Build the public version of one scored build_variants.py kernel, ready for `kaggle kernels push`.

    python kaggle/lightning/build_public_variant.py <kernel> <kernel>=<public LB> [<kernel>=<public LB> ...]

The scores of every same-day variant (the control ntl-v28-ctrl included) go into the results table; the first kernel is
the one to publish. The output keeps that kernel's id, so the new version shows the score its version 1 earned. The code
cells are exactly what was scored; public_variant_intro.md is prepended as a markdown cell, and upstream's markdown share
figures are updated to match. Publish only if the score is at least the top public score the user set as the bar, and
never publish ntl-v28-ctrl (an unmodified fork).
"""
import json
import re
import sys
from pathlib import Path

import build_variants

HERE = Path(__file__).resolve().parent
OUT = HERE / "public"
REPO = HERE.parents[1]

PUBLIC = {  # kernel: title (with {SCORE}), tagline, results-table label, the change (markdown), upstream markdown fixes
    "ntl-v28-ctrl": None,
    "ntl-v28-helper15": (
        "{SCORE} | Spectral Helper 15%: LoRA+Llama+Qwen27B",
        "lightningv08's 0.642 ensemble, with Kirill Tymchishin's spectral helper at 15% instead of 10%.",
        "Kirill spectral helper 15% (Qwen 10%)",
        """## The change: Kirill's spectral helper gets 15% instead of 10%

```python
# run_fulltext_selected_ensemble.py, after ATLAS and before the Qwen3.8-27B mix and the final Sinkhorn
prior = .90*prior + .10*helper   # upstream V28
prior = .85*prior + .15*helper   # this notebook
```

The RECIPE status, its `helper_probability_share` (0.15) and the run script's guard that asserts the recipe change with
it. The helper is [Kirill Tymchishin's spectral readout bank](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404):
eight heads on the public LoRA24 features, fitted on the 2,699 official passages plus 4,404 synthetic ones.

**Why this knob.** The helper is the largest single step in this lineage. Adding it at 10% took lightningv08's own
notebook from 0.636 to 0.641 (Qwen at 5%) and from 0.637 to 0.642 (Qwen at 10%). That is one effect seen twice on
nearly the same rows, not two independent results. It also took fleongg's 0.636 to Kirill's 0.639. No public notebook
had scored any other share. Effective shares on rows without ATLAS move from reference 0.689 / complement 0.122 /
helper 0.090 / Qwen 0.100 to 0.650 / 0.115 / 0.135 / 0.100. Runtime is unchanged: the helper is already computed.""",
        [("The spectral heads contribute **10% after ATLAS**", "The spectral heads contribute **15% after ATLAS**"),
         ("`0.90 × R + 0.10 × H`", "`0.85 × R + 0.15 × H`"),
         ("**81% reference, 9% spectral heads and 10% Qwen**", "**76.5% reference, 13.5% spectral heads and 10% Qwen**")]),
    "ntl-v28-helper-125": (
        "{SCORE} | Spectral Helper 12.5%: LoRA+Llama+Qwen27B",
        "lightningv08's 0.642 ensemble, with Kirill Tymchishin's spectral helper at 12.5% instead of 10%.",
        "Kirill spectral helper 12.5% (Qwen 10%)",
        """## The change: Kirill's spectral helper gets 12.5% instead of 10%

```python
# run_fulltext_selected_ensemble.py, after ATLAS and before the Qwen3.8-27B mix and the final Sinkhorn
prior = .90*prior + .10*helper     # upstream V28
prior = .875*prior + .125*helper   # this notebook
```

The RECIPE status, its `helper_probability_share` (0.125) and the run script's guard that asserts the recipe change with
it. The helper is [Kirill Tymchishin's spectral readout bank](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404):
eight heads on the public LoRA24 features, fitted on the 2,699 official passages plus 4,404 synthetic ones.

**Why this knob.** The helper is the largest single step in this lineage. Adding it at 10% took lightningv08's own
notebook from 0.636 to 0.641 (Qwen at 5%) and from 0.637 to 0.642 (Qwen at 10%), one effect seen twice on nearly the
same rows. It also took fleongg's 0.636 to Kirill's 0.639. No public notebook had scored any other share; this is the
smaller of two steps I tried (see the table). Runtime is unchanged: the helper is already computed.""",
        [("The spectral heads contribute **10% after ATLAS**", "The spectral heads contribute **12.5% after ATLAS**"),
         ("`0.90 × R + 0.10 × H`", "`0.875 × R + 0.125 × H`"),
         ("**81% reference, 9% spectral heads and 10% Qwen**", "**78.75% reference, 11.25% spectral heads and 10% Qwen**")]),
    "ntl-v28-qwen-15": (
        "{SCORE} | Qwen3.8-27B at 15%: LoRA + Llama + Qwen",
        "lightningv08's 0.642 ensemble, with its full-text Qwen3.8-27B classifier at 15% instead of 10%.",
        "Qwen3.8-27B 15% (helper 10%)",
        """## The change: the full-text Qwen3.8-27B classifier gets 15% instead of 10%

```python
RECIPE['qwen_probability_share'] = 0.10   # upstream V28
RECIPE['qwen_probability_share'] = 0.15   # this notebook; run_fulltext_selected_ensemble.py passes it to blend_selected
```

The RECIPE status and the run script's guard that asserts the recipe change with it. The Qwen3.8-27B classifier is
lightningv08's own NF4 model with a LoRA on the last eight layers, run on the full text of every row.

**Why this knob.** In lightningv08's own versions, Qwen at 5% scored 0.641 and at 10% 0.642, and 5% to 10% also helped by
0.001 without the helper (0.636 to 0.637). Each step is within noise, so 15% tests whether the trend continues.
Runtime is unchanged: Qwen already scores every row.""",
        [("Qwen contributes **10% on every test response**", "Qwen contributes **15% on every test response**"),
         ("`0.90 × U + 0.10 × Q`", "`0.85 × U + 0.15 × Q`"),
         ("**81% reference, 9% spectral heads and 10% Qwen**", "**76.5% reference, 8.5% spectral heads and 15% Qwen**")]),
}
PUBLIC["ntl-v28-rare-helper15"] = (
    "{SCORE} | Rare-Class SVC + Spectral Helper 15%",
    "lightningv08's 0.642 ensemble with Kirill Tymchishin's rare-class n-gram SVC and his spectral helper at 15%.",
    "Kirill rare-class SVC + spectral helper 15%",
    """## The changes: Kirill's rare-class SVC, and his spectral helper at 15%

1. **The main n-gram SVC is Kirill Tymchishin's rare-class readout.** V28's cached `ntl627-svc-prefit` is replaced by
   [kirilltym/ntl-rare-class-svc-1516](https://www.kaggle.com/datasets/kirilltym/ntl-rare-class-svc-1516) (v1, CC BY 4.0):
   the same four TF-IDF views, label order and loader, refit on 79,998 passages, which are the 2,699 official ones,
   the 75,783 public pool passages and 1,516 extra GLM-5.3, GLM-5.3-flash and Kimi-K3 passages
   ([Daniel Andres Miranda's](https://www.kaggle.com/datasets/danielandresmiranda/ntl-openrouter-glm-kimi)). The idea
   was shown by [JN Liang](https://www.kaggle.com/code/jnliang/0-640-kirilltym-0-639-glm-kimi-svc-pool-patch)
   (kirilltym 0.639 to 0.640), and Kirill applied it to V28 in
   [his own fork](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature). It still enters the blend
   at weight 1.
2. **Kirill's spectral helper gets 15% instead of 10%**:

```python
# run_fulltext_selected_ensemble.py, after ATLAS and before the Qwen3.8-27B mix and the final Sinkhorn
prior = .90*prior + .10*helper   # upstream V28
prior = .85*prior + .15*helper   # this notebook
```

Relative to Kirill's V28 fork, the helper share is the only change. **Why the helper share:** adding the helper at 10%
took lightningv08's own notebook from 0.636 to 0.641 (Qwen at 5%) and from 0.637 to 0.642 (Qwen at 10%), one effect seen
twice on nearly the same rows, and no public notebook had scored any other share. Runtime is unchanged. A resolver
accepts exactly one mount layout of the rare-class dataset, and the self-test pins all 11 of its files by sha256.""",
    PUBLIC["ntl-v28-helper15"][4])
PUBLIC["ntl-v28-dual"] = (
    "{SCORE} | Dual-Head Qwen3.8-27B: LoRA + Llama + Qwen",
    "lightningv08's 0.642 ensemble, with the 27B checkpoint's own prompt-centred head switched on (Roman Tamrazov's dual head).",
    "dual-head Qwen3.8-27B (Qwen 10%, helper 10%)",
    """## The change: the 27B checkpoint's prompt-centred head, after Roman Tamrazov

lightningv08's full-text Qwen3.8-27B checkpoint has two trained heads, `plain` and `centered` (Linear 40960 -> 18), but
V28 only calls `plain`. Following [Roman Tamrazov's dual head](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i):

1. group the test passages with `topic_groups` at threshold .90 and keep clusters of 8-18 rows;
2. inside each kept cluster, centre the pooled 40960-d features on the cluster mean and score them with `centered`;
3. mix `dual = (1-a)*softmax(plain) + a*softmax(centered)`, with a = .35 where the two heads agree on the argmax and
   .18 where they disagree.

The dual probabilities replace the plain Qwen probabilities inside V28's unchanged 10% Qwen share. Rows outside such
clusters get V28's arithmetic exactly, up to the final Sinkhorn. Qwen 10% and Kirill's helper 10% are unchanged.

**What to expect.** The centred head gets an effective 1.8-3.5% of the final mix, and only on rows in 8-18-row topic
clusters. On train, no rows qualify when a prompt has 3 or fewer answers and about 90% do at 9 or more, so how much this
touches depends on how many answers per prompt the hidden test has. The idea is already in the ensemble for the two 14B
LoRA decoders (prompt-centred heads on clusters of 2-18); this extends it to the 27B head. The new code is in
`dual_head.py`: the worker runs a smoke test and the clustering before the 27B model loads, re-checks the plain head on
the saved features, and raises on any inconsistency instead of falling back.""",
    [])
LABELS = {"ntl-v28-ctrl": "unmodified V28 (control)"}
CHANGES = {"ntl-v28-rare-helper15": "two changes"}


def acknowledgements():
    text = (REPO / "docs" / "public_notebooks.md").read_text(encoding="utf-8")
    table = text[text.index("| Notebook |"):text.index("\n## What the survey found")].strip()
    if table.count("\n| [") != 13:
        raise SystemExit("expected the 13-row acknowledgements table in docs/public_notebooks.md")
    return table


def main(args):
    kernel, scores = args[0], dict(a.split("=", 1) for a in args[1:])
    if kernel not in scores or "ntl-v28-ctrl" not in scores or not PUBLIC.get(kernel):
        raise SystemExit("usage: build_public_variant.py <kernel> ntl-v28-ctrl=<score> <kernel>=<score> [...]; "
                         "never the control")
    title, tagline, _, change, fixes = PUBLIC[kernel]
    rows = ["| variant | public LB |", "|---|---:|"]
    for name, score in scores.items():
        label = LABELS.get(name) or PUBLIC[name][2]
        rows.append(f"| **{label}, this notebook** | **{score}** |" if name == kernel else f"| {label} | {score} |")
    intro = (HERE / "public_variant_intro.md").read_text(encoding="utf-8")
    title = title.replace("{SCORE}", scores[kernel])
    for key, value in {"TITLE": title, "CHANGES": CHANGES.get(kernel, "one change"), "TAGLINE": tagline, "CHANGE": change, "RESULTS": "\n".join(rows),
                       "ACK": acknowledgements()}.items():
        intro = intro.replace("{" + key + "}", value)
    if re.search(r"\{[A-Z]+\}", intro) or len(title) > 50:
        raise SystemExit(f"unfilled placeholder, or title over 50 characters: {title!r}")

    build_variants.build_variant(kernel)
    nb = json.loads((HERE / kernel / f"{kernel}.ipynb").read_text(encoding="utf-8"))
    for old, new in fixes:
        hits = [c for c in nb["cells"] if c["cell_type"] == "markdown" and old in c["source"]]
        if len(hits) != 1 or hits[0]["source"].count(old) != 1:
            raise SystemExit(f"markdown fix matched {len(hits)} cells: {old!r}")
        hits[0]["source"] = hits[0]["source"].replace(old, new)
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro})
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(title=title, is_private=False)
    out = OUT / kernel
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {out / meta['code_file']}: {meta['id']} titled {title!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
