"""Publication page for a scored build_z684.py fork: our header, the upstream module notes (numbers updated to the
variant), the scored code cells byte for byte.

    python kaggle/nlm/publish_z684.py <variant> <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_z684 import URL as Z684, VARIANTS
from build_e683 import URL as E683
from publish_e683 import ATLAS9, CHEN, K, LV681, OQ4, OURS682, pct
from publish_l681 import ASSETS, text

HERE = Path(__file__).resolve().parent
KRSHS = f"{K}/code/krshs90/qwen-embedding-clean-lora-ensemble-v14"
WADI = f"{K}/code/wadigongren/name-that-llm-0684-clean-baseline"
N14B = f"{K}/datasets/nishantdahal/ntl-handwriting-blend"
# public LB of each build_z684.py fork, filled in as they score
RESULTS = {}


def changes(oq, w, gsq, neg="0", sib="0.0", typo="0", wemm="0"):
    out = []
    if oq != "0.72":
        out.append(("v4 LoRA only" if float(oq) == 1 else f"v4 LoRA {pct(oq)}",
                    "**the two-stage Qwen3-14B member reads " + ("Nuri Dudak's q14 LoRA v4 alone" if float(oq) == 1 else
                    f"q14 LoRA v4 at {pct(oq)} and Nishant Dahal's two-stage LoRA at {pct(1 - float(oq))}") +
                    " instead of 72% / 28%**. The 0.684 itself is ozertuu's Atlas V9 (0.679) with this member moved from an "
                    "equal three-adapter average to 72% v4 / 28% two-stage: +0.005."))
    if w != "2.5":
        out.append((f"14B Member x{w}", f"**that member's ensemble weight at {w} instead of 2.5**"))
    if gsq != "0.03":
        out.append((f"GSQ {pct(gsq)}", f"**the Qwen3.8-27B GSQ3 classifier at {pct(gsq)} instead of 3%** (lightningv08's "
                    f"step from 3% to 6% took his line from 0.680 to [0.681]({LV681}))"))
    if float(wemm) > 0:
        out.append((f"WeMM {pct(wemm)}", f"**CHEN Han's second native expert, Tencent WeMM-Embedding-4B (five fold states), at a "
                    f"{pct(wemm)} log-linear share after the Qwen3-Embedding expert**, run as the public [0.683]({E683}) runs it: its "
                    "`wemm_embedding_worker.py` and `wemm_fusion.py` unchanged (wired by lightningv08 in his five-fold embeddings "
                    "notebook), one replica per T4 (folds 0-2 and 3-4), with a deadline after which unscored folds fall back to the "
                    "mean of the scored ones; any failure leaves the 0.684 path untouched. The 0.684 has only the Qwen3-Embedding "
                    "expert; on the 0.683, WeMM at 10% instead of 7% lost 0.007 and 4-5% about the same as 7%."))
    if typo == "1":
        out.append(("Typography Step", f"**the public [0.683]({E683})'s per-fold typography step on every native fold before the "
                    "mean**: a smoothed Bernoulli score per class from which of 37 typographic markers (dashes, curly quotes, "
                    "semicolons, list and heading marks, stock phrases) a passage contains, fitted on the official rows and "
                    "clipped to ±0.30 logits, gated to close calls, longer passages and passages with markers; then the "
                    "0.683's -15 negative rules (spaced em dash, curly apostrophe, curly opening quote). The 0.684 has no "
                    "typography step of its own; `typography_evidence` is the 0.683's, unchanged (V31 lineage)."))
    if neg == "1":
        out.append(("Typography Rules", f"**the public [0.683]({E683})'s typographic negative evidence on the final scores**: "
                    "-15 logits for the classes the official rows never show with a passage's spaced em dash (all but "
                    "claude-opus-4-6, claude-opus-5, glm-5.3 and glm-5.3-flash), curly apostrophe or curly opening quote "
                    "(the Claude and GLM classes). The 0.684 has no typography step of its own."))
    if float(sib) > 0:
        out.append(("Sibling Re-rank", f"**the public [0.683]({E683})'s sibling-style re-ranking on the final scores** "
                    "(omkarpattanaik123's `sibling_style.py`, unchanged): within the Claude, Gemini, GPT and GLM families a small "
                    "logistic model on stylometric counts shifts at most 10% of a close call's probability between siblings, "
                    "each family's total unchanged; here its out-of-fold check groups the official rows by prompt. On the "
                    "0.683, a cap of 0 or 0.05 instead of 0.10 lost about 0.005."))
    return out


def row(label, oq, w, gsq, score):
    return f"| {label} | {oq} | {w} | {gsq} | {score} |\n"


def build(name, score, scored, title=None):
    oq, w, gsq, *post = VARIANTS[name]
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / name / f"{name}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    ch = changes(oq, w, gsq, *post)
    assert ch
    title = title or f"{score} | 0.684 + " + " + ".join(k for k, _ in ch)
    assert 5 <= len(title) <= 50, (title, len(title))
    mix = lambda q: "v4 only" if float(q) == 1 else f"{pct(q)} / {pct(1 - float(q))}"
    sibrows = "".join(row(f"our fork: {', '.join(k for k, _ in changes(*VARIANTS[v]))}", mix(VARIANTS[v][0]), VARIANTS[v][1],
                          pct(VARIANTS[v][2]), s)
                      for v, s in sorted(RESULTS.items(), key=lambda x: (x[1], x[0]), reverse=True) if v != name)
    head = f"""# {title}

**Public LB {score}** · the public 0.684 with {", ".join(k for k, _ in ch)} · T4 ×2 · internet off · every input is public

**In short.** This is the public [0.684]({Z684}) (lbjyang's copy, version 1; [krshs90's]({KRSHS}) and [wadigongren's]({WADI}) 0.684s are the same code), run unchanged except for the {len(ch)} setting{'s' if len(ch) > 1 else ''} below. Nothing is retrained and the runtime is the same or shorter. The 0.684 is ozertuu's [NTL Atlas V9 Qwen Embed]({ATLAS9}) (version 2, public 0.679) plus one fix: the two-stage Qwen3-14B member now really weights its adapters, Nuri Dudak's [q14 LoRA v4]({OQ4}) at 72% and Nishant Dahal's [two-stage LoRA]({N14B}) at 28% (Atlas V9 averaged those two and q14 LoRA v1 equally). That one member moved the board by +0.005, so this notebook pushes the same lever further.

## What changed

* {chr(10).join("* " + v for _, v in ch)[2:]}

Every other weight, model and correction is the 0.684's. The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Notebook | 14B member: v4 / two-stage | Member weight | GSQ share | Public LB |
|---|---:|---:|---:|---:|
""" + row(f"[ozertuu Atlas V9]({ATLAS9}), version 2", "v1 + v4 + two-stage, equal", "2.5", "3%", "0.679") \
        + row(f"[the public 0.684]({Z684})", "72% / 28%", "2.5", "3%", "0.684") \
        + sibrows + row("**this notebook**", mix(oq), w, pct(gsq), f"**{score}**") + f"""
Each row is one submission, so read single steps of 0.001 with care: the public board scores only part of the test set.

## How it works

* **LoRA24 members.** Truncated Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2) and Llama-3.1-8B (1) with Dhruv Tibarewal's v5 LoRA adapters (five folds + full each), the two-stage Qwen3-14B member ({w}: {mix(oq)} q14 LoRA v4 / two-stage) and Nishant Dahal's Qwen2.5-14B (1): mean-pooled hidden states at eight depths, plain and prompt-centred ("twin") heads, Gaussian group heads on the two v5 members, and a DeBERTa encoder (1).
* **CHEN Han's native expert** ([concyclics, .679]({CHEN})): five prompt-fold Qwen3-Embedding-8B classifiers, read out at eight depths with plain and centred heads, mixed geometrically at 40%. The same embedding model clusters the test passages into prompt groups (cosine 0.70) for the centred heads.
* **Qwen3.8-27B GSQ3 classifier** (lightningv08) at {pct(gsq)}, and the complementary stylometric stack (DeBERTa-v3 token head + TF-IDF + LinearSVC) at 15%. The pre-fitted Open Pool SVM / TF-IDF / distortion heads are loaded but weighted 0.
* **Balancing.** Soft Sinkhorn at strength 0.30, target mix 0.35.

All fitted states come from the attached datasets ([lightningv08's asset dataset]({ASSETS}) and the adapter datasets); training happened beforehand. The upstream module notes follow below, with the numbers this version changes updated.

## Credits and licences

* **ozertuu (Ertuğrul Özer)**: the pipeline ([NTL Atlas V9 Qwen Embed]({ATLAS9})).
""" + ("* **omkarpattanaik123 (Omkar Pattanaik)**: the typography step, typographic negative evidence and sibling-style re-ranking ported from the public [0.683](" + E683 + ") (its typography follows mercurydust's and Roman Tamrazov's V31 line).\n" if post else "") + f"""* **lbjyang**: the public [0.684]({Z684}) this one runs (also published, byte for byte, by [krshs90]({KRSHS}) and [wadigongren]({WADI})).
* **Nuri Dudak**: the [q14 LoRA v4]({OQ4}) and v1 adapters. **Nishant Dahal**: the two-stage Qwen3-14B and Qwen2.5-14B LoRAs. **Dhruv Tibarewal**: the v5 LoRA24 adapters, the matched clean pool and the offline wheels.
* **CHEN Han (concyclics)**: the native Qwen3-Embedding-8B expert. **lightningv08 (Vladislav Lavrentev)**: the Qwen3.8-27B GSQ3 classifier, the pre-fitted states and the [asset dataset]({ASSETS}).
* Lucas Boesen, Roman Tamrazov, mercurydust, Areeb Ali, prvsiyan, pjmathematician, alexchilton (the Llama-3.1-8B mirror) and the other data and model authors whose inputs are attached; crtproduction for the Open Pool (CC BY-SA 4.0).
* **Models.** Qwen models: Alibaba Qwen team, under their own licences. **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/). Each input keeps its own licence.

Our earlier public notebook: [0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50]({OURS682}).

## Running it

Attach the inputs below (the same as the 0.684's), choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit that version. The visible three-row test only runs a fast format check; Kaggle's rerun on the hidden rows runs everything.
"""
    cells = list(nb["cells"])
    # drop our build's one-line header and the 0.684's own header (the new head replaces both)
    assert text(cells[0]).startswith(f"# {name}")
    assert text(cells[1]).startswith("# 🏆 SOTA 0.684")
    out = [{"cell_type": "markdown", "metadata": {}, "source": head}] + cells[2:]
    hit = 0
    for c in out[1:]:
        if c["cell_type"] != "markdown":
            continue
        s = text(c)
        if "`--gsq_share`: `0.03`" in s:
            hit += 1
            s = s.replace("`--gsq_share`: `0.03`", f"`--gsq_share`: `{gsq}`")
        c["source"] = s
    assert hit == 1
    nb_out = dict(nb, cells=out)
    assert [text(c) for c in nb_out["cells"] if c["cell_type"] == "code"] == codes
    meta = json.loads((HERE / name / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(title=title, is_private=False)
    dest = HERE / "publish" / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{name}.ipynb").write_text(json.dumps(nb_out, indent=1, ensure_ascii=False), encoding="utf-8")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"publish/{name}: title {title!r}, {len(out)} cells, head {len(head)} chars")


if __name__ == "__main__":
    args = sys.argv[1:]
    t = None
    if "--title" in args:
        i = args.index("--title"); t = args[i + 1]; del args[i:i + 2]
    build(args[0], args[1], args[2], t)
