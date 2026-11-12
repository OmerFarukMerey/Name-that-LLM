"""Publication page for a scored build_e683.py fork: our header, the upstream section notes (numbers updated to the
variant), the scored code cells byte for byte.

    python kaggle/nlm/publish_e683.py <variant> <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_e683 import URL as APEX, VARIANTS
from publish_l681 import ABOOD, ASSETS, CHEN, HELIOS, K, LV681, OURS as OURS671, text

HERE = Path(__file__).resolve().parent
OURS682 = f"{K}/code/omerfarukmerey/0-682-0-681-qwen3-8-27b-gsq-12-mix-0-50"
LV5F = f"{K}/code/lightningv08/0-680-five-fold-embeddings-qwen3-8-27b-lora"
EVGEN = f"{K}/code/evgendvorkin/name-that-llm-deep-forensic-eda-with-analogie"
WEMM = f"{K}/models/dangkhoa2016/tencent-wemm-embedding-4b"
Z684 = f"{K}/code/lbjyang/ntl-684-ozertuu-mirror"
ATLAS9 = f"{K}/code/ozertuu/ntl-atlas-v9-qwen-embed"
OQ4 = f"{K}/datasets/nuridudak/ntl-q14-lora-v4"
# public LB of each build_e683.py fork, filled in as they score
RESULTS = {}


def pct(x):
    return f"{int(round(float(x) * 100))}%"


def changes(tm, gsq, wemm, sib, atlas, oq=None):
    out = []
    if oq:
        out.append(("v4 LoRA" if float(oq) == 1 else f"v4 LoRA {pct(oq)}",
                    "**the two-stage Qwen3-14B member (weight 2.5) reads Nuri Dudak's [q14 LoRA v4](" + OQ4 + ")"
                    + (" alone" if float(oq) == 1 else f" at {pct(oq)} plus Nishant Dahal's two-stage LoRA at {pct(1 - float(oq))}")
                    + " instead of the two-stage LoRA alone**. This is the change behind the public [0.684](" + Z684 + "), which is "
                    "ozertuu's [Atlas V9](" + ATLAS9 + ") (0.679) with that member's adapters weighted 0.72 / 0.28: +0.005 "
                    "there. v4 uses the same recipe as the two-stage LoRA (Qwen3-14B-Base, 27 layers kept, the same eight read-out "
                    "depths, 447 tokens, the same end token); its notes give out-of-fold balanced accuracy 0.636. The member's "
                    "checkpoint inventory gains v4's four files" + (", and `core_readout.member_scores` weights the adapters by "
                    "`adapter_weights` (equal weights otherwise, so every other member is unchanged)" if float(oq) != 1 else "") + "."))
    if gsq != "0.06":
        out.append((f"GSQ {pct(gsq)}", f"**the Qwen3.8-27B GSQ3 classifier at {pct(gsq)} inside the parent instead of 6%** "
                    f"(`ensemble.py`'s assert on the share is updated with it)"))
    if tm != ".35":
        out.append(("Mix 0.50", "**the balancing's target class shares 50% of the way from the blend's own shares to the "
                    f"official rows' instead of 35%** (aboodps's [Sinkhorn 0.50]({ABOOD})). The balancing is V31's soft "
                    "Sinkhorn (`core_math.balance_logits`), applied once per native fold after the geometric mixtures, so this "
                    "one default changes all five folds."))
    if wemm != "0.07":
        out.append((f"WeMM {pct(wemm)}", f"**the WeMM-Embedding-4B expert's log-linear share at {pct(wemm)} instead of 7%** "
                    "(on the GSQ 12% + Mix 0.50 forks, 4% and 7% scored 0.681 and 10% 0.674)"))
    if sib != "0.10":
        out.append(("No Re-ranker" if float(sib) == 0 else f"Re-ranker {pct(sib)}",
                    "**the sibling-style re-ranking " + ("switched off** (its cap `max_alpha` 0.10 -> 0; the module still "
                    "runs and returns the logits unchanged)" if float(sib) == 0 else f"capped at {pct(sib)} instead of 10%**")
                    + ". Our offline check of that step, re-implemented on a prompt-grouped out-of-fold prediction bank of the "
                    "official rows, found no gain: 0.0007 lower balanced accuracy at caps 0.05 and 0.10, 0.0045 lower at 0.20."))
    if atlas != ".15":
        out.append((f"ATLAS {pct(atlas)}", f"**ATLAS at {pct(atlas)} on the low-margin passages instead of 15%** (version 15's "
                    "value; heliosli's 15% scored the same 0.681 on version 15)"))
    return out


def extra(sib, oq=None):
    e = "typography" if float(sib) == 0 else f"re-ranker {pct(sib)} + typography"
    return e + ("" if not oq else " + v4 LoRA" if float(oq) == 1 else f" + v4 LoRA {pct(oq)}")


def row(label, native, wemm, gsq, tm, atlas, extra, score):
    return f"| {label} | {native} | {wemm} | {gsq} | {tm} | {atlas} | {extra} | {score} |\n"


def build(name, score, scored, title=None):
    tm, gsq, wemm, sib, atlas, *rest = VARIANTS[name]
    oq = rest[0] if rest else None
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / name / f"{name}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    ch = changes(tm, gsq, wemm, sib, atlas, oq)
    assert ch
    title = title or f"{score} | 0.683 + " + " + ".join(k for k, _ in ch)
    assert 5 <= len(title) <= 50, (title, len(title))
    sibrows = "".join(row(f"our fork: {', '.join(k for k, _ in changes(*VARIANTS[v]))}", "40%", pct(VARIANTS[v][2]),
                      pct(VARIANTS[v][1]), "0" + VARIANTS[v][0], pct(VARIANTS[v][4]), extra(VARIANTS[v][3], *VARIANTS[v][5:]), s)
                  for v, s in sorted(RESULTS.items(), key=lambda x: (x[1], x[0]), reverse=True) if v != name)
    head = f"""# {title}

**Public LB {score}** · omkarpattanaik123's 0.683 with {", ".join(k for k, _ in ch)} · T4 ×2 · internet off · every input is public

**In short.** This is omkarpattanaik123's [Sovereign Titan Apex]({APEX}) (version 1, public LB 0.683), run unchanged except for the {len(ch)} setting{'s' if len(ch) > 1 else ''} below. Nothing is retrained and the runtime is the same. The table shows every variant we scored around it, including the ones that lost.

## What changed

* {chr(10).join("* " + v for _, v in ch)[2:]}

Every other weight, model and correction is version 1's. The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Notebook | Qwen3-Emb. share | WeMM share | GSQ share | Target mix | ATLAS | Extra steps | Public LB |
|---|---:|---:|---:|---:|---:|---|---:|
""" + row(f"[lightningv08 0.681]({LV681}), version 15", "40%", "", "6%", "0.35", "10%", "", "0.681") \
        + row(f"[our 0.682]({OURS682})", "40%", "", "12%", "0.50", "10%", "", "0.682") \
        + row(f"[omkarpattanaik123 0.683]({APEX}), version 1", "40%", "7%", "6%", "0.35", "15%", extra("0.10"), "0.683") \
        + sibrows + row("**this notebook**", "40%", pct(wemm), pct(gsq), "0" + tm, pct(atlas), extra(sib, oq), f"**{score}**") + f"""
Each row is one submission, so read single steps of 0.001 with care: the public board scores only part of the test set.

## How it works

* **Parent.** lightningv08's weighted multi-model stack from his [0.681]({LV681}): LoRA24 read-outs of Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2), Llama-3.1-8B (1, its first 24 layers), {'Nishant Dahal' + chr(39) + 's two-stage Qwen3-14B (2.5' if not oq else 'the two-stage Qwen3-14B member (2.5, here ' + ('Nuri Dudak' + chr(39) + 's q14 LoRA v4' if float(oq) == 1 else f'q14 LoRA v4 {pct(oq)} + Nishant Dahal' + chr(39) + f's two-stage LoRA {pct(1 - float(oq))}')}), Nishant Dahal's Qwen2.5-14B (1) and a DeBERTa encoder (1), the Open Pool text heads pre-fitted on crtproduction's Open Pool version 13 (an n-gram SVM at 12, TF-IDF and distortion members), the complementary stack at 15%, ATLAS at {pct(atlas)} on low-margin passages{" (heliosli's step)" if atlas == ".15" else ""} and his trained Qwen3.8-27B GSQ3 classifier, here at {pct(gsq)}.
* **Two embedding experts by CHEN Han** ([concyclics, .679]({CHEN})). Five prompt-fold Qwen3-Embedding-8B classifiers (rank-16 LoRA, read-outs at eight depths, plain and prompt-centred heads) enter the geometric mixture at 40%. Then five Tencent WeMM-Embedding-4B fold states enter a second log-linear mixture at {pct(wemm)}, one model copy per T4 (folds 0-2 and 3-4), as wired in lightningv08's [five-fold embeddings notebook]({LV5F}).
* **Per fold, then averaged.** Each of the five fold mixtures gets V31's soft Sinkhorn balancing (strength 0.30, target mix 0{tm}) and the hierarchical typography step once. omkarpattanaik123's typographic negative evidence then subtracts 15 logits from the classes that never use a passage's spaced em dash, curly apostrophe or curly opening quote in the official rows. The five corrected matrices are averaged.
* **Sibling-style re-ranking** (omkarpattanaik123). Within the Claude, Gemini, GPT and GLM families, a small logistic model on stylometric counts, rated by prompt-grouped out-of-fold balanced accuracy, shifts {"no probability here (cap 0)" if float(sib) == 0 else f"at most {pct(sib)} of a close call's probability"} between siblings. Each family's total mass is unchanged.

All fitted states are in lightningv08's [asset dataset]({ASSETS}); training happened beforehand. The upstream section notes follow below, with the numbers this version changes updated.

## Credits and licences

* **omkarpattanaik123 (Omkar Pattanaik)**: the [0.683 notebook]({APEX}) this one runs: the combination, the sibling-style re-ranking and the typographic negative evidence. evgendvorkin's [forensic EDA notebook]({EVGEN}) runs the same inference cells.
* **lightningv08 (Vladislav Lavrentev)**: the [0.681]({LV681}) pipeline and parent stack, the trained Qwen3.8-27B GSQ3 classifier, the WeMM wiring in his [five-fold embeddings notebook]({LV5F}) and the [asset dataset]({ASSETS}).
* **CHEN Han (concyclics)**: both native experts (Qwen3-Embedding-8B and WeMM-Embedding-4B) and their trained fold states ([.679]({CHEN})).
* **heliosli**: the mount shim and ATLAS 15% ([lv681 a15]({HELIOS})).
* **aboodps**: the Sinkhorn target mix 0.50 ([NTL 672 Sinkhorn 050]({ABOOD})).
""" + (f"* **ozertuu and lbjyang**: the adapter weighting ported here, from the [0.684]({Z684}) (lbjyang's copy; krshs90's and wadigongren's 0.684s are the same code) built on ozertuu's [Atlas V9]({ATLAS9}); **Nuri Dudak**: the [q14 LoRA v4]({OQ4}).\n" if oq else "") + f"""* The authors lightningv08 credits for the parent: Roman Tamrazov, mercurydust and Areeb Ali (the public recipe, typography and Open Pool read-outs), Joseph Adamski (ATLAS), Nishant Dahal, Nuri Dudak, Dhruv Tibarewal, Lucas Boesen, prvsiyan, Kirill Tymchishin and the other model, adapter and data authors named in the modules below; crtproduction for the Open Pool (CC BY-SA 4.0).
* **Models.** Qwen models: Alibaba Qwen team, under their own licences. [Tencent WeMM-Embedding-4B]({WEMM}) (Kaggle upload by dangkhoa2016): Tencent, under its own licence. **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/). Each input keeps its own licence.

Our earlier public notebooks: [0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50]({OURS682}) and [0.671 | All Public: Response Corpus + Pinned Pool]({OURS671}).

## Running it

Attach the inputs below (the same as version 1's except kirilltym's spectral-heads dataset, which version 1 attaches but never reads), choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit that version. The three visible test rows only run a CPU format preview; Kaggle's rerun on the hidden rows runs everything.
"""
    cells = list(nb["cells"])
    # drop our build's one-line header, omkarpattanaik123's header and lightningv08's older intro (50% / 3%)
    assert text(cells[0]).startswith(f"# {name}")
    assert text(cells[1]).startswith("# 🏆 Name That LLM: V28 Sovereign Titan Apex Fusion")
    assert text(cells[2]).startswith("# Native Embedding LoRA")
    out = [{"cell_type": "markdown", "metadata": {}, "source": head}] + cells[3:]
    # the shim cell sits between the preview's note and its code: give it its own note, put the preview's back above it
    assert text(out[1]).startswith("## Exact public sample preview")
    assert out[2]["cell_type"] == "code" and text(out[2]).startswith("# Mount shim added by heliosli")
    shim_note = {"cell_type": "markdown", "metadata": {}, "source": "## Mount shim (heliosli)\n\nThe cells below bind fixed "
                 "input paths for one mount layout; this cell maps each of them to wherever that source is mounted in this session."}
    out[1:3] = [shim_note, out[2], out[1]]
    hit = 0
    for c in out[1:]:
        if c["cell_type"] != "markdown":
            continue
        s = text(c)
        old = "with a 3% share inside the parent mixture"
        hit += s.count(old)
        c["source"] = s.replace(old, f"with a {pct(gsq)} share inside the parent mixture")
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
