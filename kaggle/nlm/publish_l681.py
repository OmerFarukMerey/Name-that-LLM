"""Publication page for a scored build_l681.py fork: our header, lightningv08's own section texts (numbers updated to the
variant), the scored code cells byte for byte.

    python kaggle/nlm/publish_l681.py <variant> <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_l681 import VARIANTS

HERE = Path(__file__).resolve().parent
K = "https://www.kaggle.com"
LV681 = f"{K}/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b"
LV680 = f"{K}/code/lightningv08/0-680-modernbert-lora-sparse-ensemble"
CHEN = f"{K}/code/concyclics/0-679"
HELIOS = f"{K}/code/heliosli/name-that-llm-lv681-a15"
ABOOD = f"{K}/code/aboodps/ntl-672-sinkhorn-050"
ASSETS = f"{K}/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets"
OURS = f"{K}/code/omerfarukmerey/0-671-all-public-response-corpus-pinned-pool"
# public LB of each build_l681.py fork (submitted 2026-11-09)
RESULTS = {"ntl-l681-tm50": "0.680", "ntl-l681-g09": "0.680", "ntl-l681-g09-tm50": "0.680",
           "ntl-l681-g12-tm50": "0.682", "ntl-l681-n35-g09-tm50": "0.676"}


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def changes(tm, gsq, native):
    out = []
    if tm != ".35":
        out.append(("Target Mix 0.50", f"**the balancing's target class shares 50% of the way from the blend's own "
                    f"shares to the official rows' instead of 35%** (aboodps's [0.669 step]({ABOOD})). The balancing is V31's soft Sinkhorn (`core_math.balance_logits`), "
                    "applied once per native fold after the geometric mixture, so this one default changes all five folds."))
    if gsq != "0.06":
        out.append((f"GSQ {int(float(gsq) * 100)}%", f"**the Qwen3.8-27B GSQ3 classifier at {int(float(gsq) * 100)}% inside the "
                    f"parent instead of 6%** (lightningv08's own step from 3% to 6% took his notebook from "
                    f"[0.680]({LV680}) to 0.681; `ensemble.py`'s assert on the share is updated with it)"))
    if native != "0.4":
        out.append((f"Native {int(float(native) * 100)}%", f"**the native expert's geometric share at "
                    f"{int(float(native) * 100)}% instead of 40%** (50% scored 0.679 at GSQ 3%, against 0.680 at 40%)"))
    return out


def build(name, score, scored, title=None):
    tm, gsq, native = VARIANTS[name]
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / name / f"{name}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    ch = changes(tm, gsq, native)
    assert ch
    short = {"Target Mix 0.50": "Mix 0.50"}
    title = title or next(x for x in (f"{score} | 0.681 + " + " + ".join(k for k, _ in ch),
                                      f"{score} | 0.681 + " + " + ".join(short.get(k, k) for k, _ in ch)) if len(x) <= 50)
    assert 5 <= len(title) <= 50, (title, len(title))
    pct = lambda x: f"{int(round(float(x) * 100))}%"
    head = f"""# {title}

**Public LB {score}** · lightningv08's 0.681 with {", ".join(k for k, _ in ch)} · T4 ×2 · internet off · every input is public

**In short.** This is lightningv08's [0.681 | Qwen3-Embedding LoRA + Qwen3.8-27B]({LV681}) (version 15, the one that scored), run unchanged except for {len(ch)} mixing setting{'s' if len(ch) > 1 else ''}; nothing is retrained and the runtime is the same. Its cells read fixed input paths written for one mount layout; heliosli's mount shim ([lv681 a15]({HELIOS}), also 0.681) maps them to whatever layout a copy gets, so this notebook runs as a plain copy.

## What changed

* {chr(10).join("* " + v for _, v in ch)[2:]}

Every other weight, model and correction is version 15's. The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Notebook | Native share | GSQ share | Other | Public LB |
|---|---:|---:|---|---:|
| [lightningv08 0.680]({LV680}) | 40% | 3% | | 0.680 |
| forks with the native share at 50% (feiyu26, dfenglei) | 50% | 3% | | 0.679 |
| [lightningv08 0.681]({LV681}), version 15 | 40% | 6% | | 0.681 |
| [heliosli lv681 a15]({HELIOS}) | 40% | 6% | ATLAS 15% | 0.681 |
""" + "".join(f"| our fork: {', '.join(k for k, _ in changes(*VARIANTS[v]))} | {pct(VARIANTS[v][2])} | {pct(VARIANTS[v][1])} | "
              f"{'target mix 0.50' if VARIANTS[v][0] != '.35' else ''} | {s} |\n"
              for v, s in sorted(RESULTS.items(), key=lambda x: (x[1], x[0]), reverse=True) if v != name) + f"""| **this notebook** | {pct(native)} | {pct(gsq)} | {"target mix 0.50" if tm != ".35" else ""} | **{score}** |

All rows are one submission each, so read single steps of 0.001 with care: the public board scores a part of the test set. What the grid does say: the native expert's share is the most sensitive setting (with GSQ 9% and target mix 0.50, 40% scored 0.680 and 35% 0.676; 50% scored 0.679 in the forks above), target mix 0.50 or GSQ 9% on their own did not help (0.680 each), and the Qwen3.8-27B classifier at 12% together with target mix 0.50 is the one setting that moved above 0.681.

## How the 0.681 works

lightningv08's parent is his weighted multi-model stack: LoRA24 read-outs of Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2), Llama-3.1-8B (1, its first 24 layers), Nishant Dahal's two-stage Qwen3-14B (2.5) and Qwen2.5-14B (1) and a DeBERTa encoder (1), the Open Pool text heads (an n-gram SVM at 12, TF-IDF and distortion members) pre-fitted on crtproduction's Open Pool version 13, the complementary stack at 15%, ATLAS at 10% on low-margin passages and his own trained Qwen3.8-27B GSQ3 classifier inside the parent. CHEN Han's ([concyclics, .679]({CHEN})) native expert, five prompt-fold Qwen3-Embedding-8B classifiers with rank-16 LoRA, read-outs at eight depths and plain plus prompt-centred heads, is then mixed in geometrically at {pct(native)}. Each of its five folds gets V31's balancing and typography step once; the five corrected probability matrices are averaged. All fitted states are in lightningv08's [asset dataset]({ASSETS}); training happened beforehand. lightningv08's own section notes follow below, with the numbers this version changes updated.

## Credits and licences

* **lightningv08 (Vladislav Lavrentev)**: the [0.681 notebook]({LV681}) this one runs, its whole pipeline, the trained Qwen3.8-27B GSQ3 classifier and the [asset dataset]({ASSETS}).
* **CHEN Han (concyclics)**: the native Qwen3-Embedding-8B multi-depth LoRA expert and its trained fold states ([.679]({CHEN})).
* **heliosli**: the mount shim ([lv681 a15]({HELIOS})).
""" + (f"* **aboodps**: the Sinkhorn target mix 0.50 ([NTL 672 Sinkhorn 050]({ABOOD})).\n" if tm != ".35" else "") + f"""* The authors lightningv08 credits for the parent: Roman Tamrazov, mercurydust and Areeb Ali (the public recipe, typography and Open Pool read-outs), Joseph Adamski (ATLAS), Nishant Dahal, Nuri Dudak, Dhruv Tibarewal, Lucas Boesen, prvsiyan, Kirill Tymchishin and the other model, adapter and data authors named in the modules below; crtproduction for the Open Pool (CC BY-SA 4.0).
* **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/). Qwen models: Alibaba Qwen team, under their own licences. Each input keeps its own licence.

Our own earlier, all-public line is [0.671 | All Public: Response Corpus + Pinned Pool]({OURS}).

## Running it

Attach the inputs below (the same as version 15's except kirilltym's spectral-heads dataset, which version 15 attaches but never reads), choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit that version. The three visible test rows only run a CPU format preview; Kaggle's rerun on the hidden rows runs everything.
"""
    cells = [c for c in nb["cells"]]
    # drop our build's one-line header (first cell) and lightningv08's intro (second): the new head replaces both
    assert cells[0]["cell_type"] == "markdown" and text(cells[0]).startswith(f"# {name}")
    assert cells[1]["cell_type"] == "markdown" and text(cells[1]).startswith("# Native Embedding LoRA")
    out = [{"cell_type": "markdown", "metadata": {}, "source": head}] + cells[2:]
    # the shim went in before the first code cell, i.e. between the preview's note and its code: put the note back above it
    assert text(out[1]).startswith("## Exact public sample preview") and text(out[2]).startswith("## Mount shim")
    assert out[3]["cell_type"] == "code" and text(out[3]).startswith("# Mount shim added by heliosli")
    out[1:4] = [out[2], out[3], out[1]]
    for c in out[1:]:
        if c["cell_type"] != "markdown":
            continue
        s = text(c)
        s = s.replace("with a 6% share inside the parent mixture", f"with a {pct(gsq)} share inside the parent mixture")
        c["source"] = s
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
