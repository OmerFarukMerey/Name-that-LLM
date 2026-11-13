"""Publication page for a scored build_g692.py fork: our header, the upstream section notes, the scored code cells byte
for byte.

    python kaggle/nlm/publish_g692.py <variant> <score> <scored.ipynb> [--title "..."]

Writes kaggle/nlm/publish/<variant>/ (notebook + kernel-metadata.json, public) for kaggle/tools/quick_save.py.
"""
import json
import sys
from pathlib import Path

from build_e683 import URL as E683
from build_g692 import URL as G692, VARIANTS
from publish_e683 import CHEN, K, OURS682
from publish_l681 import ASSETS, text

HERE = Path(__file__).resolve().parent
ESTO = f"{K}/code/estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4"
ESTO_TRAIN = f"{K}/code/estopadilla/ntl-glm-likelihood-train"
ALEX = f"{K}/code/alexchilton/name-that-llm-0-684-all-public-what-failed"
HELIOS = f"{K}/code/heliosli/name-that-llm-lv683-a15"
LV = f"{K}/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b"
JL = f"{K}/datasets/jamesliebel/glm53-iq3xxs-segs-a-mirror"
# public LB of each build_g692.py fork, filled in as they score
RESULTS = {}


def changes(neg, sib):
    out = []
    if sib:
        out.append(("Sibling Re-rank", f"**the public [0.683]({E683})'s sibling-style re-ranking on the stack's final logits, "
                    "before the GLM stage reads them** (omkarpattanaik123's `sibling_style.py` and his `ensemble.py` block, "
                    "unchanged): within the Claude, Gemini, GPT and GLM families a small logistic model on stylometric counts "
                    "shifts at most 10% of a close call's probability between siblings, each family's total unchanged. On the "
                    "0.683, switching it off lost 0.005. CPU only (seconds), so the GLM stage keeps its time budget."))
    if neg:
        out.append(("Typography Rules", f"**the public [0.683]({E683})'s typographic negative evidence at the end of every "
                    "native fold's typography step**: -15 logits for the classes the official rows never show with a "
                    "passage's spaced em dash (all but claude-opus-4-6, claude-opus-5, glm-5.3, glm-5.3-flash), curly "
                    "apostrophe or curly opening quote (the Claude and GLM classes)."))
    return out


def build(name, score, scored, title=None):
    neg, sib = VARIANTS[name]
    nb = json.loads(Path(scored).read_text(encoding="utf-8"))
    built = json.loads((HERE / name / f"{name}.ipynb").read_text(encoding="utf-8"))
    codes = [text(c) for c in nb["cells"] if c["cell_type"] == "code"]
    assert codes == [text(c) for c in built["cells"] if c["cell_type"] == "code"], "scored code differs from the build"
    ch = changes(neg, sib)
    short = {"Sibling Re-rank": "Sibling", "Typography Rules": "Typo Rules"}
    title = title or f"{score} | GLM Likelihood + " + " + ".join(short.get(k, k) for k, _ in ch)
    assert 5 <= len(title) <= 50, (title, len(title))
    rows = "".join(f"| our fork: {', '.join(k for k, _ in changes(*VARIANTS[v]))} | {s} |\n"
                   for v, s in sorted(RESULTS.items(), key=lambda x: (x[1], x[0]), reverse=True) if v != name)
    head = f"""# {title}

**Public LB {score}** · the public 0.692 with {", ".join(k for k, _ in ch)} · T4 ×2 · internet off · every input is public

**In short.** This is mathleteutd's [GLM-5.3 Likelihood + GSQ 3%]({G692}) (version 3, public 0.692), run unchanged except for the step{'s' if len(ch) > 1 else ''} below, both taken from the public [0.683]({E683}). They run on the CPU in seconds, so the GLM likelihood stage keeps all of its time. Nothing is retrained.

## What changed

* {chr(10).join("* " + v for _, v in ch)[2:]}

Every other weight, model and correction is the 0.692's. The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Notebook | Public LB |
|---|---:|
| [our 0.682]({OURS682}): lightningv08's 0.681 + GSQ 12% + Sinkhorn target mix 0.50 | 0.682 |
| [estopadilla's 0.689]({ESTO}): alexchilton's all-public stack (our 0.682 recipe) + the GLM likelihood stage | 0.689 |
| [mathleteutd's 0.692]({G692}): the 0.689 with GSQ 12% -> 3% | 0.692 |
{rows}| **this notebook** | **{score}** |

Each row is one submission, so read single steps of 0.001 with care: the public board scores only part of the test set.

## How it works

1. **The stack** (lightningv08's [0.681]({LV}) line, as alexchilton runs it): LoRA24 read-outs of Qwen3-14B, Qwen3-8B, Llama-3.1-8B, Nishant Dahal's two-stage Qwen3-14B and Qwen2.5-14B, a DeBERTa encoder, the Open Pool text heads, the complementary stack, ATLAS on low-margin rows, the Qwen3.8-27B GSQ classifier at 3%, and CHEN Han's five-fold Qwen3-Embedding-8B expert at 40%; per fold soft Sinkhorn balancing (target mix 0.50) and typography, then the fold mean.
2. **This notebook's steps**{' (above)' if ch else ''}: {"; ".join(k for k, _ in ch)} on those logits.
3. **The GLM stage** (estopadilla): the texts are ranked by the stack's probability of glm-5.3 + glm-5.3-flash; as many top texts as the time left allows (at most 30%) are scored under the real GLM-5.3-Flash and GLM-5.3 (jamesliebel's public IQ3_XXS GGUF segments, run layer by layer by estopadilla's `ntl-segrun`); a small logistic head trained on those likelihood features ([his training notebook]({ESTO_TRAIN})) is blended into the stack's logits on the scored texts only.

## Credits and licences

* **mathleteutd**: the [0.692]({G692}) this one runs (GSQ 3%). **estopadilla (José Padilla)**: the [GLM likelihood stage]({ESTO}), `ntl-segrun` and the trained head. **jamesliebel**: the public [GLM-5.3 / GLM-5.3-Flash GGUF segment mirrors]({JL}).
* **omkarpattanaik123 (Omkar Pattanaik)**: the sibling-style re-ranking and typographic negative evidence ported here from the [0.683]({E683}).
* **alexchilton**: the [all-public stack]({ALEX}); **heliosli**: the [mount shim]({HELIOS}); **lightningv08 (Vladislav Lavrentev)**: the pipeline, the Qwen3.8-27B GSQ3 classifier and the [asset dataset]({ASSETS}); **CHEN Han**: the native expert ([.679]({CHEN})); the model, adapter and data authors named in the cells below; crtproduction for the Open Pool (CC BY-SA 4.0).
* **Models.** GLM-5.3 / GLM-5.3-Flash: Z.ai, under their own licence. Qwen models: Alibaba Qwen team, under their own licences. **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/). Each input keeps its own licence.

Our earlier public notebook: [0.682 | 0.681 + Qwen3.8-27B GSQ 12% + Mix 0.50]({OURS682}).

## Running it

Attach the inputs below (the same as the 0.692's), choose **GPU T4 ×2** and Internet **off**, then *Save Version → Save & Run All* and submit that version. The three visible test rows only run a fast format preview and the input checks; Kaggle's rerun on the hidden rows runs everything, about 8.7 hours.
"""
    cells = list(nb["cells"])
    # drop our build's one-line header and mathleteutd's header (the new head credits the same people)
    assert text(cells[0]).startswith(f"# {name}")
    assert text(cells[1]).startswith("# Name That LLM | GLM-5.3 Likelihood + GSQ 3%")
    out = [{"cell_type": "markdown", "metadata": {}, "source": head}] + cells[2:]
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
