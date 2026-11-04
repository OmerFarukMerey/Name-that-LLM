"""Public, documented version of a scored nlm-v27 fork (only for a fork that beats the top public Code score).

    python kaggle/nlm/publish_nlm.py <kernel> <results.json> <scored.ipynb>

<results.json>: {"score": "0.65x", "rows": [[version, change, public LB], ...]} from submissions/INDEX.md, optionally
"dp_check" (a sentence on ntl-l24-dp-check for the faster l24_predict), "did_not_transfer" ([[change, offline,
public LB], ...] rows added to that table) and, for ntl-v31-*, "a_note" (text appended to the ATLAS section) and
"llama_cache_gone" (true: say that avazaku's Llama cache is no longer available and how to rerun without it).
<scored.ipynb>: the scored version as Kaggle has it (kaggle/tools/pull_version.py <user/kernel> <version> <out_dir>, run through kaggle/tools/kglrun.py).
Writes kaggle/nlm/publish/<kernel>/ (notebook + kernel-metadata.json). The code cells are the scored version's,
byte for byte (checked against kaggle/nlm/<kernel>/, the build that was pushed); the private intro cell is replaced
by the documentation and a short heading goes before each code cell. The document is assembled from what the variant
changes (build_nlm.py's VARIANTS): our Ministral-3-8B member ("mn8" in the name, weight in WEIGHT), the second ATLAS
bundle ("-a2", exp019), ATLAS-8B at 20/30/40% ("-a20" / "-a30" / "-a40"; "-m45": also on rows with top-2 margin
below 0.45), Sinkhorn balancing at 0.5 / 0.25 instead of 1.0 ("-sk50" / "-sk25", exp024), the faster l24_predict ("-dp",
exp018), and for "ntl-v38-*" the base is Roman Tamrazov's 0.654 notebook, i.e. his typography step at the end of infer.py
(exp020). For "ntl-v31-*" (exp025) the base is mercurydust's 0.660 notebook ("V31", on Roman's) and build_v31
writes its own document: the faster run in every one, ATLAS-8B's weight ("-a20" / "-a30") and the typography step
with its margin gate only ("-mg"). The metadata keeps the
kernel id, so pushing it adds a version to the scored kernel (Kaggle keeps showing the scored version's LB score),
sets a readable title (which renames the slug) and makes it public. Pushing also reruns the commit: the 3-row
FAST_COMMIT path, ~5-18 min.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
K = "https://www.kaggle.com"

# Ministral weight of each variant that has the member (kept in step with build_nlm.py's VARIANTS)
WEIGHT = {"ntl-nlm27-mn8-noatlas": 1.0, "ntl-nlm27-mn8w2-noatlas": 2.0, "ntl-nlm27-mn8w3-noatlas": 3.0,
          "ntl-nlm27-mn8-deadline": 1.0, "ntl-nlm27-mn8w2-deadline": 2.0, "ntl-nlm27-mn8w3-deadline": 3.0,
          "ntl-nlm27-mn8w2-dp": 2.0, "ntl-nlm27-mn8w3-dp": 3.0, "ntl-nlm27-mn8w2-dp-a2": 2.0, "ntl-nlm27-mn8w2-dp-a20": 2.0,
          "ntl-v38-mn8w2-dp": 2.0, "ntl-v38-mn8w2-dp-a2": 2.0,
          "ntl-v38-mn8w3-dp": 3.0, "ntl-v38-mn8w2-dp-a20": 2.0, "ntl-v38-mn8w3-dp-a20": 3.0}
WHO = "{K}/code/romantamrazov/name-that-llm-who-am-i"
SOFT_SK = "{K}/code/peterdreamhan/name-that-llm-soft-sinkhorn-0-5-t4"
ATLAS_TEXT = {
    "noatlas": ("**ATLAS is off** (`--atlas_w 0`). In the 0.651 recipe ATLAS must finish by a fixed 8.3 h mark or is "
                "dropped whole; switching it off makes the run deterministic and leaves the Ministral member all the time "
                "it needs. Our ATLAS-off control without Ministral scored {control} (the recipe with ATLAS: 0.651)."),
    "deadline": ("**ATLAS keeps the rows it finishes.** Upstream projects ATLAS's finish time at checkpoints and, if the "
                 "projection passes the fixed budget, raises and drops ATLAS entirely. Here ATLAS scores rows shortest "
                 "first until the deadline itself and mixes in every row it finished. With time to spare this is "
                 "upstream's ATLAS exactly; short of time it keeps the rows that fit instead of none, which matters "
                 "because the Ministral member takes ~45 min of the same 9 h."),
}
ATLAS_TEXT["dp"] = ATLAS_TEXT["deadline"].replace("~45 min of the same 9 h.", "time from the same 9 h. And the three "
                                                  "8B members now run on both GPUs at once (below), so ATLAS has more "
                                                  "time than in the 0.651 recipe itself.")
ATLAS_TEXT["dp-plain"] = ("ATLAS scores rows shortest first up to its deadline and keeps every row it finished (upstream "
                          "drops ATLAS whole when a projection of its finish passes the fixed budget); here the run "
                          "reaches ATLAS ~1.5 h earlier than the 0.651 recipe itself, so it finishes with time to spare.")
ATLAS_TEXT["dp-plain-m45"] = ATLAS_TEXT["dp-plain"].replace("so it finishes with time to spare.",
                                                            "which pays for the extra passages it scores here.")
PAY = [   # the changes that buy time back without changing a prediction
    "* **Llama-3.1-8B is copied to local disk before loading.** Read straight from the notebook-output mount, "
    "`from_pretrained` took ~2,050 s (CPU probe) and ~2,140 s (our GPU run); one sequential copy takes 88-135 s and the "
    "load from local disk ~60 s. Same bytes, same predictions.",
    "* **The TF-IDF member is not fitted.** Its blend weight is 0 (`--tfidf_w 0`), so its ~6 min fit multiplied into "
    "nothing (maverickss26's restructured 0.651 skips it as well).",
]
PAY_DP = ("* **The 8B members use both GPUs at once.** `l24_predict` loads every member with `device_map='auto'`, which "
          "splits even a 24-layer 8B model (~11 GiB in fp16) across the two T4s, so one GPU waits while the other runs "
          "its half of the layers, and it averages the hidden states on the CPU (eight ~100 MB copies per batch). Now "
          "every member averages on the GPU, and for the {n8} 24-layer 8B members ({names}) a full copy sits on each "
          "GPU and the recipe's batches, in its order, alternate between them in two threads; if the largest batch "
          "would leave under 1 GiB free it falls back to the split copy. Same computation per row, about half the "
          "time.{dp_check}")
MN8_SECTION = """## The new member: Ministral-3-8B-Base, two stages

* **Recipe** — Lucas Boesen's public LoRA24 recipe: the decoder truncated to 24 layers, LoRA r16 on every linear layer, masked mean-pool of 8 depths (2, 4, 6, 8, 12, 16, 20, 24) with one LayerNorm each, a plain head and a prompt-centred head trained on same-prompt triplets. Dhruv's two stages: stage 1 on his 120k public pool with class weights, stage 2 two epochs on the 2,699 official rows (lr 5e-5, heads 5e-4), once per prompt-grouped fold and once on all rows. At inference the six adapters (5 folds + full fit) are merged into the base in turn and averaged, exactly like the recipe's own members.
* **Training** — [stage 1]({K}/code/omerfarukmerey/ntl-l24-mn8-s1): 5,293 steps in 10 h on T4 ×2 (fp16, data-parallel, 0 overflow) = 0.99 pool epochs. [Stage 2]({K}/code/omerfarukmerey/ntl-l24-mn8-s2): 2 h. Both are public.
* **Out of fold** (5 prompt-grouped folds, the split of the recipe's own fold adapters): **0.605 balanced accuracy alone** (the recipe's Llama member 0.596, Qwen3-8B 0.615, Qwen3-14B 0.634). It is also the most different: its argmax agrees with the other decoders on 69-71% of passages, they agree with each other on 70-73%.
* **Offline blend test** — a replica of the recipe's core (its three fold-adapter members + the refitted SVM + a DeBERTa stand-in, the same weighted log-probability blend and Sinkhorn): Ministral adds +0.0004 at weight 1.0, +0.0037 at 1.5, +0.0041 at 2.0 and +0.0059 at 3.0; in the Llama member's place it adds +0.0048 where Llama adds +0.0029. That replica is only a yardstick (encoder gains of +0.008 in it lost 0.006 on the LB), which is why weights 1, 2 and 3 went to the leaderboard. Per class it helps claude-sonnet-5, gemini-3.8-flash, gpt-5.6-luna and kimi-k3, and costs a little on gemini-3.6/3.7-flash and gpt-5.6-terra.
"""
A2_SECTION = """## A second ATLAS: Qwen3-4B

ATLAS ([Ko et al.](https://arxiv.org/abs/2610.19583)) scores a passage under 18 conditioned copies of one frozen language model, one per candidate author (a learned prefix and LoRA weights per source), and attributes it to the author under whom it is most likely. The recipe mixes Joseph Adamski's Qwen3-8B bundle at 10% into the passages whose top-2 margin is below 0.3, and that is worth **+0.008** on the public LB (our ATLAS-off control scored 0.643).

* **The second bundle** — harshitakumari256's [ntl-atlas-bundle]({K}/datasets/harshitakumari256/ntl-atlas-bundle): an independent implementation on **Qwen3-4B-Base**, trained differently (one epoch on ~35k public passages, then three on the official rows; readout: softmax of a fitted temperature × mean token log-likelihood). It ships out-of-fold scores for all 2,699 training passages (5 folds grouped by prompt): **0.427 balanced accuracy alone**, weak as a classifier but a different kind of evidence.
* **Offline test** — in our out-of-fold replica of the recipe's core (which has no ATLAS-8B), mixing it at 10% into the passages with top-2 margin below 0.3 (55% of them; 3.9% of argmaxes change) adds **+0.0074** [95% prompt bootstrap +0.0004, +0.0145]; at 20% +0.0078, at 30% +0.0086, with our Ministral member in the blend +0.005 to +0.007.
* **How it runs** — after ATLAS-8B, on the same passages, at 10%: the two mixes together keep 80% of the ensemble's probabilities. The 4B fits one T4, so one worker process per GPU scores half the passages, shortest first, with the bundle's own scoring code, saving after every 16, until ATLAS's deadline; passages it has not reached keep ATLAS-8B's mix only. On T4 ×2 it loads in 132 s and then scores ~87 tokens/s per GPU: ~35 min for ~2,200 passages, which the faster run below pays for.
* **A caution** — the bundle's final weights saw every training passage (it labels 400 of them 100% right), so only its out-of-fold file can rate it.
"""
A20_SECTION = """## ATLAS at {pct}%

ATLAS ([Ko et al.](https://arxiv.org/abs/2610.19583), Joseph Adamski's Qwen3-8B bundle) is a source-conditioned likelihood: the passage is scored under 18 conditioned copies of one frozen language model, one per candidate author. Its author mixed it at 10% into seven public notebooks (+0.002 to +0.007 each) and never published another weight. In this recipe it is worth **+0.008** (our ATLAS-off control scored 0.643, the recipe 0.651), on the passages whose top-2 margin is below 0.3.

At 10%, the mix can flip a passage only where the ensemble's top-2 margin is below ~0.11; at {pct}% the reach grows to ~{reach}, {inside}. This version mixes ATLAS at **{pct}%** (`--atlas_w {flag}`). {cost}
"""
A_INSIDE = {False: "still inside the same passages", True: "beyond the 0.3 margin that selects the passages ATLAS scores, so that cut now sets the limit"}
A_COST = "It costs no time: ATLAS scores the same passages either way."
M45_INSIDE = ("and this version also widens the passages ATLAS scores from a top-2 margin below 0.3 to below 0.45 "
              "(`--atlas_margin 0.45`) to cover it")
M45_COST = ("ATLAS scores those extra passages shortest first up to its fixed deadline, so if time runs short the longest "
            "ones keep the ensemble's answer; the faster run below pays for them.")
V38_SECTION = """## The 0.654 step: a typography likelihood (Roman Tamrazov)

Roman Tamrazov's [Name That LLM | Who am I?]({who}) (public **0.654**) is the 0.651 recipe plus one step at the end of `infer.py`, copied here verbatim:

* **Markers** — 34 yes/no features of a passage: a spaced em dash, an em dash, an en dash, curly or angle quotes, a straight double quote, a semicolon, a colon, parentheses, brackets, `**`, a backtick, an ellipsis, a bullet, a newline, a blank line, "e.g.", "i.e.", "However,", "Overall,", "In conclusion", "Therefore,", "Moreover,", a tab, " - ", " = ", a dash list, a numbered list.
* **Likelihood** — for each model, the share of its training passages that show each marker ((hits + 2) / (n + 4), clipped to [0.015, 0.985]). A passage's evidence for a model is its mean Bernoulli log-likelihood over the 34 markers, centred over the 18 models, doubled and clipped to ±0.35.
* **Gate** — the evidence is added to the ensemble's log-probabilities after Sinkhorn, scaled by (0.8 − top-2 margin) / 0.8, so confident passages keep their answer and uncertain ones take the full nudge.
* **It replaces the recipe's last step** (typography guards, then one-model-per-prompt assignment inside near-complete prompt clusters). The hidden test has only a few passages per prompt, so that assignment rarely fires there; the gated likelihood is a soft version of the guards.
* **Offline** — in our out-of-fold replica of the recipe's core (marker rates fitted on the other prompt-grouped folds) it adds +0.0030 over the same pipeline without the last step, and −0.0011 on top of the second ATLAS (both act on the uncertain passages), so the leaderboard has the last word.
"""
SK_SECTION = """## Softer balancing: Sinkhorn at {alpha}

The recipe balances last: Sinkhorn rescales the test probabilities, model by model, until every model gets the same total mass, since balanced accuracy weighs every model's recall equally. `--sinkhorn α` sets how far the final log-probabilities move towards the balanced ones, `(1 − α) · log P + α · log Q`. Q differs from P only by one factor per model and one per passage, so this adds **α times the per-model log-offset** that full balancing would add; the recipe uses α = 1.

* **Why less than full** — full balancing bets that every model has about the same number of hidden passages. The competition says the counts per model are unequal and that the public and private leaderboards use different model mixes. On an uneven mix full balancing over-corrects: a [forum analysis]({K}/competitions/name-that-llm/discussion/752365) of out-of-fold predictions found 50 iterations losing ~0.02 on a moderately uneven mix, while 1-3 iterations stayed within 0.001 of no balancing.
* **On the leaderboard** — PeterDreamHan's [Soft Sinkhorn 0.5]({soft_sk}) changed only this value on the 0.651 recipe and scored **0.653** (+0.002). This version uses **α = {alpha}** (`--sinkhorn {alpha}`).
* **Where it acts** — after the complementary stack and ATLAS are mixed in and before the typography step, so it touches a different stage from both.
"""
SPEED_TEXT = {   # intro of the "time" section: (heading, sentence before the list)
    "mn8": ("Paying for its time", "Six adapter passes over ~4,000 hidden passages cost an 8B member ~45-50 min, and the "
            "recipe's ATLAS budget ends at a fixed mark. {npay} changes that leave the predictions unchanged buy {back}:"),
    "a2": ("Paying for its time", "The second ATLAS needs ~35 min, and the recipe's ATLAS budget ends at a fixed mark. "
           "{npay} changes that leave the predictions unchanged buy ~1.5 h:"),
    "plain": ("A faster run, same predictions", "{npay} changes that leave the predictions unchanged reach ATLAS ~1.5 h "
              "earlier, so it always finishes inside its fixed budget:"),
    "m45": ("A faster run, same predictions", "{npay} changes that leave the predictions unchanged reach ATLAS ~1.5 h "
            "earlier, which pays for the extra passages it scores here:"),
}
DOC = """# {title}

**Public LB {score}** · {builds_on} · T4 ×2 · internet off · every input is public

This is Dhruv Tibarewal's [nlm-v27]({K}/code/dhruvtibarewal/nlm-v27) (public 0.651) as [avazaku reproduced it]({K}/code/avazaku/name-that-llm-0-651-verified-reproduction), with Llama-3.1-8B from his [public, hash-checked cache]({K}/code/avazaku/name-that-llm-verified-llama-asset-cache), plus {what}. {atlas}

The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Version | Change vs the 0.651 recipe | Public LB |
|---|---|---:|
{rows}

{sections}## {speed_head}

{speed_intro}

{pay}

## How the 0.651 recipe works

1. **Members** — Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2) and Llama-3.1-8B (1) LoRA24 members, each averaged over 5 fold adapters + a full fit; Nishant Dahal's Qwen3-14B (2.5) and Qwen2.5-14B (1) part-two checkpoints; DeBERTa-v3-large (1); a 4-view n-gram SVM on train + pool (1.5).
2. **Test prompt recovery** — capped average-linkage clustering on content-word TF-IDF; on recovered groups, prompt-centred "twin" heads and the shared-prompt Gaussian head add evidence.
3. **Blend** — weighted sum of log-probabilities, softmax, then the complementary stack at 15%.
4. **ATLAS** — a source-conditioned likelihood (Joseph Adamski's public bundle) mixed at 10% on passages whose top-2 margin is below 0.3.
5. **Balancing and rules** — Sinkhorn to equal per-model mass (balanced accuracy rewards recall on rare models), typography guards, one-model-per-prompt assignment in near-complete clusters.

## Things that did not transfer (so you don't have to try them)

| Change on top of a public recipe | Offline | Public LB |
|---|---:|---:|
| Dhruv's ModernBERT-large member (w 1.5) on the 0.651 recipe | +0.008 | 0.645 |
| nlm-v27 without its Llama member | | 0.643 (Llama is worth +0.008) |
| SVM trained with quality-tiered pool weights (three bases) | +0.028 member | +0.001 / -0.001 / -0.003 |
| A Mistral-7B member mixed into the 0.642 V28 stack at 10% | +0.028 bank | -0.001 |
| Qwen3.8-27B share 15% / 20% in V28 (10%) | | -0.001 / -0.002 |{dnt_rows}

## Running it

Attach the inputs below, choose **GPU T4 ×2**, Internet **off**, then *Save Version → Save & Run All* and submit that version. On the 3 visible rows the commit takes {commit}; Kaggle's rerun on the hidden rows takes several hours and runs everything.

## Credits and licences

* **Dhruv Tibarewal** — the 0.651 recipe, its LoRA24 members, pool and adapters ([nlm-v27]({K}/code/dhruvtibarewal/nlm-v27), [Fully Reproducible]({K}/code/dhruvtibarewal/name-that-llm-fully-reproducible)).
{v38_credit}* **avazaku** — the [verified reproduction]({K}/code/avazaku/name-that-llm-0-651-verified-reproduction) and the [Llama asset cache]({K}/code/avazaku/name-that-llm-verified-llama-asset-cache) this notebook loads Llama from.
* **Nishant Dahal** — the [two-stage LoRA recipe and part-two checkpoints]({K}/code/nishantdahal/two-stage-lora-part-two).
* **Lucas Boesen** — the LoRA24 recipe{mn8_recipe} ([trainer]({K}/code/lucasboesen/name-that-llm-train-lora24-s7)), GLM-4-9B and encoder features for the stack.
* **prvsiyan** — the [shared-prompt Gaussian and the complementary stack]({K}/code/prvsiyan/the-prompt-is-not-the-author); **Kirill Tymchyshyn** — the [combined ensemble]({K}/code/kirilltym/0-600-teach-the-model-remove-the-prompt); **Joseph Adamski** — [ATLAS]({K}/code/josephadamski91/teach-the-model-10-atlas) (CC BY-SA 4.0); **vnton37nguyn**, **Zbigniew Kirchner**, **pjmathematician** — stack members; **Nuri Dudak** and every author of the public generation datasets below — the pool.{a2_credit}
* **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).
* {mn8_licence}**GLM-4-9B-0414** (in the complementary stack, Lucas Boesen's truncated copy) — Zhipu AI / THUDM, MIT. Qwen models — Alibaba Qwen team, under their own licences. Each input keeps its own licence; the notebook's code is Apache 2.0 like its parent.

### Inputs

{inputs}
"""
SECTIONS = {  # heading for the code cell whose source starts with the key (the code cells themselves are unchanged)
    "# Name That LLM - full solution": "Inputs\nFinds every base model, adapter and checkpoint under `/kaggle/input`. "
    "Llama-3.1-8B comes from avazaku's hash-checked cache (shard sizes asserted) and is copied to local disk first{inputs_extra}. "
    "`FAST_COMMIT` is true only on the 3 visible test rows.",
    "%%writefile gauss.py": "Shared-prompt Gaussian head (`gauss.py`)\nA two-way model *x = author mean + prompt effect + "
    "noise*, integrated over the unknown prompt, with injective group marginals (prvsiyan / Kirill Tymchyshyn / sangam21, "
    "Apache 2.0).",
    "%%writefile infer.py": "Main pipeline (`infer.py`)\nDhruv Tibarewal's pipeline: pooled hidden states of each LoRA24 "
    "member, plain and prompt-centred heads, the 4-view n-gram SVM, test prompt recovery, the weighted log-probability "
    "blend, the complementary stack, ATLAS, Sinkhorn balancing, typography guards and dense-group assignment.{infer_extra}",
    "%%writefile comp_stack.py": "Complementary stack (`comp_stack.py`)\nThe public frozen-feature ensemble (prvsiyan, Lucas "
    "Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%.",
    "%%writefile atlas2_worker.py": "Second ATLAS worker (`atlas2_worker.py`)\nOne process per GPU: loads the Qwen3-4B "
    "bundle with its own `atlas_core.py`, scores its half of the passages shortest first, saves after every 16 and stops "
    "at ATLAS's deadline.",
    "# complementary public stack in its own process": "Run the complementary stack\nIn its own process, so all GPU memory "
    "is free again afterwards; its wall time is charged to ATLAS's budget.",
    "W_N14, W_N8 = 1.0, 0.5": "Members and run\n{run_text}",
}


V31 = "{K}/code/mercurydust/name-that-llm-submission"
V31_DOC = """# {title}

**Public LB {score}** · builds on mercurydust's 0.660, which builds on Roman Tamrazov's 0.654 and the 0.651 recipe · T4 ×2 · internet off · every input is public

This is mercurydust's [Name That LLM-Submission]({v31}) (public **0.660**), version 2, the code that scored. That notebook is Roman Tamrazov's [Name That LLM | Who am I?]({who}) (0.654) with a set of changes its author calls V31, and Roman's is Dhruv Tibarewal's [nlm-v27]({K}/code/dhruvtibarewal/nlm-v27) (0.651) as [avazaku reproduced it]({K}/code/avazaku/name-that-llm-0-651-verified-reproduction), with Llama-3.1-8B from his [public, hash-checked cache]({K}/code/avazaku/name-that-llm-verified-llama-asset-cache). On top of V31 this version adds {what}.

The code cells below are byte for byte the version that scored {score}.

## Leaderboard A/B

| Version | Change | Public LB |
|---|---|---:|
{rows}

## What V31 changed (mercurydust)

Our reading of the code, against Roman's 0.654 notebook:

* **External rows used** — the matched-clean external passages are de-duplicated against the official rows (exact text after Unicode and whitespace normalisation) and used: the 4-view n-gram SVM now trains on official + external (weight 0.5) + pool (weight 0.35) rows, and the TF-IDF (weight 0.25) and text-distortion (0.10) members, both at 0 before, come back.
* **Family gates applied** — the complementary stack fitted within-family gate models (Claude style, Claude LoRA, GPT, Gemini, GLM) whose output never reached its logits. They now rerank the models inside each family where the family's own top two are close (margin below 0.45 within the family), keeping the family's total probability.
* **Softer balancing** — Sinkhorn at 0.45 towards a mix of 0.65 × the predicted and 0.35 × the training model proportions, instead of 1.0 towards equal proportions. On the 0.651 recipe, PeterDreamHan's [Soft Sinkhorn 0.5]({soft_sk}) had shown that softer balancing pays (0.653).
* **Typography step reworked** — 37 markers (Roman's 34 + `###`, a code fence, `> `), each model's marker rates shrunk towards the overall rate, markers weighted by how much their rates differ between models, evidence clipped to ±0.30, and a gate that multiplies Roman's margin gate (here (0.6 − margin) / 0.6) by a passage-length gate and a marker-count gate.
* `--post 0`, and optional token-attention heads are skipped when their backbone is not attached (the Qwen3.5 one is not, so V31's `NTL_DROP_Q35 = '0'` changes nothing here).

{sections}## A faster run

Two changes that leave the predictions unchanged reach ATLAS over an hour earlier, and ATLAS uses that time:

{pay}

## How the pipeline works

1. **Members** — Qwen3-14B-Base (weight 3), Qwen3-8B-Base (2) and Llama-3.1-8B (1) LoRA24 members, each averaged over 5 fold adapters + a full fit; Nishant Dahal's Qwen3-14B (2.5) and Qwen2.5-14B (1) part-two checkpoints; DeBERTa-v3-large (1); the 4-view n-gram SVM (1.5); TF-IDF (0.25) and text-distortion (0.10) members.
2. **Test prompt recovery** — capped average-linkage clustering on content-word TF-IDF; on recovered groups, prompt-centred "twin" heads and the shared-prompt Gaussian head add evidence.
3. **Blend** — weighted sum of log-probabilities, softmax, then the complementary stack (with its family gates) at 15%.
4. **ATLAS** — a source-conditioned likelihood (Joseph Adamski's public bundle) mixed at {atlas_pct}% on passages whose top-2 margin is below 0.3.
5. **Balancing** — V31's soft Sinkhorn (above).
6. **Typography step** — V31's, which writes the submission.

## Things that did not transfer (so you don't have to try them)

| Change on top of a public recipe | Offline | Public LB |
|---|---:|---:|
| Dhruv's ModernBERT-large member (w 1.5) on the 0.651 recipe | +0.008 | 0.645 |
| nlm-v27 without its Llama member | | 0.643 (Llama is worth +0.008) |
| SVM trained with quality-tiered pool weights (three bases) | +0.028 member | +0.001 / -0.001 / -0.003 |{dnt_rows}

## Running it

Attach the inputs below, choose **GPU T4 ×2**, Internet **off**, then *Save Version → Save & Run All* and submit that version. On the 3 visible rows the commit takes ~5 min: it resolves every input and writes a placeholder; Kaggle's rerun on the hidden rows takes several hours and runs everything.

## Credits and licences

* **mercurydust** — the V31 changes and the [0.660 notebook]({v31}) this one forks.
* **Roman Tamrazov** — the [0.654 notebook]({who}) V31 starts from, and the typography step it reworks.
* **PeterDreamHan** — [Soft Sinkhorn 0.5]({soft_sk}), the first public evidence that softer balancing pays.
* **Dhruv Tibarewal** — the 0.651 recipe, its LoRA24 members, pool, matched-clean passages and adapters ([nlm-v27]({K}/code/dhruvtibarewal/nlm-v27), [Fully Reproducible]({K}/code/dhruvtibarewal/name-that-llm-fully-reproducible)).
* **avazaku** — the [verified reproduction]({K}/code/avazaku/name-that-llm-0-651-verified-reproduction) and the [Llama asset cache]({K}/code/avazaku/name-that-llm-verified-llama-asset-cache) this notebook loads Llama from.
* **Nishant Dahal** — the [two-stage LoRA recipe and part-two checkpoints]({K}/code/nishantdahal/two-stage-lora-part-two).
* **Lucas Boesen** — the LoRA24 recipe ([trainer]({K}/code/lucasboesen/name-that-llm-train-lora24-s7)), GLM-4-9B and encoder features and token-attention heads for the stack.
* **prvsiyan** — the [shared-prompt Gaussian and the complementary stack]({K}/code/prvsiyan/the-prompt-is-not-the-author); **Kirill Tymchyshyn** — the [combined ensemble]({K}/code/kirilltym/0-600-teach-the-model-remove-the-prompt); **fleongg** — [NTL Part Two Complement]({K}/code/fleongg/ntl-part-two-complement); **Joseph Adamski** — [ATLAS]({K}/code/josephadamski91/teach-the-model-10-atlas) (CC BY-SA 4.0; method: [Ko et al.](https://arxiv.org/abs/2610.19583)); **vnton37nguyn**, **Zbigniew Kirchner**, **pjmathematician** — stack members; the text-distortion member follows Stamatatos (EACL 2017); **Nuri Dudak** and every author of the public generation datasets below — the pool and external passages.
* **Built with Llama.** Llama 3.1 is licensed under the [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/) (Copyright © Meta Platforms, Inc.) and its [Acceptable Use Policy](https://www.llama.com/llama3_1/use-policy/).
* **GLM-4-9B-0414** (in the complementary stack, Lucas Boesen's truncated copy) — Zhipu AI / THUDM, MIT. Qwen models — Alibaba Qwen team, under their own licences. Each input keeps its own licence; the notebook's code is Apache 2.0 like its parents.

### Inputs

{inputs}
"""
MG_SECTION = """## The typography step with its margin gate only

V31's typography step multiplies its margin gate by two more: a length gate (√(words / 80), clipped to [0.4, 1]) and a marker-count gate (√((markers present + 1) / 4), clipped to [0.45, 1]). They shrink the nudge on short passages and on passages with few markers. This version keeps the margin gate alone (one changed line in `infer.py`).

* **Offline** — in our out-of-fold replica of the recipe's core (5 prompt-grouped folds, marker rates fitted on the other folds, V31's balancing): no typography step 0.6472, Roman's step 0.6476, V31's step 0.6480, and V31's evidence with the margin gate alone **0.6498** (+0.0019 over V31's step; 95% prompt bootstrap −0.0004 to +0.0044). It is also ahead or level on test pools with uneven model mixes.
* Out-of-fold gains have mis-ranked additions before (see the last table), so the leaderboard A/B above has the last word.
"""
PAY_DEADLINE_V31 = ("* **ATLAS runs to its deadline.** V31 stops ATLAS at its deadline and keeps the passages it finished, but "
                    "also stops it at the first checkpoint where a projection of its finish passes the budget, which can "
                    "leave much of the time unused. That stop is removed: ATLAS keeps scoring, shortest passages first, "
                    "until the deadline itself.")
V31_SECTIONS = dict(SECTIONS, **{
    "%%writefile infer.py": "Main pipeline (`infer.py`)\nDhruv Tibarewal's pipeline with Roman Tamrazov's and "
    "mercurydust's changes: pooled hidden states of each LoRA24 member, plain and prompt-centred heads, the n-gram SVM "
    "(on official + external + pool rows), the TF-IDF and text-distortion members, test prompt recovery, the weighted "
    "log-probability blend, the complementary stack, ATLAS, V31's soft Sinkhorn and its typography step, which writes "
    "the submission.{infer_extra}",
    "%%writefile comp_stack.py": "Complementary stack (`comp_stack.py`)\nThe public frozen-feature ensemble (prvsiyan, Lucas "
    "Boesen, vnton37nguyn, Kirill Tymchyshyn), mixed in at 15%, with V31's within-family gates. Optional token-attention "
    "heads are skipped when their backbone is not attached.",
})


LLAMA_GONE = [   # (text, replacement) in the V31 document once avazaku's Llama cache is gone
    ("every input is public", "every input was public when it scored"),
    ("with Llama-3.1-8B from his [public, hash-checked cache]({K}/code/avazaku/name-that-llm-verified-llama-asset-cache).".format(K=K),
     "with Llama-3.1-8B from his hash-checked cache (no longer available, see *Running it*)."),
    ("this notebook loads Llama from.", "this notebook loaded Llama from (no longer available)."),
    ("Kaggle's rerun on the hidden rows takes several hours and runs everything.\n",
     "Kaggle's rerun on the hidden rows takes several hours and runs everything.\n\n"
     "**Llama-3.1-8B:** avazaku's Llama asset cache, which this version loaded Llama from, is no longer available "
     "(checked 2026-10-29). alexchilton's public model [Llama 3.1 8B base]({K}/models/alexchilton/llama-3-1-8b-base) "
     "holds the same files: on Kaggle, the sha256 of all four weight shards and the hashes of the config, tokenizer "
     "and index files match the copy this version loaded (checked 2026-10-29). To rerun with it, attach it "
     "(Transformers, `default`, version 1), and in cell 1 replace `BL = mdir('/llama-3.1/', '/8b/')` with "
     "`BL = mdir('/llama-3-1-8b-base/')` and remove `'name-that-llm-verified-llama-asset-cache' in BL and` from the "
     "assert. Meta's own model (`metaresearch/llama-3.1`, Transformers, `8b`, version 2, after accepting Meta's "
     "licence, as Dhruv's nlm-v27 does) needs only the assert change. The shard-size check stays, so the weights "
     "must be the same files.\n".format(K=K)),
]


LV08 = "{K}/code/lightningv08/0-661-qwen3-8-27b-lora-spectral-ensemble"
LV08_DS = "{K}/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets"
AC_MODEL = "{K}/models/alexchilton/llama-3-1-8b-base"
GSQ_SECTION = """## lightningv08's Qwen3.8-27B readout at {gpct}%

lightningv08's public [0.661 notebook]({lv08}) is mercurydust's 0.660 recipe (rebuilt from cached fits) plus one new member: his Qwen3.8-27B "GSQ3" LoRA readout, a 3-bit packed copy of Qwen3.8-27B with a LoRA on its last eight decoder layers and a linear head on the mean-pooled states of eight depths, trained on ~120k public generated passages and then on the official rows. It reads every token of a passage (batch size 1, one worker per T4). He mixes it at 10% after ATLAS and before the balancing: **+0.001** over 0.660, with ATLAS at 10%.

Our ATLAS change was measured on the same 0.660 base, so this version stacks the two. His five `gsq/` files run unchanged, in their own processes with his pinned wheels (transformers 5.14.1, installed into a separate directory), right after the complementary stack. Their wall time comes out of ATLAS's budget together with the stack's: on Kaggle's T4 ×2 we timed ~1.45 s per passage per GPU plus ~3 min to copy the 11.85 GB base to local disk, i.e. under an hour on the hidden rows, and ATLAS finished well inside its budget in our 0.663 runs. The readout is mixed at **{gpct}%** (`--gsq_w {gflag}`), after ATLAS and before the balancing, as in his notebook. If the stage fails or passes its 90-minute limit, the run continues exactly as it would without it.

His asset collection also holds wheels, configs and checkpoints that this pipeline's input searches would pick up: a bitsandbytes 0.50.2 wheel, Llama and Qwen bases in other layouts, and a Qwen3.5-architecture config that would switch on the optional Qwen3.5 token-attention head. So `glob.glob` hides that collection in the notebook and in `comp_stack.py`, and the GSQ stage reaches it by explicit paths.
"""
M45_NOTE = ("\nATLAS also scores the passages whose top-2 margin is below **0.45** instead of 0.3 (`--atlas_margin 0.45`), "
            "so at {pct}% its reach is no longer capped by the 0.3 cut. That costs time: ATLAS scores more passages.\n")
SK_SECTION = """## Balancing at α = {alpha}

V31's soft Sinkhorn first balances the ensemble's probabilities towards target model proportions (0.65 × the predicted ones + 0.35 × the training ones), then keeps a share α of the balanced log-probabilities: (1 − α) log P + α log Q. V31 uses α = 0.45; this version uses **α = {alpha}** (`--sinkhorn {alpha}`), i.e. {direction} balancing. On the 0.651 recipe, PeterDreamHan's [Soft Sinkhorn 0.5]({soft_sk}) showed that this knob moves the score (1.0 → 0.5: +0.002).{k_note}
"""
COMP_SECTION = """## Complementary stack at {cpct}%

The complementary public stack enters the first probability mixture at 15% (`--comp_w 0.15`): P = 0.85 × the core ensemble + 0.15 × the stack. It is trained differently from the core (stylegram and LoRA24 heads, family gates), the way ATLAS is, and ATLAS's share was the lever that moved this recipe. This version mixes the stack at **{cpct}%** (`--comp_w {cflag}`).{k_note}
"""
OPEN_POOL_DS = "{K}/datasets/crtproduction/name-that-llm-open-pool"
ZK_0664 = "{K}/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public"
JU_0664 = "{K}/code/judith007/0-664-hyperparameter-tuned-who-are-u"
ATLAS_FULL = "{K}/code/omerfarukmerey/ntl-atlas-full-ext"
OSVC_SECTION = """## The SVM learns from the open pool

The 4-view n-gram SVM (character 3-5, word 1-3, word-shape 2-6 and function-word 1-3 n-grams, one LinearSVC) trains on the official rows (weight 1), the matched-clean external passages (0.5) and a pool (0.35). V31's pool is Dhruv Tibarewal's `nlm_pool_v5` (~120k generated passages). Following zbigniewkirchner's [0.664 notebook]({zk}), this version uses crtproduction's [Name That LLM Open Pool]({pool}) instead: ~198k public real-model passages (CC BY-SA 4.0), minus any passage regenerated from a training prompt (the flag is in the file; none are, and none equals an official passage). The setup cell writes that pool to `/kaggle/working/open_pool.parquet` and passes it as `--pool`.

Its weight in the blend is **{w}** (`--svc_w {w}`; V31: 1.5). The SVM's member is the log-softmax of its raw decision values, which are flat: out of fold its top-2 gap averages 0.38 nat, against 3.0 for the Qwen3-14B member's log-probabilities, so at 1.5 (against that member's 3) it has little say; at 12 it weighs about as much as that member. zbigniewkirchner's own public ladder on this pool: svc_w 1.5 → 0.664, 6 → 0.659, 12 → 0.664, 20 → 0.664.{k_note}
"""
AX_SECTION = """## ATLAS-8B retrained with open-pool passages, and a sharper mix

* **Retraining** — Joseph Adamski's bundle trains ATLAS on the official rows only (150 passages per model). We retrained it from scratch with his code (`ntl_atlas`, `ntl_rlcd`, Apache 2.0) and his training configuration (Qwen3-8B-Base, rank 16, 5 epochs, 128-row steps, AdamW 1e-4, 5% warm-up then cosine, clip 1.0), on every official row plus 500 passages per model from the [open pool]({pool}) (each epoch sees all official rows and a fifth of the extra ones): kernel [ntl-atlas-full-ext]({full}), 180 steps, ~3.1 h on T4 ×2. This notebook loads those adapters in place of his (his bundle stays attached, for its code layout). The bundle's `atlas_bundle.json` copies his `training_config` verbatim, so its `data_policy` line ("official training file only … no external passages") describes his bundle, not this one; its `exp029` block records what this one saw (2,699 official rows + 9,000 open-pool passages).
* **Out of fold** — the same training with prompt fold 0 (30 prompts, 540 passages) held out: ATLAS alone **0.406** balanced accuracy against **0.341** without the extra passages (paired prompt bootstrap +0.065 [+0.037, +0.093]).
* **A sharper mix** — ATLAS's scores are nearly flat (mean top score 0.067 of 18, top-2 margin 0.002), so V31's linear mix, (1 − w)·P + w·A on the passages whose top-2 margin is below 0.3, moves almost no pick by itself: it mostly shrinks those rows towards uniform before the balancing. Here, after that mix, the same rows are multiplied by **A^{k}** and renormalised (`--atlas_pow {k}`), a log-linear mix that lets ATLAS's evidence move picks.
* **Offline** — in our out-of-fold replica of the recipe's core (its three fold-adapter LoRA members, the refitted SVM and a DeBERTa stand-in, V31's balancing), with prompt folds 0 and 1 held out in turn (one retraining each, 1,079 passages): the 20% linear mix 0.6328, × ATLAS^4 **0.6504** (+0.018 [+0.003, +0.032], paired prompt bootstrap), and every k from 2 to 8 gains on both folds. On fold 0 the same power mix with the official-only retraining loses 0.004-0.024, and a retraining with 1,000 passages per model gains nothing this way, so the leaderboard A/B above has the last word.
"""
TYPO_SECTION = """## Typography evidence at ×{scale}

V31's typography step scales its marker evidence by 1.5 / (sum of the marker weights) and clips it to ±0.30 before the gates. judith007's [0.664 notebook]({ju}) scales it by 1.2 (with Sinkhorn 0.60; our Sinkhorn 0.60 without it scored 0.658); {this} (one changed line). Out of fold the scale moves balanced accuracy by under 0.001 either way (1.0, 1.2 and 1.5), on even and on uneven model mixes.
"""
TYPO_THIS = {"1.2": "this version does too", "1.0": "this version goes a step further, to 1.0"}
PIN_SECTION = """## Extra passages pinned to their 2026-10-28 versions

The complementary stack fits its stylegram SVC, that SVC's centred twin and the GPT / Gemini / GLM family gates on public extra-passage datasets, and Kaggle attaches each dataset's newest version whenever a notebook is saved. Three of them appended batches after our 0.663 run (abdullah1afify/ntl-generations, lucasboesen/ntl-free-proxy-generations and prvsiyan's extended dataset: +274 rows after the stack's filters), and two public reruns of the 0.663 code on the newer files scored 0.662. This version's loader keeps only the batches each file had on 2026-10-28 (a whitelist of batch tags), which gives back exactly the 10,325 rows of that run, so later uploads cannot move it either.
"""
P30_SECTION = """## Extra passages pinned to their 2026-10-30 batches

The complementary stack fits its stylegram SVC, that SVC's centred twin and the GPT / Gemini / GLM family gates on public extra-passage datasets, and Kaggle attaches each dataset's newest version whenever a notebook is saved. Our earlier versions kept the batches each file had on 2026-10-28: 10,325 rows after the stack's filters (prompts answered by at least 12 models, 30-300 words, one passage per prompt and model). judith007's [0.664 notebook]({ju}) read the files as they were on 2026-10-31, which is 418 rows more (10,743): batches d1029-d1030 of abdullah1afify/ntl-generations, v2b19-v2b20 of lucasboesen/ntl-free-proxy-generations and version 5 of prvsiyan's extended dataset. The batches added since add no rows (their prompts do not yet have 12 models' answers across the files), and abdullah1afify's two new files (plain Claude and paid fal.ai passages) match none of the loader's file names. This version keeps the 2026-10-30 batches (a whitelist of batch tags), i.e. exactly those 10,743 rows, so later uploads cannot move it.{p_note}
"""
AR_0666 = "{K}/code/areebbb/ntl-we-back-0-666"
AR_J_AJ = "{K}/code/ayodejiibrahimlateef/ntllm-areeb-j-open-pool-control-20261102"
AR_J_RK = "{K}/code/rahulkrttty/ntl-v14-apex-0-666"
SOV_NB = "{K}/code/synthreaper/ntl-sovereign-gold-apex"
ZK_0667 = "{K}/code/zbigniewkirchner/0-667-pairwise-verifier-matched-external-rows"
AC_0666 = "{K}/code/alexchilton/name-that-llm-0-666-all-public-what-failed"
KT_SIB = "{K}/code/kirilltym/lora-trained27-gentle-sibling-correction"
LV08_0666 = "{K}/code/lightningv08/0-666-qwen3-8-27b-lora-weighted-ensemble"
AB_DS = "{K}/datasets/mercurydust/ntllm-chatgpt-a-b"
AC_ALL = "{K}/code/alexchilton/name-that-llm-0-663-all-public-what-failed"
AF_0667 = "{K}/code/mohammedafreedpasha/name-that-llm-solution"
MH_0667 = "{K}/code/matterhorn3838/name-that-llm-v1"
AC_0667 = "{K}/code/alexchilton/name-that-llm-0-667-all-public-what-failed"
PP13_REF = "omerfarukmerey/ntl-open-pool-v13-pin"
V35_SECTION = """## The pairwise verifier with V35's settings

mohammedafreedpasha's [Name That LLM Solution]({af}) (public 0.667) is the 0.667 notebook with one change, "V35" of mercurydust's pairwise verifier. The verifier scores a test passage against each model by pairing it with verified reference passages of that model (char and word similarity, the A/B evidence-quote profiles, task and structure agreement, length and punctuation); V35 changes five numbers in it:

| setting | V34.1 | V35 |
|---|---|---|
| reference passages per model | 6 | **8** |
| a model's score: mean of its best pair probabilities | top 3 | **top 4** |
| passages an evidence-quote key needs to become a prototype (all rows / inside a training fold) | 8 / 6 | **6 / 5** |
| punctuation marks compared | 11 | **12** (adds `-`) |
| weight of the fused evidence on passages with top-2 margin below 0.35 | 0.10 | **0.12** |

This version applies exactly that change to the 0.667 code (the diff is the five lines above, the version tag and the log label).{v35_note}
"""
PP13_SECTION = """## The open pool at its Kaggle version 13

crtproduction published release v12 of the [open pool]({pool}) on 2026-11-03 (Kaggle version 14: +20,781 passages, 211,369 in all), and a saved notebook always attaches each dataset's newest version. The 0.667 code scored 0.667 on version 13 ([zbigniewkirchner]({zk7}), [matterhorn3838]({mh})) and 0.666 in two runs on version 14 (mohammedafreedpasha's previous version, alexchilton's [latest version]({ac7})), a gap inside the run-to-run noise but in the wrong direction. So this version reads version 13 from an unmodified copy, [ntl-open-pool-v13-pin]({pin}) (the same `ntl_open_pool.parquet`, sha256 `38e48b40…`, CC BY-SA 4.0 like the original): the SVM's pool and the matched rows are exactly those of the 0.667 runs (the commit log prints `svc pool = open pool 190588`).{pp_note}
"""
XM_SECTION = """## Matched open-pool rows behind the n-gram members ({xm}×)

Three members learn from external passages besides the official rows: the TF-IDF logistic regression (weight 0.25), the text-distortion member (0.10) and the 4-view SVM. V31 passes them Dhruv Tibarewal's `nlm_matched_clean` (24,443 passages after de-duplication) at sample weight 0.5 (`--extra`, `--extra_w 0.5`). areebbb's [0.666 notebook]({ar}) (a controlled ladder on alexchilton's [all-public fork]({ac}) of our 0.663) replaces them with passages drawn from the [open pool]({pool}) by its `match_weight` column, which crtproduction fitted so that the drawn passages match each model's typography in the official rows (dashes, quotes and the like): the same count per model as `nlm_matched_clean` in their version J, the one that scored 0.666 (two public reruns of it, [ayodeji's]({aj}) and [rahulkrttty's]({rk}), scored 0.666 too), and twice that count (48,886 passages) in their next version K. This version draws **{xm}×** ({n_rows} passages; a model with fewer pool passages gives all it has) with the same weights and seed: the setup cell writes them to `/kaggle/working/extra_matched.parquet` and passes that file as `--extra`. In the SVM the drawn passages count at 0.5 and the rest of the open pool at 0.35 (each exact duplicate is kept once). The draw is pandas' `sample(n, weights='match_weight', random_state=0)` written out in numpy (newer pandas refuses so heavily weighted a draw without replacement), so it gives the same rows under any pandas version.{x_note}
"""
XM_ROWS = {1: "24,443", 2: "48,886", 3: "73,226", 4: "94,688"}   # after infer.py's exact de-duplication (exp034)
TF_SECTION = """## The TF-IDF member at {tfw}

With the matched rows the TF-IDF logistic regression gets much better on its own: out of fold (5 prompt-grouped folds of train, the notebook's own code) it goes from 0.418 balanced accuracy with `nlm_matched_clean` to 0.464 with the 2× draw and 0.471 with 4×. V31 still weighs it 0.25, against 3 / 2 / 1 for the three LoRA24 members and 12 for the SVM. In our out-of-fold replica of the core (the three fold-adapter LoRA members, the open-pool SVM at 12, a DeBERTa stand-in, Sinkhorn 0.30), with the TF-IDF, distortion and SVM members all refitted on the same external rows, K itself scores +0.0026 over `nlm_matched_clean` (on the board J, the 1× draw, gained +0.002), and raising the TF-IDF weight on top of K adds 0.5 **+0.0044** (four of the five folds up), 0.75 +0.0030, 1.0 +0.0041, 1.5 +0.0011; with 4× rows every weight scores below 2×. Before the matched rows, 0.5 had lost on the board (areebbb's ladder, version I: −0.002), and the replica agreed (−0.0007), so the leaderboard has the last word. This version uses **{tfw}** (`--tfidf_w {tfw}`).{t_note}
"""
KILUPY_DS = "https://www.kaggle.com/datasets/kilupy/ntl-18-model-corpus"
KX_TF = {True: " In the replica the TF-IDF member then wants more weight: 0.6554 at 1.0, **0.6568 at 1.5** and 0.6587 at 2.0, "
               "against 0.6542 for K's rows with the TF-IDF member at 0.5; without kilupy's rows, 1.5 loses (−0.0033). Most of "
               "that edge comes from one of the five folds, so this is the least certain of the changes here.",
         False: " Their only earlier board test came with the TF-IDF member at 1.5 on K's rows (0.655), and a TF-IDF weight "
                "above 0.25 lost on its own (0.663 at 0.5 and at 0.75, against 0.666), so here the TF-IDF member stays at 0.25."}
KX_ROWS = {1: "44,946 external rows", 2: "69,389 external rows after de-duplication"}
SEY_DS = "https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages"
MW_SECTION = """## The stack's open-pool rows weighted by typography match

The open pool's `match_weight` column (crtproduction's fit of each passage's typography to its model's official rows) already decides which pool passages the n-gram members' external rows are drawn from. The stack's open-pool prompts (previous section) enter without it. This version gives each of those rows its `match_weight` (normalised to mean 1 per model, clipped to [0.25, 4]) in the stylegram SVC, its centred twin and the GPT / Gemini / GLM gates; a model's extra rows keep their total share, and any failure falls back to equal weights. Out of fold, a TF-IDF member trained on open-pool rows alone (71,541 of them) scores **+0.017** balanced accuracy on the official rows with these weights (0.4290 → 0.4460, 95% prompt bootstrap +0.008 to +0.027).{mw_note}\n"""
KC_SECTION = """## kilupy's Claude passages in the stack's Claude gate

The stack re-ranks the five Claude models with a gate: a style SVM on the official Claude rows plus seyitkaangunes's [Claude passages from the Anthropic API]({sey}). kilupy's [18-model corpus]({kil}) (CC BY 4.0) holds 16,603 more Anthropic-API Claude passages, 3,338 prompts each answered by all five; this version adds them to that gate at the same weight as the existing API passages. Out of fold (five prompt-grouped folds of train), a Claude-only classifier trained on external rows alone separates the five official Claude models at 0.662 balanced accuracy with them against 0.642 without.{kc_note}\n"""
SIB_SECTION = """## A conservative sibling-style residual (V40A at 0.08)

After the typography step, Roman Tamrazov's sibling-style residual (V40A, version 33 of his [Who am I?]({who}) notebook, brought into this line by synthreaper's [Sovereign Gold Apex]({sov})) re-ranks the models inside each family (Claude, Gemini, GPT, GLM). Per family, a logistic regression on 144 style features (16 length and case statistics, 52 marker rates, 76 function-word rates) is fitted on the official rows; its prompt-grouped out-of-fold accuracy sets how far it may move a passage, and it moves only passages whose family top two are close (gate (0.65 − margin) / 0.65), keeping each family's total probability. Sovereign uses max_alpha 0.10; Roman's own version used 0.08, as do Kirill Tymchishin's [gentle sibling correction]({kt}), lightningv08's [0.666 notebook]({lv}) and this version. It moves few passages: out of fold on our replica of V31 it changed 0.7% of them at 0.08 (+0.0000), and on our replica of the core with K's rows it scored +0.0008 at 0.10, both inside the noise; on the board Roman's version scored 0.660, the same as V31 without it. Here it is one more small, independent nudge on close calls.{s_note}
"""
SOV_SECTION = """## Sovereign Gold Apex's post-hoc steps

synthreaper's [Sovereign Gold Apex]({sov}) (public 0.665; alexchilton's [verbatim run]({ac6}) scored 0.666) is the other public 0.666 line: One Swap (open-pool SVM at 12, Sinkhorn 0.30), then after the typography step (evidence ×1.35 instead of ×1.5) Roman Tamrazov's sibling-style residual (V40A, max_alpha 0.10) and mercurydust's V34.1 pairwise verifier: a same-model pair classifier trained on the official rows with mercurydust's [ChatGPT A/B annotations]({ab}) (revision 03), which compares each test passage with six verified references per model and nudges the passages whose top-2 margin is below 0.35 (weight 0.10); ATLAS's margin is 0.32. The matched open-pool rows are not in it. This version adds its three cells and its block after the typography step verbatim. On the board Sovereign's steps took One Swap's 0.664 to 0.665-0.666; out of fold on our replica of K's core they scored −0.0007 (V40A +0.0008; the verifier alone scores 0.044 on unseen prompts, chance 0.056), so this is the board's call.{v_note}
"""
G37_SECTION = """## Gemini 3.7 Flash versus 3.8 Flash

The largest confusion in our out-of-fold replica of the core is gemini-3.7-flash called gemini-3.8-flash: of 150 real 3.7 passages 48 are called 3.7 and 73 are called 3.8, and the blend predicts 3.7 81 times against 199 for 3.8 (150 of each). The three LoRA members are not the cause (on real 3.7 passages they favour 3.7 by 0.37-0.49 in log-probability, on real 3.8 passages 3.8 by about as much); the n-gram members favour 3.8 on both (the SVM by 0.43 and 0.64), and the SVM weighs 12. Their data explains it: the open pool keeps only the 3,278 gemini-3.7 passages with a documented Vertex AI route (crtproduction found AI Studio's \"3.7\" passages indistinguishable from 3.8) against 21,550 for 3.8. This version adds **+0.3 to the 3.7 log-score after the balancing step** (one line in `infer.py`). In the replica that is +0.0015 (3.7 recall 0.32 → 0.47, 3.8 0.72 → 0.61), and a shift chosen on the other folds was +0.45 on four of the five folds (+0.0015, 95% prompt bootstrap −0.0037 to +0.0067).{g_note}
"""
KX_SECTION = """## kilupy's 18-model corpus added to those rows

kilupy's [18-model prompt-paired corpus]({kil}) (published 2026-11-02, CC BY 4.0) holds 16,603 Claude passages from the Anthropic API (3,338 prompts, each answered by all five Claude candidates) and 300 passages from each of the other 13 models via OpenRouter, about 100 words each; none of them is in the open pool or the official data. This version appends all 20,503 to the matched rows ({kx_rows}), so they reach the TF-IDF, distortion and SVM members at the same weight (0.5). Out of fold, with every member refitted (five prompt-grouped folds of train, the notebook's own code): a Claude-only classifier trained on external rows alone separates the five official Claude models better with them (balanced accuracy 0.642 → 0.662), and the SVM alone improves (0.511 → 0.516).{kx_tf}{kx_note}
"""
CX_SECTION = """## The stack's extra passages, with the open pool's multi-model prompts

The complementary stack's n-gram parts (its stylegram SVC, that SVC's prompt-centred twin and the GPT / Gemini / GLM family gates) learn from the official rows plus public extra passages: one passage per (prompt, model), 30-300 words, from prompts that at least 12 of the 18 models answered, so the centred twin can subtract each prompt's mean. The open pool records the prompt behind each of its passages (`prompt_group`), and 2,560 of its prompts have 12 or more models' answers. This version's loader adds them as one more source (prompt ids prefixed `op_`, the loader's own filters and de-duplication applied after the existing files): **43,071 passages from 2,956 prompts instead of 10,743 from 738**. A class's extra passages still share half of its weight in the stylegram SVC, and count 0.25 each in the gates.{c_note}
"""
TPU_KERNELS = {"op8": "omerfarukmerey/ntl-tpu-op8", "ll8o": "omerfarukmerey/ntl-tpu-ll8q14", "q14o": "omerfarukmerey/ntl-tpu-ll8q14"}
TPU_DESC = {"op8": ("Qwen3-8B-Base", "op8v1", "Qwen3-8B"), "ll8o": ("Llama-3.1-8B", "ll8op1", "Llama-3.1-8B"),
            "q14o": ("Qwen3-14B-Base", "q14op1", "Qwen3-14B")}
TPU_SECTION = """## A new member: {desc}, trained on Kaggle's free TPU

The recipe's LoRA members are its strongest parts, and all of them were trained on T4s on a 120k-passage pool. This version adds a sixth one that we trained on Kaggle's free **TPU v5e-8** (20 h a week, separate from the GPU quota), following GoPain's [measurements]({gp}) (a Qwen3-8B LoRA classifier ~14× faster than on T4 ×2, bf16 instead of fp16): kernel [{kernel}]({tk}). It is the same model as the recipe's LoRA24 members (Lucas Boesen's LoRA24 recipe with Dhruv Tibarewal's two stages): {desc} cut to its first {keep} layers and frozen, LoRA rank 16 on every projection, the mean of 8 depths with one LayerNorm each into a plain and a prompt-centred head, trained on same-prompt triplets. What is new is the data and the hardware: stage 1 reads crtproduction's [open pool]({pool}) ({pool_rows} passages; no train-prompt rows) with class weights, then stage 2 fine-tunes on the 2,699 official rows once per held-out prompt fold and once on all of them, so this notebook reads it like the 8B members: five fold adapters and the full one, averaged, at weight **{w}**, with its prompt-centred twin. The trainer is a torch_xla SPMD port of the GPU recipe (one process, the batch split over the 8 chips, fixed batch shapes so each compiles once, XLA's own gradient checkpointing); a GPU reload check through this notebook's own merge path agreed with the TPU's predictions.{tpu_note}
"""
PAY_COPY_AC = ("* **Llama-3.1-8B is copied to local disk before loading.** From avazaku's notebook-output mount, "
               "`from_pretrained` took ~2,050 s (CPU probe) and ~2,140 s (our GPU run), against 88-135 s for one sequential "
               "copy and ~60 s to load from local disk; from alexchilton's model the copy takes 203-274 s. Same bytes, same "
               "predictions.")
V31AC_SECTIONS = dict(V31_SECTIONS, **{
    "# Name That LLM - full solution": "Inputs\nFinds every base model, adapter and checkpoint under `/kaggle/input`. "
    "Llama-3.1-8B comes from alexchilton's public copy (shard sizes asserted) and is copied to local disk first{inputs_extra}. "
    "`FAST_COMMIT` is true only on the 3 visible test rows.",
    "# fork (exp027): lightningv08's Qwen3.8-27B GSQ3 readout": "GSQ stage (`run_gsq`)\nInstalls lightningv08's pinned "
    "wheels into their own directory, copies the 11.85 GB GSQ3 base to local disk, runs his two isolated T4 workers "
    "(`gsq/infer.py`) on every test passage and saves the plain head's logits with the passage ids. Any failure, or a run "
    "past 90 minutes, leaves the pipeline as it is without GSQ.",
    "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/gsq_infer_model.py": "GSQ model (`gsq/gsq_infer_model.py`, "
    "lightningv08, unchanged)\nThe 64-layer Qwen3.8-27B with 3-bit packed linear layers, its embedding decoded on the CPU, "
    "mean-pooled states at eight depths and the readout heads.",
    "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/gsq_infer_layers.py": "GSQ layers (`gsq/gsq_infer_layers.py`, "
    "lightningv08, unchanged)\nThe Triton kernel that unpacks the 3-bit weights on the T4, and the LoRA layers.",
    "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/delta_turing.py": "Gated delta rule (`gsq/delta_turing.py`, "
    "lightningv08, unchanged)\nA chunked gated-delta-rule kernel for the model's linear-attention layers on the T4 "
    "(adapted from the Qwen team's code, Apache 2.0).",
    "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/adapter.py": "Adapter loader (`gsq/adapter.py`, lightningv08, "
    "unchanged)\nLoads the final LoRA and heads after checking their training record.",
    "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/infer.py": "GSQ workers (`gsq/infer.py`, lightningv08, unchanged)\n"
    "One process per T4, each on every other passage; the rows are reassembled in order.",
    "%%writefile sibling_style.py": "Sibling-style head (`sibling_style.py`, Roman Tamrazov's V40A as in synthreaper's Sovereign Gold Apex, unchanged)\nStyle features and the per-family residual that `infer.py` applies after the typography step.",
    "%%writefile v34_pairwise.py": "Pairwise verifier (`v34_pairwise.py`, mercurydust's V34.1 as in Sovereign Gold Apex, unchanged)\nA same-model pair classifier on the official rows with the ChatGPT A/B annotations; writes per-passage scores for `infer.py`.",
    "# V34.1: run the lightweight A/B Pairwise verifier": "Run the pairwise verifier\nOn the test passages, before the main pipeline (about 30 s on the 3 visible rows).",
    "# complementary public stack in its own process": "Run the complementary stack, then the GSQ stage\nEach in its own "
    "processes, so all GPU memory is free again afterwards; their wall time is charged to ATLAS's budget.",
})


def build_v31ac(kernel, results, scored):
    """ntl-v31ac-* (exp027): the V31 document for our a20 code with Llama from alexchilton's public copy, ATLAS's weight
    ("-a20" / "-a30" / "-a40", "-m45": also on rows with top-2 margin below 0.45) and lightningv08's GSQ readout ("-gsqNN")."""
    nb = json.loads((HERE / kernel / f"{kernel}.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    codes = code_cells(nb)
    assert code_cells(json.loads(Path(scored).read_text(encoding="utf-8"))) == codes, "local build != scored version"
    tokens = kernel.split("-")
    pct = next(int(t[1:]) for t in tokens if re.fullmatch(r"a[1-9][05]", t))
    gpct = next((int(t[3:]) for t in tokens if re.fullmatch(r"gsq[1-9][0-9]", t)), None)
    sk = next((int(t[2:]) for t in tokens if re.fullmatch(r"sk[1-9][05]", t)), None)   # exp028 knobs
    cpct = next((int(t[4:]) for t in tokens if re.fullmatch(r"comp[1-9][0-9]", t)), None)
    m45 = "m45" in tokens
    osvc = next((t[4:] for t in tokens if re.fullmatch(r"osvc(12|15|20)", t)), None)   # 2026-11-01: open-pool SVM,
    osvc_w = {"12": "12", "15": "1.5", "20": "20"}.get(osvc)
    ax = next((int(t[2:]) for t in tokens if re.fullmatch(r"ax[1-9]", t)), None)       # retrained ATLAS x A^k,
    typo = next((f"{int(t[4:]) / 10:.1f}" for t in tokens if re.fullmatch(r"typo1[02]", t)), None)   # typography x1.2 / x1.0
    p30 = "p30" in tokens   # 2026-11-02: extra passages pinned to their 10-30 batches (judith007's rows)
    xm = next((int(t[2:]) for t in tokens if re.fullmatch(r"xm[1-4]", t)), None)   # matched open-pool external rows (areebbb's J / K)
    sib = "sib08" in tokens   # 2026-11-03: V40A alone at max_alpha 0.08
    sov = "sov" in tokens     # Sovereign Gold Apex's post-hoc steps (V40A 0.10 + V34.1, typography x1.35, ATLAS margin 0.32)
    g37 = "g37" in tokens     # exp042: +0.3 on gemini-3.7-flash after the balancing
    v35 = "v35" in tokens     # 2026-11-03: mohammedafreedpasha's V35 settings of the pairwise verifier
    pp13 = "pp13" in tokens   # the open pool read from our unmodified copy of its Kaggle version 13
    ew = "ew10" in tokens   # --extra_w 1.0
    cx = "cx" in tokens     # exp035: the stack's extra passages + the open pool's 12-model prompts
    tf = next((t for t in tokens if re.fullmatch(r"tf(50|75|100|150)", t)), None)   # the TF-IDF member's weight
    tfw = {"tf50": "0.5", "tf75": "0.75", "tf100": "1.0", "tf150": "1.5"}.get(tf)
    kx = "kx" in tokens     # exp039: kilupy's 18-model corpus appended to the matched rows
    mw = "mw" in tokens     # 2026-11-04 (exp046): the stack's open-pool rows weighted by match_weight
    kc = "kc" in tokens     # 2026-11-04 (exp039): kilupy's Claude passages in the stack's Claude gate
    v14 = "v14" in tokens   # 2026-11-04: the open pool at its Kaggle version 14, not pinned
    tpu = [(m.group(1), int(m.group(2)) / 10) for m in (re.fullmatch(r"(op8|ll8o|q14o)w(5|10|15|20|30)", t) for t in tokens) if m]
    assert tokens[:2] == ["ntl", "v31ac"] and set(tokens[2:]) <= {f"a{pct}", f"gsq{gpct}", "m45", f"sk{sk}", f"comp{cpct}",
                                                                  f"osvc{osvc}", f"ax{ax}", f"typo{typo and typo.replace('.', '')}",
                                                                  "p30", f"xm{xm}", "ew10", f"{tf}", "cx", "kx", "sib08", "sov", "g37",
                                                                  "v35", "pp13", "mw", "kc", "v14", *(f"{m}w{int(w * 10)}" for m, w in tpu)}, kernel
    assert not (sib and sov) and not ((sib or sov) and typo) and (sov or not v35)
    assert pp13 == (PP13_REF in meta["dataset_sources"]) and not (pp13 and OPEN_POOL_DS.split("/datasets/")[1] in meta["dataset_sources"])
    assert cx == ("# fork (exp035): the open pool's prompts" in "".join(codes))
    assert kx == ("# fork (exp039): kilupy's corpus" in codes[0]) and (not kx or xm in (1, 2)) and (not kx or tf or xm == 1)
    assert mw == ("# fork (exp046): with its typography match_weight" in "".join(codes)) and (cx or not mw)
    assert kc == ("# fork (exp039): kilupy's Claude rows" in "".join(codes)) \
        and (kx or kc) == ("kilupy/ntl-18-model-corpus" in meta["dataset_sources"])
    assert not (v14 and pp13) and (not v14 or OPEN_POOL_DS.split("/datasets/")[1] in meta["dataset_sources"])
    assert (f"_draw(g, min(len(g), int({xm} * _mix.get(m, 0))))" in codes[0] and osvc == "12") if xm \
        else "extra_matched.parquet" not in codes[0]
    run = codes[-1]
    assert f"'--atlas_w', '{pct / 100:.2f}'" in run and m45 == ("'--atlas_margin', '0.45'," in run)
    assert (f"'--gsq_w', '{gpct / 100:.2f}'" in run) if gpct else ("--gsq_w" not in run)
    assert (f"'--sinkhorn', '{sk / 100:.2f}'," in run) if sk else ("'--sinkhorn', '0.45'," in run)
    assert (f"'--comp_w', '{cpct / 100:.2f}'," in run) if cpct else ("'--comp_w', '0.15'," in run)
    assert ((f"'--svc_w', '{osvc_w}'," in run) if osvc else ("'--svc_w', '1.5'," in run)) and ("open_pool.parquet" in codes[0]) == bool(osvc)
    assert ((f"'--atlas_pow', '{ax}'," in run) if ax else ("--atlas_pow" not in run)) \
        and ("if 'ntl-atlas-full-ext' in x))" in codes[0]) == bool(ax)
    infer = next(c for c in codes if c.startswith("%%writefile infer.py"))
    assert (f"np.clip({typo} * _evidence" in infer if typo else True) and ("np.clip(1.5 * _evidence" in infer) != bool(typo or sov)
    assert sov == ("np.clip(1.35 * _evidence" in infer) == ("'--atlas_margin', '0.32'," in run) \
        == any(c.startswith("%%writefile v34_pairwise.py") for c in codes) == ("max_alpha=0.10, diagnostics_path" in infer)
    assert sib == ("max_alpha=0.08, diagnostics_path" in infer) and (sib or sov) == any(c.startswith("%%writefile sibling_style.py") for c in codes)
    assert g37 == ("tot[:, LABELS.index('gemini-3.7-flash')] += 0.3" in infer)
    pair = next((c for c in codes if c.startswith("%%writefile v34_pairwise.py")), "")
    assert v35 == ("_safe_logits + 0.12 * _gate[:, None] * _pair_log" in infer) == ("type=int,default=8); ap.add_argument('--seed'" in pair) \
        == ("np.sort(pp,axis=1)[:,-min(4,R):]" in pair) == ("'--refs-per-class', '8'," in "".join(codes))
    assert not sov or v35 or "_safe_logits + 0.10 * _gate[:, None] * _pair_log" in infer
    for m, w in tpu:   # our TPU-trained member: its line in the members cell, its kernel attached
        tag = TPU_DESC[m][1]
        assert f'{{"name": "{m}", "type": "l24"' in run and f"[adir(f'{tag}_f{{k}}') for k in range(5)] + [adir('{tag}')]" in run \
            and f'"w": {w}, ' in run.split(f'{{"name": "{m}"', 1)[1].split("\n", 1)[0], (m, w)
        assert TPU_KERNELS[m] in meta["kernel_sources"], m
    pinned = "            if _pin is not None:   # fork: only the batches these files had on 2026-10-28" in "".join(codes)
    assert p30 == ("            if _pin is not None:   # fork: only the batches these files had on 2026-10-30" in "".join(codes)) \
        and not (p30 and pinned) and (not p30 or ".isin([1, 2, 4, 5])\n" in "".join(codes))
    assert "mdir('/llama-3-1-8b-base/')" in codes[0] and "atlas predicted overrun" not in "".join(codes)
    assert (f"'--extra_w', '{'1.0' if ew else '0.5'}'," in run) and (f"'--tfidf_w', '{tfw or '0.25'}'," in run)
    title = f"{results['score']} | 0.660 V31 + ATLAS {pct}%" + (" on More Rows" if m45 else "") \
        + (f" + Qwen3.8-27B {gpct}%" if gpct else "") + (f" + Sinkhorn {sk / 100:.2f}" if sk else "") \
        + (f" + Stack {cpct}%" if cpct else "")
    if xm:   # areebbb's K family: the open pool behind every n-gram member
        knobs = [f"Sk {sk / 100:g}"] * (sk not in (None, 30)) + [f"Typo {typo}"] * bool(typo) \
            + ["Extra w 1"] * ew + [f"TF-IDF {tfw}"] * bool(tf) + ["Stack Pool"] * cx + ["Pool Weights"] * mw + ["Kilupy Rows"] * kx + ["Claude Gate"] * kc \
            + ["Sibling 0.08"] * sib + ["Sovereign"] * sov + ["V35 Verifier"] * v35 + ["Gemini 3.7 Fix"] * g37 \
            + [f"TPU {TPU_DESC[m][2]}" for m, _ in tpu]
        title = results.get("title") or next(t for t in (f"{results['score']} | {h}" + "".join(f" + {k}" for k in knobs) for h in (
            f"Open-Pool SVC + {xm}x Matched Rows", f"Open Pool + {xm}x Matched Rows", f"{xm}x Matched Rows")) if len(t) <= 50)
    elif p30 and not (osvc or ax or typo) and len(title + " + Newer Data") <= 50:
        title += " + Newer Data"
    elif osvc or ax or typo or p30:   # too many changes for the "0.660 V31 +" prefix: name the changes, shortest form that fits
        head = ["Retrained ATLAS" if ax else f"ATLAS {pct}%"] + (["Open Pool" if ax else "Open-Pool SVC"] if osvc else [])
        knobs = [[f"Sinkhorn {sk / 100:.2f}"] if sk else [], [f"Typography {typo}"] if typo else [], ["Newer Data"] if p30 else []]
        for form in (lambda k: k, lambda k: [x.replace("0.30", "0.3").replace("0.20", "0.2") for x in k],
                     lambda k: [x.replace("Sinkhorn ", "Sk ").replace("Typography ", "Typo ") for x in k],
                     lambda k: [f"{len(k)} Knobs"] if len(k) > 1 else k):
            title = f"{results['score']} | " + " + ".join(head + form(sum(knobs, [])))
            if len(title) <= 50:
                break
    assert 5 <= len(title) <= 50, title
    what = ["**a faster run** (the two 8B LoRA members on both GPUs at once and Llama-3.1-8B read from local disk, "
            "which leave the predictions unchanged, and ATLAS scoring up to its deadline)",
            f"**ATLAS mixed at {pct}% instead of 10%**" + (" on more passages" if m45 else "")]
    what += [f"**lightningv08's Qwen3.8-27B readout mixed at {gpct}%** (from his [0.661 notebook]({LV08.format(K=K)}))"] if gpct else []
    what += [f"**the balancing at α = {sk / 100:.2f} instead of 0.45**"] if sk else []
    what += [f"**the complementary stack at {cpct}% instead of 15%**"] if cpct else []
    what += [f"**ATLAS-8B retrained with open-pool passages and mixed more sharply (× ATLAS^{ax})**"] if ax else []
    what += [f"**the n-gram SVM trained on crtproduction's open pool, at weight {osvc_w}** (zbigniewkirchner's change)"] if osvc else []
    what += [f"**the typography evidence at ×{typo}**" + (" (judith007's change)" if typo == "1.2" else " (after judith007's ×1.2)")] if typo else []
    what += ["**the extra passages pinned to their 2026-10-28 versions**"] if pinned else []
    what += [f"**the TF-IDF, distort and SVM members' external rows drawn from the open pool by typography match, {xm}× "
             f"the old count** (areebbb's change: 1× in their scored version J, 2× in K)"] if xm else []
    what += ["**those external rows at full weight** (`--extra_w 1.0`)"] if ew else []
    what += ["**kilupy's 18-model corpus added to those rows** (16,603 Claude passages from the Anthropic API + 300 per other "
             "model, CC BY 4.0)"] if kx else []
    what += [f"**the TF-IDF member at weight {tfw}**"] if tf else []
    what += ["**the complementary stack's extra passages extended with the open pool's prompts that 12+ models answered**"] if cx else []
    what += ["**those open-pool rows weighted by their typography `match_weight`**"] if mw else []
    what += ["**kilupy's Anthropic-API Claude passages added to the stack's Claude gate**"] if kc else []
    what += ["**the extra passages pinned to their 2026-10-30 batches** (the 10,743 rows judith007's 0.664 read)"] if p30 else []
    what += ["**Roman Tamrazov's sibling-style residual (V40A) at the conservative max_alpha 0.08**"] if sib else []
    what += ["**synthreaper's Sovereign Gold Apex steps** (V40A sibling-style residual, mercurydust's V34.1 pairwise verifier, "
             "typography ×1.35, ATLAS margin 0.32)"] if sov else []
    what += ["**mohammedafreedpasha's V35 settings for that pairwise verifier** (8 reference passages per model instead of 6, "
             "the mean of the best 4 pairs instead of 3, smaller prototype supports, `-` among the punctuation marks, weight "
             "0.12 instead of 0.10)"] if v35 else []
    what += ["**the open pool read at its Kaggle version 13** (190,588 passages, the file the 0.667 runs read, from an "
             "unmodified copy)"] if pp13 else []
    what += ["**+0.3 on gemini-3.7-flash's log-score after the balancing**"] if g37 else []
    what += ["**the open pool read at its current Kaggle version 14** (211,369 passages, what a new copy attaches)"] if v14 else []
    what += [f"**a sixth LoRA24 member that we trained on Kaggle's free TPU v5e-8** ({TPU_DESC[m][0]}: crtproduction's open pool, "
             f"then the official rows; weight {w:g})" for m, w in tpu]
    a_sec = A20_SECTION.replace("In this recipe", "In the 0.651 recipe").format(
        pct=pct, reach=f"{pct / (100 - pct):.2f}", flag=f"{pct / 100:.2f}",
        inside=("inside the passages it now scores (margin below 0.45)" if m45 and pct / (100 - pct) <= 0.45
                else A_INSIDE[pct / (100 - pct) > (0.45 if m45 else 0.3)].replace("0.3 margin", "0.45 margin" if m45 else "0.3 margin")),
        cost="" if m45 else A_COST) + (M45_NOTE.format(pct=pct) if m45 else "") + results.get("a_note", "")
    g_sec = GSQ_SECTION.format(gpct=gpct, gflag=f"{gpct / 100:.2f}", lv08=LV08.format(K=K)) if gpct else ""
    k_note = results.get("k_note", "")
    k_sec = (SK_SECTION.format(alpha=f"{sk / 100:.2f}", direction="softer" if sk < 45 else "stronger",
                               soft_sk=SOFT_SK.format(K=K), k_note=k_note) if sk else "") \
        + (COMP_SECTION.format(cpct=cpct, cflag=f"{cpct / 100:.2f}", k_note=k_note) if cpct else "")
    p_sec = PIN_SECTION if pinned else P30_SECTION.format(ju=JU_0664.format(K=K), p_note=results.get("p_note", "")) if p30 else ""
    fmt = dict(K=K, zk=ZK_0664.format(K=K), ju=JU_0664.format(K=K), pool=OPEN_POOL_DS.format(K=K), full=ATLAS_FULL.format(K=K),
               ar=AR_0666.format(K=K), ac=AC_ALL.format(K=K), aj=AR_J_AJ.format(K=K), rk=AR_J_RK.format(K=K))
    o_sec = OSVC_SECTION.format(w=osvc_w, k_note=results.get("o_note", ""), **fmt) if osvc else ""
    if xm:   # release v11 of the pool (Kaggle version 13) holds 190,588 passages, release v12 (version 14) 211,369
        o_sec = o_sec.replace("~198k public real-model passages (CC BY-SA 4.0)",
                              "211,369 public real-model passages (release v12, CC BY-SA 4.0)" if v14 else
                              "190,588 public real-model passages (release v11, CC BY-SA 4.0)")
    x_sec = "\n".join(x for x in (AX_SECTION.replace("{k}", str(ax)).format(**fmt) if ax else "", o_sec,
                                   XM_SECTION.format(xm=xm, n_rows=XM_ROWS[xm], x_note=results.get("x_note", ""), **fmt) if xm else "",
                                   KX_SECTION.format(kil=KILUPY_DS, kx_rows=KX_ROWS[xm], kx_tf=KX_TF[bool(tf)],
                                                     kx_note=results.get("kx_note", "")) if kx else "",
                                   TF_SECTION.format(tfw=tfw, t_note=results.get("t_note", "")) if tf else "",
                                   CX_SECTION.format(c_note=results.get("c_note", "")) if cx else "",
                                   MW_SECTION.format(mw_note=results.get("mw_note", "")) if mw else "",
                                   KC_SECTION.format(sey=SEY_DS, kil=KILUPY_DS, kc_note=results.get("kc_note", "")) if kc else "",
                                   SIB_SECTION.format(s_note=results.get("s_note", ""), who=WHO.format(K=K), sov=SOV_NB.format(K=K),
                                                      kt=KT_SIB.format(K=K), lv=LV08_0666.format(K=K)) if sib else "",
                                   SOV_SECTION.format(v_note=results.get("v_note", ""), sov=SOV_NB.format(K=K),
                                                      ac6=AC_0666.format(K=K), ab=AB_DS.format(K=K)) if sov else "",
                                   V35_SECTION.format(af=AF_0667.format(K=K), v35_note=results.get("v35_note", "")) if v35 else "",
                                   PP13_SECTION.format(pool=OPEN_POOL_DS.format(K=K), zk7=ZK_0667.format(K=K), mh=MH_0667.format(K=K),
                                                       ac7=AC_0667.format(K=K), pin=f"{K}/datasets/{PP13_REF}",
                                                       pp_note=results.get("pp_note", "")) if pp13 else "",
                                   G37_SECTION.format(g_note=results.get("g_note", "")) if g37 else "",
                                   *(TPU_SECTION.format(desc=TPU_DESC[m][0], keep=27 if m == "q14o" else 24, w=f"{w:g}",
                                                        kernel=TPU_KERNELS[m].split("/")[1], tk=f"{K}/code/{TPU_KERNELS[m]}",
                                                        gp=results.get("gp_url", f"{K}/competitions/name-that-llm/discussion"),
                                                        pool=OPEN_POOL_DS.format(K=K), pool_rows=results.get("tpu_pool_rows", "190,588"),
                                                        tpu_note=results.get("tpu_note", "")) for m, w in tpu),
                                   TYPO_SECTION.format(scale=typo, this=TYPO_THIS.get(typo), **fmt) if typo else "") if x)
    pay = [PAY_COPY_AC, PAY_DP.format(n8="two", names="Qwen3-8B, Llama-3.1-8B", dp_check=results.get("dp_check", "")),
           PAY_DEADLINE_V31]
    doc = V31_DOC.format(
        title=title, score=results["score"], K=K, v31=V31.format(K=K), who=WHO.format(K=K), soft_sk=SOFT_SK.format(K=K),
        what=", ".join(what[:-1]) + " and " + what[-1],
        rows="\n".join(f"| {a} | {b} | {c} |" for a, b, c in results["rows"]),
        sections="".join(s + "\n" for s in (a_sec, g_sec, k_sec, x_sec, p_sec) if s),
        pay="\n".join(pay), atlas_pct=pct, inputs=inputs_table(meta),
        dnt_rows="".join(f"\n| {a} | {b} | {c} |" for a, b, c in results.get("did_not_transfer", [])))
    cache = "{K}/code/avazaku/name-that-llm-verified-llama-asset-cache".format(K=K)
    swaps = [
        (f"with Llama-3.1-8B from his [public, hash-checked cache]({cache}).",
         f"with Llama-3.1-8B from his hash-checked cache. That cache has since been deleted; this version reads the same "
         f"Llama-3.1-8B files from alexchilton's public [Llama 3.1 8B base]({AC_MODEL.format(K=K)}) model (we checked the "
         "sha256 of every weight shard and the hashes of the config, tokenizer and index files)."),
        (f"and the [Llama asset cache]({cache}) this notebook loads Llama from.",
         "and the Llama asset cache (since deleted) our earlier versions loaded Llama from.\n"
         f"* **alexchilton** — the public [Llama 3.1 8B base]({AC_MODEL.format(K=K)}) model this notebook loads Llama from."),
    ]
    if gpct:
        swaps += [
            ("5. **Balancing** — V31's soft Sinkhorn (above).\n6. **Typography step** — V31's, which writes the submission.",
             f"5. **Qwen3.8-27B readout** — lightningv08's GSQ3 readout mixed at {gpct}% into every passage.\n"
             "6. **Balancing** — V31's soft Sinkhorn (above).\n7. **Typography step** — V31's, which writes the submission."),
            ("On the 3 visible rows the commit takes ~5 min: it resolves every input and writes a placeholder;",
             "On the 3 visible rows the commit takes ~15 min: it resolves every input, runs the GSQ stage on them and writes "
             "a placeholder;"),
            ("* **PeterDreamHan** —",
             f"* **lightningv08 (Vladislav Lavrentev)** — the Qwen3.8-27B GSQ3 readout, its five `gsq/` files and wheels "
             f"([0.661 notebook]({LV08.format(K=K)}), [assets]({LV08_DS.format(K=K)}); Qwen3.8-27B: Alibaba Qwen team, "
             "Apache 2.0).\n* **PeterDreamHan** —"),
        ]
    if ax:
        swaps += [
            ("ATLAS ([Ko et al.](https://arxiv.org/abs/2610.19583), Joseph Adamski's Qwen3-8B bundle) is",
             "ATLAS ([Ko et al.](https://arxiv.org/abs/2610.19583), Joseph Adamski's Qwen3-8B bundle; retrained here, see the next section) is"),
            (f"4. **ATLAS** — a source-conditioned likelihood (Joseph Adamski's public bundle) mixed at {pct}% on passages whose top-2 margin is below 0.3.",
             f"4. **ATLAS** — a source-conditioned likelihood (Joseph Adamski's ATLAS, retrained by us with open-pool passages) mixed at {pct}% "
             f"on passages whose top-2 margin is below 0.3, then those passages multiplied by ATLAS^{ax}."),
            ("* **PeterDreamHan** —",
             f"* **Joseph Adamski** — ATLAS's code and training configuration, which our retraining ([ntl-atlas-full-ext]({ATLAS_FULL.format(K=K)})) "
             "reuses unchanged (code Apache 2.0).\n* **PeterDreamHan** —"),
        ]
    if osvc:
        swaps += [("the 4-view n-gram SVM (1.5);", f"the 4-view n-gram SVM ({osvc_w}, trained with crtproduction's open pool);"),
                  ("* **PeterDreamHan** —",
                   f"* **zbigniewkirchner** — the open-pool SVM and its weight ([0.664 notebook]({ZK_0664.format(K=K)})).\n* **PeterDreamHan** —")]
    if xm:
        swaps += [("TF-IDF (0.25) and text-distortion (0.10) members.",
                   f"TF-IDF ({tfw or '0.25'}) and text-distortion (0.10) members; these three learn from the matched "
                   f"open-pool rows{' at full weight' if ew else ''} besides the official ones."),
                  ("* **PeterDreamHan** —",
                   f"* **areebbb** — the matched open-pool rows behind the n-gram members, drawn by `match_weight` "
                   f"([0.666 notebook]({AR_0666.format(K=K)}): version J, 1×, scored 0.666; version K draws 2×).\n"
                   f"* **alexchilton** — the [all-public fork]({AC_ALL.format(K=K)}) of our 0.663 that ladder runs on, and "
                   f"the public Llama-3.1-8B upload.\n"
                   + (f"* **kilupy** — the [18-model prompt-paired corpus]({KILUPY_DS}) (CC BY 4.0): Claude passages from "
                      "the Anthropic API and OpenRouter passages for the other 13 models.\n" if kx or kc else "")
                   + "* **PeterDreamHan** —")]
    if osvc or ax:
        swaps += [("* **PeterDreamHan** —",
                   f"* **crtproduction** — the [Name That LLM Open Pool]({OPEN_POOL_DS.format(K=K)}) (CC BY-SA 4.0)"
                   + (" and its typography match weights" if xm else "") + ".\n* **PeterDreamHan** —")]
    if typo:
        swaps += [("6. **Typography step** — V31's, which writes the submission.",
                   f"6. **Typography step** — V31's, with its evidence scaled by {typo} instead of 1.5; it writes the submission."),
                  ("* **PeterDreamHan** —",
                   f"* **judith007** — the {'×1.2 typography scale' if typo == '1.2' else 'smaller typography scale (×1.2) this version takes further'} "
                   f"([0.664 notebook]({JU_0664.format(K=K)})).\n* **PeterDreamHan** —")]
    elif p30:
        swaps += [("* **PeterDreamHan** —",
                   f"* **judith007** — the [0.664 notebook]({JU_0664.format(K=K)}) whose newer extra passages this version pins."
                   "\n* **PeterDreamHan** —")]
    if sib or sov:
        swaps += [("6. **Typography step** — V31's, which writes the submission.",
                   "6. **Typography step** — V31's" + (", its evidence ×1.35 (Sovereign)" if sov else "") + ".\n"
                   f"7. **Sibling-style residual** — Roman Tamrazov's V40A at max_alpha {'0.10' if sov else '0.08'}"
                   + ((", then mercurydust's V34.1 pairwise verifier with V35's settings at 0.12" if v35 else
                       ", then mercurydust's V34.1 pairwise verifier at 0.10") + " on passages whose top-2 margin is below 0.35" if sov else "")
                   + "; it writes the submission."),
                  ("* **PeterDreamHan** —",
                   f"* **synthreaper (Aditya Goyal)** — [Sovereign Gold Apex]({SOV_NB.format(K=K)}), "
                   + ("whose post-hoc cells and block this version adds verbatim; **mercurydust** — the V34.1 pairwise verifier "
                      f"and the [ChatGPT A/B annotations]({AB_DS.format(K=K)}) it reads (dataset licence unknown; its upstream terms "
                      f"apply); **alexchilton** — the [verbatim 0.666 run]({AC_0666.format(K=K)}) of Sovereign; **zbigniewkirchner** — "
                      f"the [0.667 notebook]({ZK_0667.format(K=K)}) that first joined areebbb's J rows with Sovereign's steps.\n" if sov else
                      "which brought Roman Tamrazov's V40A sibling-style residual into this line; **Kirill Tymchishin** and "
                      f"**lightningv08 (Vladislav Lavrentev)** — its conservative 0.08 cap ([gentle sibling correction]"
                      f"({KT_SIB.format(K=K)}), [0.666 notebook]({LV08_0666.format(K=K)})).\n")
                   + "* **PeterDreamHan** —")]
    if sov:
        swaps += [("mixed at {pct}% on passages whose top-2 margin is below 0.3.".format(pct=pct),
                   "mixed at {pct}% on passages whose top-2 margin is below 0.32 (Sovereign's margin).".format(pct=pct))]
    if v35:
        swaps += [("* **PeterDreamHan** —",
                   f"* **mohammedafreedpasha** — the V35 settings of the pairwise verifier ([0.667 notebook]({AF_0667.format(K=K)})).\n"
                   "* **PeterDreamHan** —")]
    if pp13:
        assert xm and osvc
        crt = f"* **crtproduction** — the [Name That LLM Open Pool]({OPEN_POOL_DS.format(K=K)}) (CC BY-SA 4.0) and its typography match weights."
        swaps += [(crt, crt[:-1] + f"; this notebook reads its version 13 from our unmodified copy, [ntl-open-pool-v13-pin]({K}/datasets/{PP13_REF}).")]
    if tpu:
        swaps += [("Nishant Dahal's Qwen3-14B (2.5) and Qwen2.5-14B (1) part-two checkpoints;",
                   "Nishant Dahal's Qwen3-14B (2.5) and Qwen2.5-14B (1) part-two checkpoints; "
                   + "; ".join(f"our TPU-trained {TPU_DESC[m][0]} LoRA24 member ({w:g}, see above)" for m, w in tpu) + ";"),
                  ("* **PeterDreamHan** —",
                   f"* **GoPain** — the [TPU v5e-8 measurements]({results.get('gp_url', K + '/competitions/name-that-llm/discussion')}) "
                   "our TPU trainer follows (torch_xla SPMD, bf16, fixed shapes, XLA checkpointing).\n* **PeterDreamHan** —")]
    if g37:
        swaps += [("5. **Balancing** — V31's soft Sinkhorn (above).",
                   "5. **Balancing** — V31's soft Sinkhorn (above), then +0.3 on gemini-3.7-flash's log-score (see above).")]
    if results.get("tldr"):   # a short summary box right under the header line
        swaps += [("\n\nThis is mercurydust's [", "\n\n" + results["tldr"] + "\n\nThis is mercurydust's [")]
    for a, b in swaps:
        assert doc.count(a) == 1, a
        doc = doc.replace(a, b)
    extra = {"inputs_extra": ("; lightningv08's asset collection is hidden from these searches (a `glob.glob` filter)" if gpct else "")
             + ("; ATLAS's bundle is our retraining (the one path containing `ntl-atlas-full-ext`)" if ax else "")
             + ("; the block after the pool search swaps the SVM's pool for crtproduction's open pool" if osvc else "")
             + (" and draws the n-gram members' external rows from it by typography match" if xm else "")
             + (", then appends kilupy's 18-model corpus to those rows" if kx else "")
             + ("; the open pool is our unmodified copy of its Kaggle version 13" if pp13 else ""),
             "infer_extra": " Our changes: in `l24_predict`, members marked `\"dp\"` (the 8B ones) load a copy on each GPU "
                            "and alternate batches between them, and every member averages its hidden states on the GPU; "
                            "ATLAS no longer stops on a projected overrun"
                            + ("; after ATLAS, the GSQ readout is mixed in (`--gsq_w`)" if gpct else "")
                            + ("; after ATLAS's linear mix, its passages are multiplied by ATLAS^k (`--atlas_pow`)" if ax else "")
                            + (f"; the typography evidence is scaled by {typo} instead of 1.5" if typo else "")
                            + ("; after the balancing, +0.3 on gemini-3.7-flash's log-score" if g37 else "")
                            + "".join(f"; the members cell lists our TPU-trained {TPU_DESC[m][0]} member (`{m}`, weight {w:g})" for m, w in tpu)
                            + ("; after the typography step, V40A's sibling-style residual (max_alpha 0.08)" if sib else "")
                            + ("; the typography evidence at ×1.35, and after the typography step Sovereign's V40A residual and V34.1 "
                               "pairwise fusion" if sov else "") + (" at 0.12 (V35; Sovereign: 0.10)" if v35 else "") + ".",
             "run_text": "The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 "
                         "--sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run "
                         "writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell. The run command "
                         f"passes `'--atlas_w', '{pct / 100:.2f}'` (V31: 0.10)"
                         + (", `'--atlas_margin', '0.45'` (V31: 0.3)" if m45 else "")
                         + (f" and `'--gsq_w', '{gpct / 100:.2f}'`" if gpct else "")
                         + (f", `'--sinkhorn', '{sk / 100:.2f}'` (V31: 0.45)" if sk else "")
                         + (f", `'--comp_w', '{cpct / 100:.2f}'` (V31: 0.15)" if cpct else "")
                         + (f", `'--svc_w', '{osvc_w}'` (V31: 1.5)" if osvc else "")
                         + (", `'--extra_w', '1.0'` (V31: 0.5)" if ew else "") + (f", `'--tfidf_w', '{tfw}'` (V31: 0.25)" if tf else "")
                         + (f", `'--atlas_pow', '{ax}'` (new)" if ax else "")
                         + (", `'--atlas_margin', '0.32'` (V31: 0.3)" if sov else "") + "."}
    table = V31AC_SECTIONS if gpct else dict(V31AC_SECTIONS, **{k: V31_SECTIONS[k] for k in ("# complementary public stack in its own process",)})
    if v35:
        table = dict(table, **{
            "%%writefile v34_pairwise.py": "Pairwise verifier (`v34_pairwise.py`, mercurydust's V34.1 as in Sovereign Gold Apex, "
            "with mohammedafreedpasha's V35 settings)\nA same-model pair classifier on the official rows with the ChatGPT A/B "
            "annotations; writes per-passage scores for `infer.py`. V35: 8 references per model, top-4 mean, prototype "
            "supports 6 / 5, `-` among the punctuation marks.",
            "# V34.1: run the lightweight A/B Pairwise verifier": "Run the pairwise verifier\nOn the test passages, before the "
            "main pipeline, with `--refs-per-class 8` (about 35 s on the 3 visible rows)."})
    if p30 or cx:   # the stack's extra-passage loader is where both changes live
        k = "%%writefile comp_stack.py"
        table = dict(table, **{k: table[k] + " Its extra-passage loader"
                               + (" keeps the 2026-10-30 batches (a whitelist of batch tags)" if p30 else "")
                               + (" and" if p30 and cx else "")
                               + (" also reads the open pool's prompts answered by at least 12 models (ids `op_...`)" if cx else "") + "."
                               + (" Those rows count by their `match_weight` in the stylegram SVC, its centred twin and the gates." if mw else "")
                               + (" The Claude gate also learns from kilupy's Claude passages." if kc else "")})
    heads = [next(h for key, h in table.items() if code.startswith(key)).format(**extra) for code in codes]   # the GSQ heading only with GSQ
    cells = [{"cell_type": "markdown", "metadata": {}, "source": doc}]
    for i, (head, code) in enumerate(zip(heads, codes), 1):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": f"## {i}. {head}"})
        cells.append({"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None})
    out_nb = dict(nb, cells=cells)
    assert code_cells(out_nb) == codes   # the scored code, unchanged and in order
    out = HERE / "publish" / kernel
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out_nb, indent=1, ensure_ascii=False)
    (out / f"{kernel}.ipynb").write_text(body, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(dict(meta, title=title, is_private=False), indent=2) + "\n",
                                              encoding="utf-8")
    sha = hashlib.sha256("".join(codes).encode()).hexdigest()[:16]
    print(f"publish/{kernel}: title {title!r}, {len(cells)} cells, code sha256 {sha}, doc {len(doc)} chars")


def build_v31(kernel, results, scored):
    nb = json.loads((HERE / kernel / f"{kernel}.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    codes = code_cells(nb)
    assert code_cells(json.loads(Path(scored).read_text(encoding="utf-8"))) == codes, "local build != scored version"
    tokens = kernel.split("-")
    pct = next((int(t[1:]) for t in tokens if re.fullmatch(r"a[1-9]0", t)), None)
    mg = "mg" in tokens
    assert tokens[:3] == ["ntl", "v31", "dp"] and set(tokens[3:]) <= {f"a{pct}", "mg"}, kernel
    assert f"'--atlas_w', '{(pct or 10) / 100:.2f}'" in codes[-1] and "atlas predicted overrun" not in "".join(codes)
    infer = next(c for c in codes if c.startswith("%%writefile infer.py"))
    assert mg == ("    _gate = _confidence_gate   # fork (exp025)" in infer)
    changes = ([f"ATLAS {pct}%"] if pct else []) + (["Typography Margin Gate" if not pct else "Margin Gate"] if mg else [])
    title = f"{results['score']} | 0.660 V31 + " + (" + ".join(changes) if changes else "Faster Run")
    assert 5 <= len(title) <= 50, title
    what = ["**a faster run** (the two 8B LoRA members on both GPUs at once and Llama-3.1-8B read from local disk, "
            "which leave the predictions unchanged, and ATLAS scoring up to its deadline)"]
    what += [f"**ATLAS mixed at {pct}% instead of 10%**"] if pct else []
    what += ["**the typography step with its margin gate only**"] if mg else []
    a20 = A20_SECTION.replace("In this recipe", "In the 0.651 recipe").format(
        pct=pct, reach=f"{(pct or 10) / (100 - (pct or 10)):.2f}", flag=f"{(pct or 10) / 100:.2f}",
        inside=A_INSIDE[(pct or 10) / (100 - (pct or 10)) > 0.3], cost=A_COST) + results.get("a_note", "") if pct else ""
    pay = [PAY[0], PAY_DP.format(n8="two", names="Qwen3-8B, Llama-3.1-8B", dp_check=results.get("dp_check", "")),
           PAY_DEADLINE_V31]
    doc = V31_DOC.format(
        title=title, score=results["score"], K=K, v31=V31.format(K=K), who=WHO.format(K=K), soft_sk=SOFT_SK.format(K=K),
        what=", ".join(what[:-1]) + " and " + what[-1] if len(what) > 1 else what[0],
        rows="\n".join(f"| {a} | {b} | {c} |" for a, b, c in results["rows"]),
        sections="".join(s.format(K=K) + "\n" for s in (a20, MG_SECTION if mg else "") if s),
        pay="\n".join(pay), atlas_pct=pct or 10, inputs=inputs_table(meta),
        dnt_rows="".join(f"\n| {a} | {b} | {c} |" for a, b, c in results.get("did_not_transfer", [])))
    extra = {"inputs_extra": "",
             "infer_extra": " Our changes: in `l24_predict`, members marked `\"dp\"` (the 8B ones) load a copy on each GPU "
                            "and alternate batches between them, and every member averages its hidden states on the GPU; "
                            "ATLAS no longer stops on a projected overrun"
                            + ("; the typography gate is the margin gate alone" if mg else "") + ".",
             "run_text": "The 0.651 members and weights with V31's run flags (`--tfidf_w 0.25 --distort_w 0.10 "
                         "--sinkhorn 0.45 --sinkhorn_target_mix 0.35 --post 0`). On the 3 visible rows the commit run "
                         "writes a placeholder; Kaggle's rerun on the ~4,000 hidden rows runs every cell."
                         + (f" The run command passes `'--atlas_w', '{pct / 100:.2f}'` (V31: 0.10)." if pct else "")}
    heads = [next(h for key, h in V31_SECTIONS.items() if code.startswith(key)).format(**extra) for code in codes]
    if results.get("llama_cache_gone"):   # avazaku's Llama cache was deleted after our versions scored (2026-10-29)
        for a, b in LLAMA_GONE:
            assert doc.count(a) == 1, a
            doc = doc.replace(a, b)
    cells = [{"cell_type": "markdown", "metadata": {}, "source": doc}]
    for i, (head, code) in enumerate(zip(heads, codes), 1):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": f"## {i}. {head}"})
        cells.append({"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None})
    out_nb = dict(nb, cells=cells)
    assert code_cells(out_nb) == codes   # the scored code, unchanged and in order
    out = HERE / "publish" / kernel
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out_nb, indent=1, ensure_ascii=False)
    (out / f"{kernel}.ipynb").write_text(body, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(dict(meta, title=title, is_private=False), indent=2) + "\n",
                                              encoding="utf-8")
    sha = hashlib.sha256("".join(codes).encode()).hexdigest()[:16]
    print(f"publish/{kernel}: title {title!r}, {len(cells)} cells, code sha256 {sha}, doc {len(doc)} chars")


def code_cells(nb):
    return [(c["source"] if isinstance(c["source"], str) else "".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def inputs_table(meta):
    rows = ["| Type | Source |", "|---|---|"]
    rows += [f"| Dataset | [{d}]({K}/datasets/{d}) |" for d in meta["dataset_sources"]]
    rows += [f"| Notebook output | [{k}]({K}/code/{k}) |" for k in meta["kernel_sources"]]
    rows += [f"| Model | [{m}]({K}/models/{m.lower()}) |" for m in meta["model_sources"]]
    rows += [f"| Competition | [{c}]({K}/competitions/{c}) |" for c in meta["competition_sources"]]
    return "\n".join(rows)


def build(kernel, results, scored):
    if kernel.startswith("ntl-v31-"):
        return build_v31(kernel, results, scored)
    if kernel.startswith("ntl-v31ac-"):
        return build_v31ac(kernel, results, scored)
    nb = json.loads((HERE / kernel / f"{kernel}.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    codes = code_cells(nb)
    assert code_cells(json.loads(Path(scored).read_text(encoding="utf-8"))) == codes, "local build != scored version"
    tokens = kernel.split("-")
    mn8, a2, dp, m45 = any(t.startswith("mn8") for t in tokens), "a2" in tokens, "dp" in tokens, "m45" in tokens
    pct = next((int(t[1:]) for t in tokens if re.fullmatch(r"a[1-9]0", t)), None)   # ATLAS-8B's weight in %, if changed
    sk = next((f"{int(t[2:]) / 100:g}" for t in tokens if re.fullmatch(r"sk[0-9]{2}", t)), None)   # Sinkhorn alpha, if changed
    v38 = tokens[1] == "v38"
    assert mn8 == (kernel in WEIGHT) and (mn8 or dp) and (a2 or pct or mn8 or sk) and (pct or not m45), kernel
    assert f" --sinkhorn {sk or '1.0'} --post 1 " in codes[-1] and not (sk and "--sinkhorn 1.0" in codes[-1])
    assert a2 == any(c.startswith("%%writefile atlas2_worker.py") for c in codes)
    assert (f"--atlas_w {pct / 100:.2f} " in codes[-1] if pct else re.search(r"--atlas_w 0(\.10)? ", codes[-1])) \
        and m45 == ("--atlas_margin 0.45 " in codes[-1])
    assert v38 == any(c.startswith("%%writefile infer.py") and "_v38_before_hard = tot.copy()" in c for c in codes)
    score, rows = results["score"], results["rows"]
    title = f"{score} | {'0.654' if v38 else '0.651'} Recipe" \
        + (" + Ministral" + ("-3-8B Member" if not (a2 or pct) else "") if mn8 else "") \
        + (" + Second ATLAS" + (" (Qwen3-4B)" if not (mn8 or pct) else "") if a2 else "") + (f", Sinkhorn {sk}" if sk else "") \
        + (f", ATLAS {pct}%" if pct else "") \
        + (" on More Rows" if m45 else "")
    assert 5 <= len(title) <= 50, title
    what = [f"**Roman Tamrazov's typography step** from [Name That LLM | Who am I?]({WHO.format(K=K)}) (public 0.654; it "
            "replaces the recipe's last step)"] if v38 else []
    what += [f"**Sinkhorn balancing at {sk} instead of 1.0** (from PeterDreamHan's [Soft Sinkhorn 0.5]({SOFT_SK.format(K=K)}), "
             "public 0.653)"] if sk else []
    what += [f"**one new ensemble member we trained: a two-stage LoRA24 model on Ministral-3-8B-Base** (Mistral AI, Apache "
            f"2.0), a decoder family the recipe did not have, at blend weight {WEIGHT[kernel]} (Llama's is 1.0)"] if mn8 else []
    what += ["**a second ATLAS model**: harshitakumari256's public Qwen3-4B ATLAS bundle, mixed at 10% after the "
             "recipe's own (Qwen3-8B) ATLAS on the same uncertain passages"] if a2 else []
    what += [f"**the recipe's ATLAS mixed at {pct}% instead of 10%**"
             + (" on the passages with top-2 margin below 0.45 instead of 0.3" if m45 else "")] if pct else []
    mode = ("dp" if dp else "deadline" if kernel.endswith("-deadline") else "noatlas") if mn8 else \
        "dp-plain-m45" if m45 else "dp-plain"
    names = "Qwen3-8B, Llama-3.1-8B, Ministral-3-8B" if mn8 else "Qwen3-8B, Llama-3.1-8B"
    pay = PAY + ([PAY_DP.format(n8="three" if mn8 else "two", names=names, dp_check=results.get("dp_check", ""))] if dp else [])
    head, intro = SPEED_TEXT["mn8" if mn8 else "a2" if a2 else "m45" if m45 else "plain"]
    control = next((r[2] for r in rows if "ATLAS off" in r[1] and "Ministral" not in r[1]), "n/a")
    doc = DOC.format(
        title=title, score=score, K=K, what=", ".join(what[:-1]) + " and " + what[-1] if len(what) > 1 else what[0], atlas=ATLAS_TEXT[mode].format(control=control),
        rows="\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows),
        sections="".join(s.format(K=K, who=WHO.format(K=K)) + "\n" for s, on in
                         ((V38_SECTION, v38), (SK_SECTION.replace("{alpha}", str(sk)).replace("{soft_sk}", SOFT_SK), sk),
                          (MN8_SECTION, mn8), (A2_SECTION, a2), (A20_SECTION.format(
                             pct=pct, reach=f"{(pct or 10) / (100 - (pct or 10)):.2f}", flag=f"{(pct or 10) / 100:.2f}",
                             inside=M45_INSIDE if m45 else A_INSIDE[(pct or 10) / (100 - (pct or 10)) > 0.3],
                             cost=M45_COST if m45 else A_COST), pct)) if on),
        builds_on="the recipes it builds on: 0.651 and 0.654" if v38 else "the recipe it builds on: 0.651",
        v38_credit=(f"* **Roman Tamrazov** — the typography step and the 0.654 notebook it comes from ([Name That LLM | "
                    f"Who am I?]({WHO.format(K=K)})), copied verbatim.\n" if v38 else "")
        + (f"* **PeterDreamHan** — the [Soft Sinkhorn 0.5]({SOFT_SK.format(K=K)}) ablation (public 0.653), whose one "
           "change this version uses.\n" if sk else ""),
        dnt_rows="".join(f"\n| {a} | {b} | {c} |" for a, b, c in results.get("did_not_transfer", [])),
        speed_head=head, speed_intro=intro.format(npay=["", "", "Two", "Three"][len(pay)],
                                                  back="it back with time to spare" if dp else "most of it back"),
        pay="\n".join(pay), inputs=inputs_table(meta),
        commit="~15 min: it checks the Ministral member and writes a placeholder" if mn8 else
               "~5 min: it resolves every input and writes a placeholder",
        mn8_recipe=" our member uses" if mn8 else "",
        a2_credit=(" **harshitakumari256** — the [Qwen3-4B ATLAS bundle]({K}/datasets/harshitakumari256/ntl-atlas-bundle) "
                   "(CC BY-SA 4.0)." if a2 else "").format(K=K),
        mn8_licence="**Ministral-3-8B-Base-2512** — Mistral AI, Apache 2.0. " if mn8 else "")
    extra = {"inputs_extra": ("; our Ministral member's base and its six adapters are looked up here too" if mn8 else "")
             + ("; so are the second ATLAS bundle and Qwen3-4B-Base" if a2 else ""),
             "infer_extra": (" One change in `l24_predict`: members marked `\"dp\"` (the 8B ones) load a copy on each GPU "
                             "and alternate batches between them, and every member averages its hidden states on the GPU "
                             "(see the time section above)." if dp else "")
             + (" `atlas2_scores` and its call after ATLAS-8B add the second ATLAS." if a2 else "")
             + (" At its end, Roman Tamrazov's typography step (verbatim from his 0.654 notebook) writes the submission "
                "in place of the guards and dense-group assignment." if v38 else ""),
             "run_text": ("The 0.651 members and weights, plus `mn8v1` (our Ministral-3-8B member). On the 3 visible rows "
                          "the commit run only checks that `mn8v1` loads and reproduces its fit (in-sample accuracy on 40 "
                          "train rows, asserted > 0.5) and writes a placeholder" if mn8 else
                          "The 0.651 members and weights. On the 3 visible rows the commit run writes a placeholder")
             + "; Kaggle's rerun on the ~4,000 hidden rows runs every cell."
             + (f" The run command passes `--sinkhorn {sk}` (the recipe: 1.0)"
                + (f" and `--atlas_w {pct / 100:.2f}` (0.10)" if pct else "") + "." if sk else "")}
    heads = [next(h for key, h in SECTIONS.items() if code.startswith(key)).format(**extra) for code in codes]
    cells = [{"cell_type": "markdown", "metadata": {}, "source": doc}]
    for i, (head, code) in enumerate(zip(heads, codes), 1):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": f"## {i}. {head}"})
        cells.append({"cell_type": "code", "metadata": {}, "source": code, "outputs": [], "execution_count": None})
    out_nb = dict(nb, cells=cells)
    assert code_cells(out_nb) == codes   # the scored code, unchanged and in order
    out = HERE / "publish" / kernel
    out.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out_nb, indent=1, ensure_ascii=False)
    (out / f"{kernel}.ipynb").write_text(body, encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(dict(meta, title=title, is_private=False), indent=2) + "\n",
                                              encoding="utf-8")
    sha = hashlib.sha256("".join(codes).encode()).hexdigest()[:16]
    print(f"publish/{kernel}: title {title!r}, {len(cells)} cells, code sha256 {sha}, doc {len(doc)} chars")


if __name__ == "__main__":
    build(sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), sys.argv[3])
