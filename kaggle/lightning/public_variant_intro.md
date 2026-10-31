# {TITLE}

*Name That LLM: {TAGLINE}*

This is **[Vladislav Lavrentev's 0.642 | LoRA + Llama + Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b)**
(version 28, the version that scored 0.642) with **{CHANGES}**, described below. Every model, weight, share and
balancing step is otherwise upstream's. The upstream notebook stands on
[fleongg's NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement),
[Kirill Tymchishin's 0.639 | Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature)
with his [spectral readout bank](https://www.kaggle.com/datasets/kirilltym/ntl-spectral-style-heads-paired-4404), and
[Nishant Dahal's two-stage LoRA, part two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two). All
13 public notebooks at 0.635+ are credited in the table further down.

{CHANGE}

## Results: same base, same day, same Docker image

Every row below is one of my submissions of the same V28 code, scored on the same public rows. Only the listed change
differs, and the first row is V28 itself, unmodified apart from the mount paths, as a control.

{RESULTS}

How to read this: the public LB shows 3 decimals and moves by about ±0.002 between runs of near-identical code, so a
difference of 0.001 is not evidence on its own. These are several near-noise variants scored on one day, so the best of
them is a little lucky by selection. The control row is the fair baseline for this table, not the listed 0.642.

## Acknowledgements: every public notebook at 0.635+

Read at the version that earned its score (2026-10-25). Everything here descends from one stack, and each row is
credited for what it added or confirmed.

{ACK}

## Other differences from upstream (no effect on predictions)

1. **Mount paths.** A notebook saved today mounts every dataset at `/kaggle/input/datasets/<owner>/<slug>`. Upstream
   hard-codes the older short paths for two datasets (its asset dataset and `nlm-l24-adapters`), so those strings are
   rewritten (145 + 2 occurrences). Without this a fresh copy fails its input checks on the hidden rerun.
2. **A commit-time self-test** (the cell before the run cell). On the 3-row public test the notebook writes a canned
   preview and skips inference, so an edit would first run hours into the hidden rerun. The self-test checks on CPU:
   the recipe, every changed line of the run script, the run script's own recipe guards, the blend arithmetic on random
   data, that every runtime module compiles, all 160 upstream input checks, 2 GPUs and the mount paths. On any failure
   it deletes the preview submission and raises.

---
*The upstream notebook's own description follows. Only the share figures in it were updated to match this notebook.*
