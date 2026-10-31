# Competition notes

Snapshot of the [competition pages](https://www.kaggle.com/competitions/name-that-llm)
taken 2026-10-25, plus measured facts about the training data. Re-check the
Kaggle pages before relying on anything here near the deadline.

## Task

Given one self-contained passage of 40–250 words, predict which of 18 frontier LLMs
wrote it. Test passages answer **prompts that do not appear in train**, so topic
words cannot be memorised; the signal is style: vocabulary, phrasing, punctuation.

| Provider | Labels |
|---|---|
| OpenAI | `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash` |
| Anthropic | `claude-fable-5`, `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5` |
| Qwen | `qwen3.8-max` |
| ZAI | `glm-5.3`, `glm-5.3-flash` |
| Moonshot AI | `kimi-k3` |

## Data

- `train.csv`: `id, prompt_id, text, model`, with **2,699** passages. There are **150 prompts**, each
  answered once by every model. The only gap is `p002`, which has no `claude-fable-5`
  answer, so classes are balanced at 150 (149 for `claude-fable-5`).
- Passage length: 36–274 words, median 101. Mean length per model ranges only from
  ~101 (`gemini-3.8-flash`) to ~124 (`claude-opus-4-8`), so length alone is weak.
- `test.csv`: `id, text`. It has 3 example rows locally; the hidden rerun has **~4,000** passages.
- Typographic characters such as `’` survive in the text. Only light cleanup was applied
  (for example, a `**bold**` wrapper around a whole passage was removed).

## Metric

**Balanced accuracy**: the mean of per-class recall over the 18 models
(`ntl.metrics.balanced_accuracy`). The number of test passages per model is
**unequal**, and the public and private leaderboards use **different model mixes**.
Every class counts equally regardless of how many passages it has.

## Submission

- Code competition: notebook only, **internet off**, CPU or GPU, **≤ 9 h** runtime.
- Read `/kaggle/input/competitions/name-that-llm/test.csv`, write `submission.parquet`
  (columns `id`, `model`) to the working directory.
- Kaggle rejects a file that is missing a column, has nulls, repeats an id, or uses a label
  outside the 18 above. `ntl.submission.write_submission` checks all of these before writing.

## Rules

- 5 submissions/day (reset 00:00 UTC); up to 2 final submissions selected for judging.
- The hidden-test rerun of a submission does not use your weekly GPU quota: with two reruns
  in progress the settings page showed 00:00 / 45 h used (2026-10-25). Only the commit run
  (3-row test) does. A ~4,000-row rerun of the 0.639 stack took about 4 h on 2026-10-25.
- Team size ≤ 5. No private code sharing outside the team, so keep the
  artifacts dataset and inference notebook **private**.
- External data and pre-trained models are allowed if they are freely and publicly
  available. Open-source code must use an OSI-approved license. The winner license is CC-BY 4.0.

## Timeline

- Start: 2026-10-14
- Final submission deadline: **2027-01-13 23:59 UTC** (Kaggle's API lists 2027-01-14 23:59).
  The entry and team-merger deadlines were unset placeholders on the page.

## Prizes

$50,000 total: $5,000 each for places 1–10.

## Validation implications

- Split by `prompt_id` (`ntl.cv`). Others report that a random split scores far lower
  (0.217 vs 0.369 grouped, LB 0.355): a validation passage's topic words then point
  toward the 17 other models that answered the same prompt.
- Train classes are balanced, so argmax of the predicted probabilities targets balanced
  accuracy directly. Do not re-weight toward any guessed test class mix.

## Measured locally / learned from the forum (2026-10-25)

- exp001 char+word TF-IDF + LR: grouped CV **0.350**. The forum's equivalent scored public LB 0.353.
- **Test-time prompt grouping does not transfer.** Clustering passages by topic
  (TF-IDF, ARI 0.94 against true prompts) plus Hungarian assignment so each model
  appears once per cluster gave +0.10 on train OOF (0.350 → 0.450). Two teams report it
  flat or negative on the public LB (0.353 → 0.344, 0.427 → 0.425). The hidden test
  appears to have only a few answers per prompt. Don't spend submissions on it.
- The best public notebook is 0.642 (lightningv08, 2026-10-25). It is the 0.639 stack, with the n-gram SVC pre-fitted
  (cached in its asset dataset), plus a full-text Qwen3.8-27B NF4 classifier. Version 25 (Qwen **5%**) scored 0.641 and
  version 28 (Qwen **10%**, otherwise identical) 0.642; V26/V27 (3%, 7%) show no score. `kernels pull` returns the latest
  version, so pin the scored one when forking. An identical-code copy of V25 on a different Docker image lists 0.639,
  so environment alone can move a score by about 0.002. Our best is 0.640 (INDEX #5 and #6); the SVC-tiers forks of V25/V28 scored 0.640/0.639. The public leaders are two-stage LoRA LLMs (Qwen3/3.5 8–14B). They train
  first on public pools of generated passages (35k → 120k rows), then on train. Encoders,
  n-gram SVMs and a complementary stack are blended in. More, cleaner, all-18-label
  data is the biggest lever reported.
- Pool-trained members show grouped CV 0.03–0.04 **above** public LB. Compare them with
  each other, not with train-only members.
- **Fork the scored version, with its Docker image.** A public notebook's listed score is its
  best version's, not necessarily the current one. maverickss26's v2 lists 0.639 (its v4); our copy
  of its v5, which has a different Docker image and a fail-safe fallback, scored 0.620 on the
  hidden rerun. That is exactly its part-two anchor alone, so the fusion stages failed without an
  error. Prefer fail-closed originals: a broken run then errors instead of scoring quietly lower.
- Light Sinkhorn balancing (1–3 iterations) helped the public LB by about +0.01 for single
  LoRA members, even with flat CV. Full balancing is risky if the test mix is uneven.
