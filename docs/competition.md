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

- 5 submissions/day; up to 2 final submissions selected for judging.
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

- Split by `prompt_id` (`ntl.cv`). A random split leaks a prompt's sibling answers
  across folds and overstates CV.
- Train classes are balanced, so argmax of the predicted probabilities targets balanced
  accuracy directly. Do not re-weight toward any guessed test class mix.
