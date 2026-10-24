# Name-that-LLM

Kaggle code competition: 18-class authorship attribution of LLM-written passages.
Competition facts are in docs/competition.md and the layout is in README.md.

- Validate only with `ntl.cv` (grouped by `prompt_id`). Test prompts are unseen, so any
  random split leaks and inflates CV.
- Score with `ntl.metrics.balanced_accuracy`, never plain accuracy.
- Inference runs offline on Kaggle: no pip installs, no downloads. Anything the
  notebook imports beyond the Kaggle image must ship in the artifacts dataset.
- Write submissions only through `ntl.submission.write_submission`; it enforces
  Kaggle's rejection rules.
- Record every experiment in experiments/README.md and every submission in
  submissions/INDEX.md.
- Run tests: `make test` (or `python -m pytest -q`).
