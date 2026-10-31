# Name-that-LLM

Solution repo for the Kaggle [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)
competition. The task is to identify which of 18 frontier LLMs wrote each short passage.
Scoring is balanced accuracy, the passages answer prompts not seen in training, and
submissions run in an offline Kaggle notebook. Task, metric, rules and data facts are in
[docs/competition.md](docs/competition.md).

## Layout

```
src/ntl/                  shared package, used locally and copied into the Kaggle dataset
  paths.py                data/output locations (local vs /kaggle/input)
  data.py                 load_train / load_test, the 18 LABELS
  cv.py                   5-fold split grouped by prompt_id (the only valid CV here)
  metrics.py              balanced accuracy + per-class recall
  submission.py           write and validate submission.parquet
experiments/              expNNN_<slug>.py scripts + results table (README.md)
  template.py             starting point: CV loop, OOF probabilities, metrics, model save
notebooks/                EDA and scratch analysis
kaggle/
  dataset/                metadata for the private artifacts dataset (ntl + model files)
  inference/              the submission notebook + kernel-metadata.json
scripts/
  build_kaggle_dataset.py stage build/kaggle-dataset/ for upload
submissions/INDEX.md      log of every submission: CV, public LB, final picks
docs/competition.md       competition facts and validation notes
docs/public_notebooks.md  the public notebooks at 0.635+, what each adds, with credits
tests/                    metric, CV, and submission-format checks
data/raw/                 (gitignored) competition files
outputs/                  (gitignored) per-experiment OOF, metrics, model weights
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python >= 3.10
pip install -e ".[dev]"
make data    # needs ~/.kaggle credentials and accepted competition rules
make test
```

## Workflow

1. **Experiment.** Copy `experiments/template.py` to `experiments/expNNN_<slug>.py`, then implement
   `fit`, `predict_proba` and `save`. Run it and add the CV score to
   [experiments/README.md](experiments/README.md).
2. **Package.** Run `make dataset EXP=expNNN_<slug>` to stage `ntl` and that model. The first upload
   uses `make dataset-create`; later uploads use `make dataset-version EXP=... MSG="..."`.
3. **Inference.** Point `EXPERIMENT` / `predict_proba` in
   `kaggle/inference/inference.ipynb` at the model, then run `make kernel` to push it.
4. **Submit.** Run `make submit V=<kernel version> MSG="..."`, or use the Submit button on Kaggle.
   Then add a row to [submissions/INDEX.md](submissions/INDEX.md).

## Results

| Submission | Experiment | CV | Public LB | Private LB |
|---|---|---:|---:|---:|
