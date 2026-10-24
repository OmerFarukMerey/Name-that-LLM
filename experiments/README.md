# Experiments

One script per idea, named `expNNN_<slug>.py` and started from [`template.py`](template.py).
Every experiment uses the same 5-fold prompt-grouped split (`ntl.cv`, seed 42), so CV
numbers in this table are directly comparable. Outputs go to `outputs/expNNN_<slug>/`
(gitignored): `oof.parquet` holds out-of-fold probabilities for later ensembling.

| Exp | Idea | CV balanced acc. | Public LB | Notes |
|---|---|---:|---:|---|
