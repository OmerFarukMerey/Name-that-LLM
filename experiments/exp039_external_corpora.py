"""exp039: public text from the 18 models that the open pool lacks. crtproduction's open pool already merges 55 public
Kaggle sources; a search of Kaggle (2026-11-02) found two new ones with real API passages, kilupy's 18-model
prompt-paired corpus (CC BY 4.0: 16,603 Claude passages from the Anthropic API, 3,338 prompts x 5 Claude models, + 300
prompts x the 13 other models via OpenRouter) and enaimaan's OpenAI responses (9,719 gpt-5.6-luna/sol/terra and
gpt-6-astra passages, mixed licence), and a search off Kaggle found lechmazur's debate transcripts on GitHub (683 shared
motions, 14 of the 18 models; no licence file). This scores each the way they would enter infer.py's n-gram members:
infer.py's TF-IDF logistic regression fitted on external rows only (K's 2x matched draw, then with each source added)
and scored on every official row, 18-class and within a family. exp034 / exp036's variant m2k refits the three members
with K's rows + kilupy out of fold for the replica.

    python -I experiments/exp039_external_corpora.py debate <lechmazur/debate clone> <out.parquet>
    python -I experiments/exp039_external_corpora.py glm53f <glm53-flash-harvest data/registers/prose.parquet> <out.parquet>
    python -I experiments/exp039_external_corpora.py transfer <ntl_open_pool.parquet> <nlm_matched_clean.parquet> name=path ...

`transfer` takes each extra source as name=path (a parquet with text and model; `debate` writes one), capped at
EXP039_CAP rows per label (default 1500, 0 = all); EXP039_FAMILY=claude (or gpt, ...) scores within that family only.
-I because every input is a download; the repo's src/ and experiments/ are put on the path below.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "experiments")]
import glob  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

# lechmazur/debate's model names -> competition labels (Fable 5.1, Opus 5.5 and Kimi K3.1 are other models); per its
# README "gpt-5.6-high" is GPT-5.6 Sol
DEBATE_LABELS = {"claude-fable-5-high": "claude-fable-5", "claude-opus-4-6-0K": "claude-opus-4-6",
                 "claude-opus-4-6-adaptive": "claude-opus-4-6", "claude-opus-4-8-adaptive": "claude-opus-4-8",
                 "claude-opus-5-high": "claude-opus-5", "claude-sonnet-5-high": "claude-sonnet-5",
                 "gemini-3.1-pro-preview": "gemini-3.1-pro-preview", "gemini-3.7-flash-high": "gemini-3.7-flash",
                 "gemini-3.8-flash-high": "gemini-3.8-flash", "glm-5-3-high": "glm-5.3", "gpt-5.5-high": "gpt-5.5",
                 "gpt-5.6-high": "gpt-5.6-sol", "gpt-6-astra-high": "gpt-6-astra", "kimi-k3": "kimi-k3", "qwen3.8-max": "qwen3.8-max"}
CLAIM_TAG = re.compile(r"\s*\[[PC]\d+(?:\.\d+)?(?:\s*[-–,/;]\s*(?:and\s+)?[PC]?\d+(?:\.\d+)?)*\]\s*")
SENT_END = re.compile(r"(?<=[.!?…”\"])\s+")


def debate(clone, out):
    """Opening, rebuttal and closing turns (not the question turns), claim tags removed, cut at sentence ends into
    passages of ~120 words (the official passages: median 101, 36-274)."""
    rows = []
    for f in sorted(glob.glob(os.path.join(clone, "transcripts", "*.md"))):
        for sec in re.split(r"\n## Turn \d+: ", open(f, encoding="utf-8").read())[1:]:
            head, _, body = sec.partition("\n")
            m = re.search(r"^- model: (\S+)", body, re.M)
            if "Questions" in head or not m or m.group(1) not in DEBATE_LABELS:
                continue
            text = re.sub(r"\s+", " ", " ".join(CLAIM_TAG.sub(" ", l).strip() for l in body.split("\n")
                                                if l.strip() and not l.startswith("- "))).strip()
            chunk, n = [], 0
            for s in SENT_END.split(text):
                chunk.append(s); n += len(s.split())
                if n >= 120:
                    rows.append((" ".join(chunk), DEBATE_LABELS[m.group(1)])); chunk, n = [], 0
            if n >= 40:
                rows.append((" ".join(chunk), DEBATE_LABELS[m.group(1)]))
    d = pd.DataFrame(rows, columns=["text", "model"]).drop_duplicates("text")
    d.to_parquet(out)
    print(len(d), "passages;", d.model.value_counts().to_dict())


def glm53f(prose, out):
    """Zek-Takai/glm53-flash-harvest's prose register (Hugging Face, MIT: 15,923 GLM-5.3-Flash stories, monologues and
    letters from the reference FP8 weights via vLLM, temperatures 0.6-1.05): the answer after any </think>, one passage
    per paragraph of 40-250 words that ends a sentence (the official passages are single paragraphs; capped answers end
    mid-sentence), headings and [stage directions] dropped."""
    d = pd.read_parquet(prose, columns=["response"])
    rows = []
    for r in d.response.astype(str):
        for p in re.split(r"\n\s*\n", r.split("</think>")[-1]):
            p = re.sub(r"\s+", " ", p).strip()
            if p[:1] not in ("#", "[", "*", "-", ">") and 40 <= len(p.split()) <= 250 and p[-1:] in ".!?”\"’'…":
                rows.append((p, "glm-5.3-flash"))
    d = pd.DataFrame(rows, columns=["text", "model"]).drop_duplicates("text")
    d.to_parquet(out)
    print(len(d), "passages; words median", d.text.str.split().str.len().median())


def fit(args):
    """infer.py's TF-IDF member on external rows only (balanced classes: a test of the rows, not of the priors)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    name, d, texts = args
    vs = [TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=60000, sublinear_tf=True, lowercase=False),
          TfidfVectorizer(token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True,
                          lowercase=False)]
    Xa = sp.hstack([v.fit_transform(d.text) for v in vs]).tocsr()
    Xt = sp.hstack([v.transform(texts) for v in vs]).tocsr()
    return name, len(d), LogisticRegression(C=20, max_iter=3000, class_weight="balanced").fit(Xa, d.model).predict(Xt)


def transfer(pool_path, mc_path, sources):
    sys.argv.append("--plain")
    import exp034_matched_extra as X
    from ntl.data import load_train
    from ntl.metrics import balanced_accuracy
    train = load_train()
    fam = os.environ.get("EXP039_FAMILY", "")
    labels = sorted(m for m in train.model.unique() if m.startswith(fam))
    cap = int(os.environ.get("EXP039_CAP", "1500"))
    base = X.extra_rows("m2", pool_path, mc_path, train.text.astype(str).to_numpy())[["text", "model"]]
    extra = {}
    for s in sources:
        name, path = s.split("=", 1)
        d = pd.read_parquet(path, columns=["text", "model"])
        extra[name] = d.sample(frac=1.0, random_state=0).groupby("model").head(cap) if cap else d
    jobs = [("K's 2x draw", base)] + [(f"+ {n}", pd.concat([base, d])) for n, d in extra.items()]
    if len(extra) > 1:
        jobs.append(("+ " + " + ".join(extra), pd.concat([base, *extra.values()])))
    test = train[train.model.isin(labels)]
    jobs = [(n, d[d.model.isin(labels)], test.text.astype(str).tolist()) for n, d in jobs]
    fams = {f: [m for m in labels if m.startswith(f)] for f in ("claude", "gemini", "gpt", "glm")}
    with ProcessPoolExecutor(max_workers=2) as pe:
        for name, n, p in pe.map(fit, jobs):
            rec = pd.Series(p == test.model.values).groupby(test.model.values).mean()
            per = "  ".join(f"{f} {rec[ms].mean():.3f}" for f, ms in fams.items() if ms)
            print(f"{name:24s} {n:6d} rows: balanced acc {balanced_accuracy(test.model, p):.4f} | mean recall {per}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "debate":
        debate(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "glm53f":
        glm53f(sys.argv[2], sys.argv[3])
    else:
        transfer(sys.argv[2], sys.argv[3], sys.argv[4:])
