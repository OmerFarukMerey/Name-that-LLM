"""The ntl627-svc-prefit cache format, shared by refit_svc.py and compare_svc.py.

Preprocessing and the loader are copied verbatim (modulo layout) from cell 10
(`reference627_common.py`) of lightningv08/0-641-lora-llama-qwen3-8-27b; the fit
options are Nishant Dahal's part-two n-gram SVC cell (min_df=2, max_features per view).

Published file formats (checked against the downloaded cache, sha256-verified):
  <view>-vocabulary.json  json.dumps(list_of_terms_in_column_order, ensure_ascii=False), UTF-8
  <view>-idf.npy          np.save(float32 idf_, shape (max_features,))
  readout.npz             np.savez_compressed(coef=float64 (18, 540000), intercept=float64 (18,),
                                              classes=int64 arange(18))
  preview.npz             np.savez_compressed(ids=<U12 (3,), direct=float64 (3, 18), portable=float64 (3, 18))
  metadata.json, COMPLETE.json   json.dumps(obj, indent=2), no trailing newline
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

# Label order of the cache (metadata.json "labels"); NOT ntl.data.LABELS (which puts claude-fable-5 at 9).
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview",
          "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-opus-4-6", "claude-opus-4-8",
          "claude-opus-5", "claude-sonnet-5", "claude-fable-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]
TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"


def shape_text(t):
    """Every run of letters becomes X or x, so only the rhythm is left."""
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_text(t):
    """Keep the small words and the punctuation, and replace every other word by CONTENT."""
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def _identity(t):
    return t


def prep_for(view):
    return shape_text if view == "shape" else small_words_text if view == "small" else _identity


def analyzer_for(view):
    return "char" if view == "shape" else "word" if view == "small" else view


def fit_options(view, lo, hi, max_features):
    """Exactly the upstream fit options (part-two n-gram SVC cell)."""
    analyzer = analyzer_for(view)
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        opts["token_pattern"] = TOKEN_PATTERN
    return opts


def fixed_vectorizer(view, lo, hi, vocabulary, idf):
    """Exactly the loader's transform-only vectorizer (cached_svc)."""
    analyzer = analyzer_for(view)
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, dtype=np.float32,
                vocabulary={t: i for i, t in enumerate(vocabulary)})
    if analyzer == "word":
        opts["token_pattern"] = TOKEN_PATTERN
    vec = TfidfVectorizer(**opts)
    vec.idf_ = np.asarray(idf)
    return vec


def cached_svc(root, texts):
    """Verbatim logic of the 0.641 notebook's loader: logits (n, 18) in LABELS order."""
    root = Path(root)
    meta = json.loads((root / "metadata.json").read_text())
    if meta["labels"] != LABELS:
        raise ValueError("Cached SVC label order differs")
    matrices = []
    for view, lo, hi, maximum, weight in meta["views"]:
        vocabulary = json.loads((root / (view + "-vocabulary.json")).read_text(encoding="utf-8"))
        prep = prep_for(view)
        vec = fixed_vectorizer(view, lo, hi, vocabulary, np.load(root / (view + "-idf.npy"), allow_pickle=False))
        matrices.append(vec.transform([prep(t) for t in texts]) * weight)
    with np.load(root / "readout.npz", allow_pickle=False) as state:
        if not np.array_equal(state["classes"], np.arange(18)):
            raise ValueError("Readout class order differs")
        logits = hstack(matrices, format="csr") @ state["coef"].T + state["intercept"]
    assert logits.shape == (len(texts), 18) and np.isfinite(logits).all()
    return logits


def load_cache(root):
    root = Path(root)
    out = {"meta": json.loads((root / "metadata.json").read_text())}
    for view, *_ in VIEWS:
        out[view + "-vocabulary"] = json.loads((root / (view + "-vocabulary.json")).read_text(encoding="utf-8"))
        out[view + "-idf"] = np.load(root / (view + "-idf.npy"), allow_pickle=False)
    with np.load(root / "readout.npz", allow_pickle=False) as z:
        out["coef"], out["intercept"], out["classes"] = z["coef"], z["intercept"], z["classes"]
    return out


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def write_view(root, view, vocabulary, idf):
    root = Path(root)
    (root / (view + "-vocabulary.json")).write_bytes(json.dumps(list(vocabulary), ensure_ascii=False).encode("utf-8"))
    np.save(root / (view + "-idf.npy"), np.asarray(idf, dtype=np.float32), allow_pickle=False)


def write_readout(root, coef, intercept):
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    assert coef.shape == (18, sum(v[3] for v in VIEWS)) and intercept.shape == (18,)
    np.savez_compressed(Path(root) / "readout.npz", coef=coef, intercept=intercept, classes=np.arange(18, dtype=np.int64))


def write_manifest(root, component):
    root = Path(root)
    files = []
    for p in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if p.is_file() and p.name not in ("publication-manifest.json", "README.md"):
            files.append(dict(path=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    write_json(root / "publication-manifest.json", dict(component=component, files=files))


def softmax_rows(d):
    return softmax(np.asarray(d, dtype=np.float64), axis=1)
