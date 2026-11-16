"""Our likelihood features for the GLM stage, shared by experiments/exp050_glm_ll_head.py (train, offline) and the
0.694 forks' stage (build_lv694.py inlines this file into the stage cell), so train and test compute the same numbers.

Per-token score arrays are (n, 4): log p of the true token, entropy, E_p[log p ^ 2], rank (tokens above the true one),
as ntl-segrun writes them for the GLM models and score_texts() for a Hugging Face reference model.
"""
import numpy as np


def one(s, n_chars):
    """Our transcription of v1src/features.py's one() (estopadilla, Apache 2.0): the same per-text features."""
    lp, h, m2, rank = (s[:, i].astype(np.float64) for i in range(4))
    n = len(lp)
    var = np.maximum(m2 - h * h, 1e-9)
    half = max(n // 2, 1)
    q = np.quantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90])
    r = {"n_tok": n, "lp_mean": lp.mean(), "lp_std": lp.std(), "lp_min": lp.min(),
         "ent_mean": h.mean(), "ent_std": h.std(),
         "fdg": (lp + h).sum() / np.sqrt(var.sum()), "surprise_mean": (-lp - h).mean(),
         "z_mean": ((lp + h) / np.sqrt(var)).mean(), "logrank_mean": np.log1p(rank).mean(),
         "top1": (rank == 0).mean(), "top5": (rank < 5).mean(), "top20": (rank < 20).mean(),
         "rank_ge100": (rank >= 100).mean(), "lp_mean_first": lp[:half].mean(),
         "lp_mean_second": lp[half:].mean() if n > half else lp.mean(),
         "bpc": -lp.sum() / max(n_chars, 1) / np.log(2), "tok_per_char": n / max(n_chars, 1)}
    r.update({f"lp_q{p:02d}": v for p, v in zip((5, 10, 25, 50, 75, 90), q)})
    return r


def pair_features(sf, sg, cut):
    """Per-token GLM-5.3-Flash - GLM-5.3 contrasts on their shared tokens (the stage's d: features only difference
    the two models' per-text means)."""
    out = []
    for a, b in zip(sf, sg):
        n = min(len(a), len(b), cut)
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        d, dh = a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]
        q = np.quantile(d, [0.10, 0.25, 0.50, 0.75, 0.90])
        out.append({"pd_std": d.std(), **{f"pd_q{p}": v for p, v in zip((10, 25, 50, 75, 90), q)},
                    "pd_pos": (d > 0).mean(), "pd_abs": np.abs(d).mean(), "pdh_mean": dh.mean(), "pdh_std": dh.std(),
                    "pd_rank_win": (a[:, 3] < b[:, 3]).mean(), "pd_rank_loss": (a[:, 3] > b[:, 3]).mean()})
    return out


def score_texts(path, texts, batch=8, cut=160):
    """Per text an (n, 4) float32 array from a Hugging Face causal LM (fp16, both T4s): log p(true token), entropy,
    E_p[log p ^ 2], rank, for the first `cut` tokens after an <|endoftext|> context token (= ntl-refll's train run)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto").eval()
    bos = tok.convert_tokens_to_ids("<|endoftext|>")
    enc = [[bos] + tok(t, add_special_tokens=False).input_ids[:cut] for t in texts]
    dev = model.get_input_embeddings().weight.device
    out = [None] * len(enc)
    order = np.argsort([len(e) for e in enc], kind="stable")
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            L = max(len(enc[i]) for i in idx)
            x = torch.full((len(idx), L), bos, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(enc[i])] = torch.tensor(enc[i])
                att[r, :len(enc[i])] = 1
            logits = model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits[:, :-1].float()
            lpa = torch.log_softmax(logits, -1)
            del logits
            lp = lpa.gather(-1, x[:, 1:].to(lpa.device)[..., None])[..., 0]
            p = lpa.exp()
            h = -(p * lpa).sum(-1)
            m2 = (p * lpa * lpa).sum(-1)
            rank = (lpa > lp[..., None]).sum(-1).float()
            del p, lpa
            s = torch.stack([lp, h, m2, rank], -1).cpu().numpy().astype(np.float32)
            for r, i in enumerate(idx):
                out[i] = s[r, :len(enc[i]) - 1]
    return out


if __name__ == "__main__":
    # python extra_features.py ref <model dir> <csv with id,text> <out.npz>   (run by the stage in a subprocess)
    import sys
    import pandas as pd
    _, mode, path, csv, dest = sys.argv
    assert mode == "ref"
    df = pd.read_csv(csv, dtype={"id": str})
    sc = score_texts(path, df.text.fillna("").astype(str).tolist())
    np.savez(dest, ids=np.array(df.id.tolist(), dtype="U64"), offsets=np.concatenate([[0], np.cumsum([len(s) for s in sc])]),
             scores=np.concatenate(sc))


def gbm_export(clf, digits=7):
    """A fitted sklearn HistGradientBoostingClassifier as compact lists: per tree the split feature, threshold and
    children of each node (children 0 = leaf) and the leaf values; gbm_logp() evaluates them with numpy. Thresholds
    keep full precision (they sit halfway between neighbouring train values); leaf values keep `digits`."""
    r = lambda x: float(f"{x:.{digits}g}")
    trees = []
    for it in clf._predictors:
        for t in it:
            n = t.nodes
            leaf = n["is_leaf"].astype(bool)
            trees.append({"f": [0 if q else int(v) for q, v in zip(leaf, n["feature_idx"])],
                          "t": [0 if q else float(v) for q, v in zip(leaf, n["num_threshold"])],
                          "l": [0 if q else int(v) for q, v in zip(leaf, n["left"])],
                          "r": [0 if q else int(v) for q, v in zip(leaf, n["right"])],
                          "v": [r(v) if q else 0 for q, v in zip(leaf, n["value"])]})
    return {"baseline": [r(v) for v in np.ravel(clf._baseline_prediction)], "n_classes": len(clf.classes_),
            "trees": trees}


def gbm_logp(g, X):
    """Log-probabilities of an exported multiclass HistGradientBoostingClassifier (trees in iteration-major order, one
    per class; x <= threshold goes left; features are finite)."""
    k = g["n_classes"]
    raw = np.tile(np.asarray(g["baseline"], np.float64), (len(X), 1))
    rows = np.arange(len(X))
    for j, t in enumerate(g["trees"]):
        f, th, lc, rc = (np.asarray(t[c]) for c in ("f", "t", "l", "r"))
        leaf, val = lc == 0, np.asarray(t["v"], np.float64)   # the root is node 0, so no child is 0
        node = np.zeros(len(X), dtype=np.int64)
        while not leaf[node].all():
            go = ~leaf[node]
            left = X[rows, f[node]] <= th[node]
            node = np.where(go, np.where(left, lc[node], rc[node]), node)
        raw[:, j % k] += val[node]
    raw -= raw.max(1, keepdims=True)
    return raw - np.log(np.exp(raw).sum(1, keepdims=True))
