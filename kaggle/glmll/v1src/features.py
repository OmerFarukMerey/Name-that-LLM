"""Per-text likelihood features from the per-token scores of ntl-segrun.

Per scored token: lp = log-prob of the true token, H = entropy of the next-token distribution,
m2 = E_p[logp^2] (so Var_p[logp] = m2 - H^2), rank = how many tokens beat the true one (0 = top-1).
"""
import numpy as np
import pandas as pd


def one(s, n_chars):
    lp, h, m2, rank = s[:, 0].astype(np.float64), s[:, 1].astype(np.float64), s[:, 2].astype(np.float64), s[:, 3]
    n = len(lp)
    var = np.maximum(m2 - h * h, 1e-9)
    half = max(n // 2, 1)
    q = np.quantile(lp, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90]) if n else np.zeros(6)
    return {
        'n_tok': n,
        'lp_mean': lp.mean(),
        'lp_std': lp.std(),
        'lp_q05': q[0], 'lp_q10': q[1], 'lp_q25': q[2], 'lp_q50': q[3], 'lp_q75': q[4], 'lp_q90': q[5],
        'lp_min': lp.min(),
        'ent_mean': h.mean(),
        'ent_std': h.std(),
        # Fast-DetectGPT analytic curvature: how far the text's log-likelihood sits above the model's own expectation
        'fdg': (lp + h).sum() / np.sqrt(var.sum()),
        'surprise_mean': (-lp - h).mean(),          # observed surprisal minus expected surprisal
        'z_mean': ((lp + h) / np.sqrt(var)).mean(),
        'logrank_mean': np.log1p(rank).mean(),
        'top1': (rank == 0).mean(),
        'top5': (rank < 5).mean(),
        'top20': (rank < 20).mean(),
        'rank_ge100': (rank >= 100).mean(),
        'lp_mean_first': lp[:half].mean(),
        'lp_mean_second': lp[half:].mean() if n > half else lp.mean(),
        'bpc': -lp.sum() / max(n_chars, 1) / np.log(2),
        'tok_per_char': n / max(n_chars, 1),
    }


def text_features(scores, texts):
    return pd.DataFrame([one(s, len(t)) for s, t in zip(scores, texts)])
