"""Roman Tamrazov's dual head for the full-text Qwen3.8-27B (ntl-v28-dual, kaggle/lightning/variants_dual.py).

The checkpoint ships a trained prompt-centred head, model.centered (Linear 40960->18), which V28 loads and never calls.
Inside topic clusters of 8-18 rows (topic_groups at threshold .90) the pooled features are centred on the cluster mean
and scored by that head; dual=(1-a)*softmax(plain)+a*softmax(centered), a=.35 where both heads agree on the argmax and
.18 otherwise (romantamrazov/name-that-llm-who-am-i V22). Every other row keeps the plain logits unchanged.
numpy only: the worker passes its torch heads in as callables. Every check raises; nothing falls back.
"""
import numpy as np
from reference627_common import topic_groups

THRESHOLD, MIN_SIZE, MAX_SIZE, AGREE, DISAGREE = .90, 8, 18, .35, .18


def dual_groups(texts):
    groups = np.asarray(topic_groups(list(texts), threshold=THRESHOLD))
    sizes = np.bincount(groups)[groups]
    return groups, (sizes >= MIN_SIZE) & (sizes <= MAX_SIZE)


def summary(groups, valid):
    sizes = np.bincount(groups)
    sizes = sizes[sizes > 0]
    return dict(threshold=THRESHOLD, rows=int(len(groups)), clusters=int(len(sizes)), centered_valid_rows=int(valid.sum()),
                centered_valid_clusters=int(len(np.unique(groups[valid]))),
                cluster_size_histogram={str(s): int(n) for s, n in enumerate(np.bincount(sizes)) if n})


def centered_head_logits(logits, features, groups, valid, centered_head, plain_head, block=32, tolerance=.05):
    """Plain logits outside valid clusters; inside, the centred head on cluster-mean-centred features.

    The plain head is re-run on the valid rows first: it must reproduce the worker's logits, which proves the feature
    rows line up with the logit rows before anything is centred.
    """
    logits, groups, valid = np.asarray(logits, dtype=np.float32), np.asarray(groups), np.asarray(valid)
    n = len(logits)
    if (logits.shape != (n, 18) or features.ndim != 2 or len(features) != n or groups.shape != (n,)
            or valid.shape != (n,) or valid.dtype != bool or not np.isfinite(logits).all()):
        raise ValueError('Dual head: inconsistent logits/features/groups')
    rows = np.flatnonzero(valid)
    sizes = np.bincount(groups)
    if len(rows) and not (np.isin(np.flatnonzero(np.isin(groups, groups[rows])), rows).all()
                          and ((sizes[groups[rows]] >= MIN_SIZE) & (sizes[groups[rows]] <= MAX_SIZE)).all()):
        raise ValueError('Dual head: valid rows are not whole 8-18-row clusters')
    centred = np.asarray(features[rows], dtype=np.float32)
    if not np.isfinite(centred).all():
        raise FloatingPointError('Dual head: nonfinite features')
    worst = 0.
    for start in range(0, len(rows), block):
        recheck = np.asarray(plain_head(np.ascontiguousarray(centred[start:start + block])), dtype=np.float64)
        worst = max(worst, float(np.abs(recheck - logits[rows[start:start + block]]).max()))
    if not worst <= tolerance:
        raise FloatingPointError(f'Dual head: plain head on the saved features differs from the logits by {worst}')
    member = groups[rows]
    for group in np.unique(member):
        mask = member == group
        centred[mask] = centred[mask] - centred[mask].mean(axis=0)
    out = logits.copy()
    for start in range(0, len(rows), block):
        value = np.asarray(centered_head(np.ascontiguousarray(centred[start:start + block])), dtype=np.float32)
        if value.shape != (len(rows[start:start + block]), 18):
            raise ValueError('Dual head: centred head output shape')
        out[rows[start:start + block]] = value
    if not np.isfinite(out).all():
        raise FloatingPointError('Dual head: nonfinite centred logits')
    return out, worst


def softmax_rows(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def dual_probabilities(plain_logits, centered_logits, valid):
    plain, centered = softmax_rows(plain_logits), softmax_rows(centered_logits)
    agree = plain.argmax(1) == centered.argmax(1)
    alpha = np.where(np.asarray(valid, dtype=bool), np.where(agree, AGREE, DISAGREE), 0.)[:, None]
    return (1 - alpha) * plain + alpha * centered, agree


def dual_head_logits(plain_logits, centered_logits, valid):
    """Logits for blend_selected: the plain logits untouched outside valid clusters, log(dual) inside them
    (blend_selected re-softmaxes, and softmax(log p) = p)."""
    plain, centered, valid = (np.asarray(plain_logits, dtype=np.float64), np.asarray(centered_logits, dtype=np.float64),
                              np.asarray(valid))
    n = len(plain)
    if plain.shape != (n, 18) or centered.shape != (n, 18) or valid.shape != (n,) or not set(np.unique(valid)) <= {0, 1}:
        raise ValueError('Dual head: inconsistent plain/centred/valid arrays')
    valid = valid.astype(bool)
    if not (np.isfinite(plain).all() and np.isfinite(centered).all()) or not np.array_equal(plain[~valid], centered[~valid]):
        raise ValueError('Dual head: centred logits must equal the plain logits outside valid clusters')
    dual, agree = dual_probabilities(plain, centered, valid)
    out = plain.copy()
    out[valid] = np.log(np.maximum(dual[valid], 1e-300))
    if not np.isfinite(out).all():
        raise FloatingPointError('Dual head: nonfinite dual logits')
    return out, dict(qwen_dual_valid_rows=int(valid.sum()), qwen_dual_agree_rows=int((agree & valid).sum()))


def fixture_texts(distinct=30, copies=10, words=40):
    """Synthetic texts with disjoint vocabularies plus one planted cluster of near-duplicates (one word swapped each)."""
    texts = [' '.join(f'solo{i}w{j}' for j in range(30)) for i in range(distinct)]
    base = [f'dupe{j}w' for j in range(words)]
    planted = sorted(range(3, 3 + 3 * copies, 3))
    for k, at in enumerate(planted):
        texts.insert(at, ' '.join(base[:k] + base[k + 1:] + [f'swap{k}x']))
    return texts, planted


def smoke_test():
    """Seconds-long check of every function above; the worker runs it before the 27B model loads."""
    texts, planted = fixture_texts()
    groups, valid = dual_groups(texts)
    if np.flatnonzero(valid).tolist() != planted:
        raise AssertionError(f'Dual head smoke: planted cluster {planted} not found, got {np.flatnonzero(valid).tolist()}')
    rng = np.random.default_rng(0)
    features = (rng.normal(size=(len(texts), 16)) + 3 * valid[:, None]).astype(np.float32)
    plain_w, centred_w = rng.normal(size=(16, 18)).astype(np.float32), rng.normal(size=(16, 18)).astype(np.float32)
    logits = features @ plain_w
    centered, worst = centered_head_logits(logits, features, groups, valid, lambda x: x @ centred_w, lambda x: x @ plain_w)
    inside = features[planted].astype(np.float64)
    expected = logits.astype(np.float64).copy()
    expected[planted] = (inside - inside.mean(0)) @ centred_w
    if not (np.allclose(centered, expected, atol=1e-3) and np.array_equal(centered[~valid], logits[~valid])):
        raise AssertionError('Dual head smoke: centred logits differ from the direct computation')
    out, report = dual_head_logits(logits, centered, valid.astype(np.uint8))
    p, q = softmax_rows(logits), softmax_rows(centered)
    a = np.where(p.argmax(1) == q.argmax(1), AGREE, DISAGREE)[:, None]
    if not (np.allclose(softmax_rows(out[valid]), ((1 - a) * p + a * q)[valid], atol=1e-12)
            and np.array_equal(out[~valid], logits[~valid].astype(np.float64)) and report['qwen_dual_valid_rows'] == len(planted)):
        raise AssertionError('Dual head smoke: dual mixture differs from the direct formula')
    return dict(report, smoke_rows=len(texts), smoke_recheck=worst)
