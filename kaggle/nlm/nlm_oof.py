"""Out-of-fold log-probabilities of fold-averaged LoRA24 members (nlm-v27's q8v5, q14v5, ll8v5; Dhruv's q25_14v5) on train.csv.

    python nlm_oof.py --specs '<json list>' --out oof.npz [--hypothesis folds.npy] [--folds folds.npy] [--enc_dir DIR]

Each member ships five fold adapters (stage 2 without one fold of the 2,699 official rows) and a full fit, so a fold
adapter's predictions on its held-out fold are out of fold. The split is not published. Without --folds, the first
member (q8v5, the cheapest) scores every train row with every fold adapter, and each prompt goes to the adapter that
gives its true labels the lowest summed log-probability: the one that never saw it. With --folds (the split inferred
by an earlier run) that step is skipped. Every member then scores only the rows each adapter held out. The plain head
only (per passage, no prompt-group twin or Gaussian), as most hidden-test passages are singletons. Feature extraction
is infer.py's l24_predict: truncated base in float16, LoRA merged into the weights, masked mean pool (first token
dropped) of 8 depths, one LayerNorm each, bfloat16 redo of overflowing batches.

A member whose 8 depths are not published (spec 'search': true) gets them from its full-data adapter (spec 'full'):
every hidden state of the first 256 train rows, which that adapter was trained on, is pooled once, and the depth set
whose plain head gives those rows the highest mean true-label log-probability wins (named candidates, then coordinate
ascent one slot at a time). --enc_dir also writes each member's OOF as <name>/oof.npz (ids, labels, logits), the
format experiments/exp015_nlm_core_oof.py reads for candidates.
"""
import argparse
import contextlib
import gc
import json
import os
import time

import numpy as np
import pandas as pd
import torch

LABELS = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash',
          'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
T0 = time.time()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def log(*x):
    print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)


def lsm(z):
    return z - z.max(1, keepdims=True) - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True))


def load_base(spec):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec['path'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(spec['path']); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = spec['keep']; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:spec['keep']]
    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()
    return tok, m, dict(m.named_modules()), next(m.parameters()).device


@contextlib.contextmanager
def merged(mods, ad):
    """infer.py's in-place LoRA merge (W += B A * alpha / r) for the duration of the block; yields the adapter's heads."""
    from safetensors.torch import load_file
    adp = os.path.join(ad, 'adapter') if os.path.exists(os.path.join(ad, 'adapter', 'adapter_config.json')) else ad
    acfg = json.load(open(os.path.join(adp, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
    sd = load_file(os.path.join(adp, 'adapter_model.safetensors')); deltas = []
    with torch.no_grad():
        for k, v in sd.items():
            if '.lora_A.' not in k:
                continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods:
                name = 'model.' + name
            W = mods[name].weight; B = sd[k.replace('.lora_A.', '.lora_B.')]
            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
            deltas.append((W, B, v))
    H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu')
    if os.path.exists(os.path.join(ad, 'meta.json')):
        perm = [json.load(open(os.path.join(ad, 'meta.json')))['labels'].index(l) for l in LABELS]
        H['plain.weight'], H['plain.bias'] = H['plain.weight'][perm], H['plain.bias'][perm]
    try:
        yield H
    finally:
        with torch.no_grad():
            for W, B, A_ in deltas:
                W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
        del deltas, sd; gc.collect(); torch.cuda.empty_cache()


@torch.no_grad()
def pooled(spec, tok, m, first, texts, depths, ovf):
    """Masked mean pool (first token dropped) of the hidden states at `depths`, float32 on the CPU: {depth: (n, hid)}."""
    enc = tok(list(texts), truncation=True, max_length=spec.get('max_len', 384))['input_ids']
    order = np.argsort([len(x) for x in enc]); bs = spec.get('bs', 16); out = {}
    for i in range(0, len(order), bs):
        b = order[i:i + bs]; Lm = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
        for r, j in enumerate(b):
            ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in depths):
            m.to(torch.bfloat16); hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16); ovf[0] += 1
        mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
        for l in depths:
            out.setdefault(l, np.zeros((len(enc), hs[l].shape[-1]), np.float32))[b] = ((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)).numpy()
    return out


def head_lp(H, feats, L):
    """infer.py's plain head: one LayerNorm per depth, concatenated, linear, log-softmax."""
    hid = H['norms.0.weight'].shape[0]
    with torch.no_grad():
        Z = []
        for i, l in enumerate(L):
            nm = torch.nn.LayerNorm(hid); nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
            Z.append(nm(torch.from_numpy(feats[l])).numpy())
    return lsm(np.concatenate(Z, -1) @ H['plain.weight'].float().numpy().T + H['plain.bias'].float().numpy())


def search_layers(spec, tok, m, mods, first, texts, y, rows=256):
    """The 8 depths of a member whose depths are not published, from its full-data adapter's fit to rows it saw."""
    ovf = [0]
    with merged(mods, spec['full']) as H:
        depths = list(range(1, spec['keep'] + 1))
        F = pooled(spec, tok, m, first, texts[:rows], depths, ovf)
    yy = y[:rows]

    def fit(L):
        lp = head_lp(H, F, L)
        return float(lp[np.arange(len(yy)), yy].mean()), float((lp.argmax(1) == yy).mean())
    k = spec['keep']
    cands = {'nlm-v27 L24': [2, 4, 6, 8, 12, 16, 20, 24], 'every 4th': [4, 8, 12, 16, 20, 24, 28, 32],
             'L24 + last': [2, 4, 6, 8, 12, 16, 20, k], 'scaled to keep': sorted({max(1, round(x * k / 24)) for x in [2, 4, 6, 8, 12, 16, 20, 24]})}
    cands = {n: L for n, L in cands.items() if len(L) == 8 and max(L) <= k}
    scored = {n: fit(L) for n, L in cands.items()}
    for n, (ll, acc) in scored.items():
        log(f"  depth search {spec['name']}: {n} {cands[n]}: mean true-label log-prob {ll:.3f}, accuracy {acc:.3f}")
    best = list(cands[max(scored, key=lambda n: scored[n][0])]); cur = fit(best)[0]
    for _ in range(2):   # coordinate ascent, one slot at a time
        for i in range(8):
            for d in depths:
                if d in best:
                    continue
                L = best[:i] + [d] + best[i + 1:]
                s = fit(L)[0]
                if s > cur + 1e-6:
                    best, cur = L, s
    ll, acc = fit(best)
    log(f"  depth search {spec['name']}: chosen {best}: mean true-label log-prob {ll:.3f}, accuracy {acc:.3f} "
        f"(rows {len(yy)}, bf16 redo batches {ovf[0]})")
    return best, {'candidates': {n: [cands[n], *scored[n]] for n in cands}, 'chosen': [best, ll, acc]}


def member(spec, texts_per_adapter, loaded=None):
    """Plain-head log-probabilities of each fold adapter on its own list of texts (infer.py's l24_predict)."""
    tok, m, mods, first = loaded or load_base(spec)
    outs, ovf = [], [0]
    for ad, texts in zip(spec['adapters'], texts_per_adapter):
        if not len(texts):
            outs.append(np.zeros((0, 18), np.float32)); continue
        with merged(mods, ad) as H:
            F = pooled(spec, tok, m, first, texts, spec['layers'], ovf)
            outs.append(head_lp(H, F, spec['layers']))
        log(f"  {spec['name']} {os.path.basename(ad)}: {len(texts)} rows (bf16 redo batches {ovf[0]})")
    del m, mods; gc.collect(); torch.cuda.empty_cache()
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True); p.add_argument('--specs', required=True); p.add_argument('--out', required=True)
    p.add_argument('--hypothesis', default=''); p.add_argument('--folds', default=''); p.add_argument('--enc_dir', default='')
    a = p.parse_args()
    train = pd.read_csv(os.path.join(a.data, 'train.csv'))
    train['text'] = train.text.fillna('')
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
    texts = train.text.values
    specs = json.loads(a.specs)
    out = dict(ids=train.id.astype(str).values, labels=np.array(LABELS), y=y)
    if a.folds:
        folds = np.load(a.folds); rest = specs
        assert len(folds) == len(train) and pd.Series(folds).groupby(train.prompt_id.values).nunique().max() == 1
        log(f'folds given: sizes {np.bincount(folds, minlength=5).tolist()}')
    else:
        # 1. the first member scores every row with every fold adapter; prompts go to the adapter that never saw them
        first = specs[0]
        allk = np.stack(member(first, [texts] * 5))                       # (5, N, 18)
        truth = allk[:, np.arange(len(y)), y]                              # (5, N) true-label log-probability
        prompts, pinv = np.unique(train.prompt_id.values, return_inverse=True)
        per_prompt = np.stack([np.bincount(pinv, weights=truth[k], minlength=len(prompts)) for k in range(5)])
        pfold = per_prompt.argmin(0); folds = pfold[pinv]
        srt = np.sort(per_prompt, 0); gap = (srt[1] - srt[0]) / np.bincount(pinv)   # per-row nats between the two lowest
        log(f'inferred folds: sizes {np.bincount(folds, minlength=5).tolist()}, per-row gap min {gap.min():.2f} '
            f'median {np.median(gap):.2f} nats')
        acc_in = [(allk[k][folds != k].argmax(1) == y[folds != k]).mean() for k in range(5)]
        acc_out = [(allk[k][folds == k].argmax(1) == y[folds == k]).mean() for k in range(5)]
        log('fold adapter accuracy, rows it trained on vs held out: ' +
            ', '.join(f'f{k} {i:.3f}/{o:.3f}' for k, (i, o) in enumerate(zip(acc_in, acc_out))))
        if a.hypothesis:
            h = np.load(a.hypothesis)
            # fold labels are arbitrary: compare the partitions
            same = np.mean([[np.array_equal(folds == i, h == j) for j in range(5)] for i in range(5)], dtype=float) * 5
            log(f'GroupKFold hypothesis: {same:.0%} of inferred folds match a hypothesis fold exactly')
        out['gap'] = gap
        out[f"oof_{first['name']}"] = allk[folds, np.arange(len(y))]
        out[f"all5_{first['name']}"] = allk.astype(np.float16)
        rest = specs[1:]
    out['folds'] = folds
    # 2. the other members: each fold adapter on the rows it held out
    search = {}
    for spec in rest:
        loaded = load_base(spec)
        if spec.get('search'):
            spec['layers'], search[spec['name']] = search_layers(spec, *loaded, texts, y)
        lp = member(spec, [texts[folds == k] for k in range(5)], loaded)
        del loaded; gc.collect(); torch.cuda.empty_cache()
        oof = np.zeros((len(y), 18), np.float32)
        for k in range(5):
            oof[folds == k] = lp[k]
            log(f"  {spec['name']} fold {k}: held-out accuracy {(lp[k].argmax(1) == y[folds == k]).mean():.3f}")
        out[f"oof_{spec['name']}"] = oof
    for k_, v in out.items():
        if k_.startswith('oof_'):
            rec = [(v[y == c].argmax(1) == c).mean() for c in range(18)]
            log(f'{k_}: OOF balanced accuracy {np.mean(rec):.4f}')
            if a.enc_dir:
                d = os.path.join(a.enc_dir, k_[4:]); os.makedirs(d, exist_ok=True)
                np.savez(os.path.join(d, 'oof.npz'), ids=np.array(train.id.astype(str).tolist()), labels=np.array(LABELS),
                         logits=v.astype(np.float32))
    if search:
        json.dump(search, open(os.path.splitext(a.out)[0] + '_depths.json', 'w'), indent=1)
    np.savez(a.out, **out)
    log('NLM OOF DONE', a.out)


if __name__ == '__main__':
    main()
