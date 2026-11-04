"""exp032: rescore one exp029 held-out fold with that run's trained ATLAS bundle, loaded the way infer.py's atlas_scores
loads it, and keep every token's log-probability under each of the 18 label adapters, so readouts other than ATLAS's
argmax-prod can be compared offline (experiments/exp032_atlas_readout.py).

    python atlas_rescore.py --bundle DIR --base DIR --train TRAIN_CSV --out DIR

Output: DIR/holdout_tokens.npz with ids, y, labels, lens (scored tokens per row), lp (float32, tokens x 18, rows
concatenated in train order), tok (the scored token ids) and prod (the bundle's normalised argmax-prod readout, which
must match the training kernel's holdout_scores.npz).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

p = argparse.ArgumentParser()
p.add_argument('--bundle', required=True, help='atlas_out of an exp029 hold-out kernel (ntl-atlas-cv-*)')
p.add_argument('--base', required=True, help='Qwen3-8B-Base directory')
p.add_argument('--train', required=True, help='official train.csv')
p.add_argument('--out', required=True)
a = p.parse_args()
T0 = time.time()


def log(*x):
    print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)


sys.path.insert(0, a.bundle)
from ntl_atlas.model import load_model, normalized_scores, prod_readout  # noqa: E402
from ntl_rlcd.data import LABELS  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
sys.path.remove(a.bundle)
META = json.load(open(os.path.join(a.bundle, 'atlas_bundle.json')))
CFG = dict(META['training_config'])
FOLD = META['exp029']['holdout_fold']
assert FOLD >= 0, 'this bundle was trained on every row'


def prompt_folds(prompt_ids, n_folds=5, seed=42):
    """ntl.cv.prompt_folds: a fold per row, every prompt in one fold."""
    prompts = np.array(sorted(set(prompt_ids)))
    shuffled = prompts[np.random.default_rng(seed).permutation(len(prompts))]
    fold_of = {q: i % n_folds for i, q in enumerate(shuffled)}
    return np.array([fold_of[q] for q in prompt_ids])


train = pd.read_csv(a.train)
train['text'] = train.text.fillna('').astype(str)
hold = train[prompt_folds(train.prompt_id.values) == FOLD].reset_index(drop=True)
model, tok = load_model(a.base, LABELS, CFG, device='auto')
model.load_trainable(load_file(os.path.join(a.bundle, 'adapters.safetensors'))); model.eval()
log(f'fold {FOLD}: {len(hold)} rows, model loaded on {model.device}')
enc = tok(hold.text.tolist(), truncation=True, max_length=CFG['max_length'])['input_ids']
order = np.argsort([len(t) for t in enc], kind='stable')
rows, prod = [None] * len(hold), np.zeros((len(hold), len(LABELS)), np.float32)
CHUNK = 3   # infer.py's --atlas_chunk
for b in range(0, len(order), CFG['score_batch_size']):
    idx = order[b:b + CFG['score_batch_size']]
    pk = tok.pad({'input_ids': [enc[int(i)] for i in idx]}, padding=True, return_tensors='pt').to(model.device)
    ids, mask = pk['input_ids'], pk['attention_mask']
    with torch.inference_mode():   # AtlasModel.score, keeping the token log-probabilities it reads out
        parts = []
        for first in range(0, len(LABELS), CHUNK):
            n = min(len(LABELS), first + CHUNK) - first
            src = torch.arange(first, first + n, device=ids.device).repeat_interleave(len(ids))
            parts.append(model.token_logp(ids.repeat(n, 1), mask.repeat(n, 1), src).reshape(n, len(ids), -1).transpose(0, 1))
        all_lp = torch.cat(parts, dim=1)
        prod[idx] = normalized_scores(prod_readout(all_lp, mask[:, 1:], CFG['prod_temperatures'])).float().cpu().numpy()
    n_tok = mask[:, 1:].sum(1).tolist()
    for j, i in enumerate(idx):
        rows[i] = (all_lp[j, :, :n_tok[j]].float().cpu().numpy().T, enc[int(i)][1:1 + n_tok[j]])
    if b % 100 == 0:
        log(f'{b + len(idx)}/{len(order)} rows')
log(f'scored {len(hold)} rows')

ref = np.load(os.path.join(a.bundle, 'holdout_scores.npz'), allow_pickle=False)
assert ref['ids'].tolist() == hold.id.astype(str).tolist(), 'row order differs from the training kernel'
diff = float(np.abs(ref['scores'] - prod).max())
log(f'argmax-prod against the training kernel: max abs diff {diff:.2e}, argmax agreement '
    f'{float((ref["scores"].argmax(1) == prod.argmax(1)).mean()):.4f}')
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, 'holdout_tokens.npz'), ids=np.array(hold.id.astype(str).tolist(), dtype='U'),
         y=hold.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy(), labels=np.array(LABELS, dtype='U'),
         lens=np.array([len(r[1]) for r in rows], np.int32), lp=np.concatenate([r[0] for r in rows]).astype(np.float32),
         tok=np.concatenate([np.array(r[1], np.int32) for r in rows]), prod=prod)
json.dump({'holdout_fold': FOLD, 'rows': len(hold), 'tokens': int(sum(len(r[1]) for r in rows)), 'prod_max_abs_diff': diff,
           'seconds': round(time.time() - T0)}, open(os.path.join(a.out, 'rescore_meta.json'), 'w'), indent=1)
log('done')
