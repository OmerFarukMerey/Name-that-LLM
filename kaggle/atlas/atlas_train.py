"""exp029: train an ATLAS adapter set on Qwen3-8B-Base with Joseph Adamski's ntl_atlas implementation of ATLAS
(arXiv 2610.19583; josephadamski91/ntl-atlas-qwen3-8b-v1, whose bundle holds the model code and the training config),
on the official train and optionally on external real-model passages, optionally holding out one prompt fold of the
official train to score afterwards.

    python atlas_train.py --bundle DIR --base DIR --train TRAIN_CSV --out DIR
        [--extra PARQUET --extra_per_label N] [--holdout K] [--epochs 5] [--seed 17] [--max_hours 9]

The bundle's own training code is not published, so the loop follows its training_config: rank 16 / alpha 32 /
dropout 0.05 adapters on q, k, v and o of the frozen fp16 base, AdamW (lr 1e-4, weight decay 0.01), 4-row
micro-batches accumulated to 128-row steps, 5% linear warmup then cosine decay, gradient clip 1.0, max length 512, plus
loss scaling for the fp16 base. Each epoch shows every official row once and 1/epochs of the external rows, shuffled,
each 128-row step cut into length-sorted micro-batches. The output directory is a bundle in the original layout
(atlas_bundle.json, adapters.safetensors, ntl_atlas/, ntl_rlcd/), so infer.py's atlas_scores loads it unchanged.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

p = argparse.ArgumentParser()
p.add_argument('--bundle', required=True, help="josephadamski91/ntl-atlas-qwen3-8b-v1 (code and training_config)")
p.add_argument('--base', required=True, help='Qwen3-8B-Base directory')
p.add_argument('--train', required=True, help='official train.csv')
p.add_argument('--out', required=True)
p.add_argument('--extra', default='', help='parquet with text/model columns (crtproduction open pool layout)')
p.add_argument('--extra_per_label', type=int, default=0)
p.add_argument('--holdout', type=int, default=-1, help='prompt fold (ntl.cv: 5 folds, seed 42) left out and scored')
p.add_argument('--epochs', type=int, default=5)
p.add_argument('--seed', type=int, default=17)
p.add_argument('--max_hours', type=float, default=9.0, help='training stops at the first step boundary after this')
p.add_argument('--reserve_gib', type=float, default=4.5, help='GPU memory left free by the weight placement')
p.add_argument('--limit', type=int, default=0, help='smoke test: use only this many official rows')
p.add_argument('--effective_batch', type=int, default=0, help='smoke test: override the 128-row step')
p.add_argument('--lr', type=float, default=0.0, help='smoke test: override the learning rate')
a = p.parse_args()
T0 = time.time()


def log(*x):
    print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)


sys.path.insert(0, a.bundle)
from ntl_atlas.model import AtlasModel  # noqa: E402
from ntl_rlcd.data import LABELS  # noqa: E402
sys.path.remove(a.bundle)
BUNDLE_META = json.load(open(os.path.join(a.bundle, 'atlas_bundle.json')))
CFG = dict(BUNDLE_META['training_config'])
assert list(BUNDLE_META['labels']) == list(LABELS), 'bundle label order differs from ntl_rlcd.data.LABELS'
L2I = {l: i for i, l in enumerate(LABELS)}
CUDA = torch.cuda.is_available()
torch.manual_seed(a.seed); np.random.seed(a.seed)
rng = np.random.default_rng(a.seed)


def prompt_folds(prompt_ids, n_folds=5, seed=42):
    """ntl.cv.prompt_folds: a fold per row, every prompt in one fold."""
    prompts = np.array(sorted(set(prompt_ids)))
    shuffled = prompts[np.random.default_rng(seed).permutation(len(prompts))]
    fold_of = {q: i % n_folds for i, q in enumerate(shuffled)}
    return np.array([fold_of[q] for q in prompt_ids])


# ---------------- data
train = pd.read_csv(a.train)
train['text'] = train.text.fillna('').astype(str)
folds = prompt_folds(train.prompt_id.values)
fit = train[folds != a.holdout] if a.holdout >= 0 else train
hold = train[folds == a.holdout].reset_index(drop=True) if a.holdout >= 0 else train.iloc[:0]
if a.limit:
    fit = fit.iloc[:a.limit]
    hold = hold.iloc[:max(18, a.limit // 4)]
fit = fit.reset_index(drop=True)
log(f'official fit rows {len(fit)} ({fit.prompt_id.nunique()} prompts), held-out rows {len(hold)} (fold {a.holdout})')

extra = pd.DataFrame(columns=['id', 'text', 'model'])
if a.extra and a.extra_per_label > 0:
    E = pd.read_parquet(a.extra)
    E = E[E.model.isin(LABELS)].copy()
    E['text'] = E.text.fillna('').astype(str)
    if 'flag_gemini37_routing_risk' in E:   # gemini-3.7 requests that the API may have answered with 3.8
        E = E[~E.flag_gemini37_routing_risk.astype(bool)]
    nw = E.text.str.split().str.len()
    E = E[nw.between(30, 300)]
    E = E[~E.text.str.strip().isin(set(train.text.str.strip()))]
    E = E.drop_duplicates(subset=['text'])
    parts = []
    for lab in LABELS:
        g = E[E.model == lab]
        take = min(a.extra_per_label, len(g))
        parts.append(g.iloc[np.sort(rng.choice(len(g), take, replace=False))])
    extra = pd.concat(parts, ignore_index=True)[['id', 'text', 'model']] if 'id' in E else \
        pd.concat(parts, ignore_index=True).assign(id=lambda d_: [f'x{i}' for i in range(len(d_))])[['id', 'text', 'model']]
    log('extra rows per label', extra.model.value_counts().reindex(LABELS).fillna(0).astype(int).to_dict())


# ---------------- model (as ntl_atlas.model.load_model, with a configurable memory reserve for training)
def load(training):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if CUDA:
        mm = {i: int(torch.cuda.get_device_properties(i).total_memory - a.reserve_gib * 2 ** 30)
              for i in range(torch.cuda.device_count())}
        base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16, attn_implementation='sdpa',
                                                    device_map='auto', max_memory=mm)
        if any(q.device.type != 'cuda' for q in base.parameters()):
            raise RuntimeError('the 16-bit base does not fit the GPUs with this reserve')
    else:
        base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32)
    model = AtlasModel(base, LABELS, CFG)
    if training:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    return model, tok


t = time.time()
model, tok = load(training=True)
DEV = model.base.get_input_embeddings().weight.device
params = [q for q in model.parameters() if q.requires_grad]
log(f'model loaded in {time.time() - t:.0f}s on {DEV}; trainable tensors {len(params)}, '
    f'{sum(q.numel() for q in params) / 1e6:.1f}M parameters')


def encode(texts):
    return tok(list(texts), truncation=True, max_length=CFG['max_length'])['input_ids']


fit_ids, fit_y = encode(fit.text), fit.model.map(L2I).to_numpy()
ext_ids, ext_y = (encode(extra.text), extra.model.map(L2I).to_numpy()) if len(extra) else ([], np.zeros(0, int))
ext_chunks = np.array_split(rng.permutation(len(ext_ids)), a.epochs) if len(ext_ids) else [np.zeros(0, int)] * a.epochs
EB, MB = a.effective_batch or CFG['effective_batch_size'], CFG['micro_batch_size']
steps = []   # each step: list of micro-batches, each a list of (token ids, label)
for ep in range(a.epochs):
    rows = [(fit_ids[i], int(fit_y[i])) for i in range(len(fit_ids))] + [(ext_ids[i], int(ext_y[i])) for i in ext_chunks[ep]]
    order = rng.permutation(len(rows))
    for s in range(0, len(order), EB):
        chunk = sorted((rows[i] for i in order[s:s + EB]), key=lambda r: len(r[0]))
        steps.append([chunk[m:m + MB] for m in range(0, len(chunk), MB)])
n_steps = len(steps)
n_warm = max(1, math.ceil(CFG['warmup_ratio'] * n_steps))
n_tok = sum(len(r[0]) for st in steps for mb in st for r in mb)
log(f'{n_steps} optimizer steps over {a.epochs} epochs, warmup {n_warm}, {n_tok / 1e6:.2f}M tokens')

opt = torch.optim.AdamW(params, lr=a.lr or CFG['learning_rate'], weight_decay=CFG['weight_decay'])
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s_: (s_ + 1) / n_warm if s_ < n_warm else 0.5 * (1 + math.cos(math.pi * (s_ - n_warm) / max(1, n_steps - n_warm))))
scaler = torch.amp.GradScaler('cuda', init_scale=2.0 ** 14) if CUDA else None


def batch(mb):
    width = max(len(r[0]) for r in mb)
    ids = torch.full((len(mb), width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(mb), width), dtype=torch.long)
    for i, (r, _) in enumerate(mb):
        ids[i, :len(r)] = torch.tensor(r); mask[i, :len(r)] = 1
    return ids.to(DEV), mask.to(DEV), torch.tensor([y_ for _, y_ in mb], dtype=torch.long, device=DEV)


def save(out_dir, done_steps, note):
    os.makedirs(out_dir, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.trainable_state().items()}, os.path.join(out_dir, 'adapters.safetensors'))
    for d in ('ntl_atlas', 'ntl_rlcd'):
        shutil.copytree(os.path.join(a.bundle, d), os.path.join(out_dir, d), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__'))
    meta = json.loads(json.dumps(BUNDLE_META))
    meta['training_config'] = dict(CFG, seed=a.seed, epochs=a.epochs)
    meta['completed_steps'] = done_steps
    meta['exp029'] = {'note': note, 'official_rows': len(fit), 'holdout_fold': a.holdout, 'extra_rows': len(extra),
                      'extra_per_label': a.extra_per_label, 'extra_source': os.path.basename(a.extra) if a.extra else '',
                      'steps_planned': n_steps, 'tokens_planned': n_tok, 'derived_from': 'josephadamski91/ntl-atlas-qwen3-8b-v1 (code, config)'}
    meta['files'] = {f: hashlib.sha256(open(os.path.join(out_dir, f), 'rb').read()).hexdigest()
                     for f in ['adapters.safetensors', 'ntl_atlas/model.py', 'ntl_atlas/__init__.py', 'ntl_rlcd/data.py', 'ntl_rlcd/__init__.py']
                     if os.path.exists(os.path.join(out_dir, f))}
    json.dump(meta, open(os.path.join(out_dir, 'atlas_bundle.json'), 'w'), indent=1)
    if len(extra):
        extra[['id', 'model']].to_csv(os.path.join(out_dir, 'extra_rows_used.csv'), index=False)


# ---------------- training
t1, seen, losses, skipped = time.time(), 0, [], 0
done = 0
for k, st in enumerate(steps):
    if time.time() - T0 > a.max_hours * 3600:
        log(f'time box reached after {k}/{n_steps} steps'); break
    opt.zero_grad(set_to_none=True)
    n_rows = sum(len(mb) for mb in st); acc = 0.0
    for mb in st:
        ids, mask, y_ = batch(mb)
        loss = model.loss(ids, mask, y_) * (len(mb) / n_rows)
        (scaler.scale(loss) if scaler else loss).backward()
        acc += float(loss.detach()); seen += int(mask.sum())
    if scaler:
        scaler.unscale_(opt)
    gn = float(torch.nn.utils.clip_grad_norm_(params, CFG['grad_clip']))
    if scaler:
        before = scaler.get_scale(); scaler.step(opt); scaler.update(); skipped += scaler.get_scale() < before
    else:
        opt.step()
    sched.step(); losses.append(acc); done = k + 1
    if k % 5 == 0 or k == n_steps - 1:
        rate = seen / max(time.time() - t1, 1e-6)
        log(f'step {k + 1}/{n_steps} loss {np.mean(losses[-5:]):.4f} grad {gn:.3f} lr {sched.get_last_lr()[0]:.2e} '
            f'{rate:.0f} tok/s eta {(n_tok - seen) / max(rate, 1e-6) / 3600:.2f} h skipped {skipped}'
            + (f' scale {scaler.get_scale():.0f}' if scaler else ''))
    if CUDA and (k + 1) % max(1, n_steps // a.epochs) == 0:   # epoch checkpoint
        save(a.out, done, 'epoch checkpoint')
save(a.out, done, f'trained {done}/{n_steps} steps')
log(f'saved {a.out} after {done}/{n_steps} steps; mean loss of the last 10 steps {np.mean(losses[-10:]):.4f}')

# ---------------- held-out fold
if len(hold):
    model.eval()
    h_ids = encode(hold.text); order = np.argsort([len(x) for x in h_ids], kind='stable')
    out = np.zeros((len(hold), len(LABELS)), np.float32); t2 = time.time()
    for s in range(0, len(order), CFG['score_batch_size']):
        idx = order[s:s + CFG['score_batch_size']]
        ids, mask, _ = batch([(h_ids[i], 0) for i in idx])
        with torch.inference_mode():
            pr, _ = model.score(ids, mask, source_chunk=3)
        out[idx] = pr.float().cpu().numpy()
    yt = hold.model.map(L2I).to_numpy(); pred = out.argmax(1)
    rec = [float((pred[yt == c] == c).mean()) for c in range(len(LABELS)) if (yt == c).any()]
    srt = np.sort(out, 1)
    R = {'holdout_fold': a.holdout, 'rows': len(hold), 'balanced_accuracy': float(np.mean(rec)), 'accuracy': float((pred == yt).mean()),
         'mean_top_prob': float(srt[:, -1].mean()), 'mean_top2_margin': float((srt[:, -1] - srt[:, -2]).mean()),
         'nll': float(-np.log(np.clip(out[np.arange(len(yt)), yt], 1e-12, None)).mean()),
         'per_label_recall': {LABELS[c]: float((pred[yt == c] == c).mean()) for c in range(len(LABELS)) if (yt == c).any()},
         'score_seconds': round(time.time() - t2), 'train_steps_done': done, 'train_steps_planned': n_steps,
         'extra_rows': len(extra), 'seed': a.seed}
    np.savez(os.path.join(a.out, 'holdout_scores.npz'), ids=np.array(hold.id.astype(str).tolist(), dtype='U'),
             labels=np.array(LABELS, dtype='U'), y=yt, scores=out)
    json.dump(R, open(os.path.join(a.out, 'holdout_metrics.json'), 'w'), indent=1)
    log('HOLDOUT', json.dumps(R))
log('done')
