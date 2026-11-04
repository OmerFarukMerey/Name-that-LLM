"""A LoRA24 member trained end to end on a Kaggle TPU v5e-8: stage 1 on a public pool, then stage 2 on the official
rows once per held-out fold and once on all of them, with out-of-fold log-probs for the replica.

l24_worker.py's recipe (base truncated to --keep layers and frozen, LoRA on every decoder projection, masked mean pool
of 8 depths with one LayerNorm each into a plain and a prompt-centred head, same-prompt triplets, class-weighted
cross-entropy, AdamW with warmup and cosine, clip 1.0) and its output layout, which infer.py's l24_predict loads as is.
What the TPU changes (after GoPain's forum measurements, 2026-11-01):
- one process, data parallel over every chip with torch_xla SPMD (the batch dimension sharded, the model replicated);
- bf16 base (no fp16 overflow, no GradScaler), fp32 LoRA and heads;
- fixed batch shapes: a triplet goes to the smallest (tokens, rows) bucket that holds its longest passage, every
  bucket spends about the same tokens per step, and leftovers that fill no whole batch are dropped, so each shape
  compiles once;
- torch_xla's own gradient checkpointing on every decoder layer (torch's is undone by XLA's common-subexpression pass);
- LoRA dropout 0 (the recomputed forward would need the dropout mask replayed).

    PJRT_DEVICE=TPU python tpu_member.py --base DIR --pool pool.parquet --train train.parquet --labels JSON --out DIR
    python tpu_member.py --cpu ...        the same on the CPU in float32, without torch_xla (local smoke test)

pool.parquet: text, model, prompt_key. train.parquet: id, text, label (index into --labels), fold, prompt_id.
Writes <out>_s1/ (stage 1), <out>_f<k>/ and <out>/ (stage 2), s1_on_train.npy, oof.npz (ids, labels, logits: the
held-out fold predictions of the fold jobs, every official row once) and summary.json.
"""
import argparse
import functools
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l24_worker as W   # noqa: E402

T0 = time.time()
S1_SHAPES = [(96, 216), (160, 144), (256, 96), (384, 48)]   # (tokens, rows); rows are multiples of 3 x 8 chips
S2_SHAPES = [(96, 72), (160, 48), (256, 24), (384, 24)]
EVAL_SHAPES = [(96, 128), (160, 96), (256, 64), (384, 32)]


def log(*x):
    print(f'[{(time.time() - T0) / 60:7.1f} min]', *x, flush=True)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--base', required=True); p.add_argument('--out', required=True)
    p.add_argument('--labels', required=True); p.add_argument('--pool', required=True); p.add_argument('--train', required=True)
    p.add_argument('--keep', type=int, default=24); p.add_argument('--layers', default='2,4,6,8,12,16,20,24')
    p.add_argument('--max-len', type=int, default=384)
    p.add_argument('--r', type=int, default=16); p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--head-lr', type=float, default=2e-3)
    p.add_argument('--lr2', type=float, default=8e-5); p.add_argument('--head-lr2', type=float, default=8e-4)
    p.add_argument('--wd', type=float, default=0.01)
    p.add_argument('--warmup', type=float, default=0.03); p.add_argument('--warmup2', type=float, default=0.1)
    p.add_argument('--epochs', type=float, default=2.0); p.add_argument('--epochs2', type=float, default=2.0)
    p.add_argument('--hours', type=float, default=3.0, help='stage 1 time box')
    p.add_argument('--jobs', default='0,1,2,3,4,full')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tp', type=int, default=1, help='tensor parallel degree (Megatron-style weight sharding over this many '
                   'chips; the rest of the chips split the batch), for bases too large to replicate on one chip')
    p.add_argument('--pool-weight', type=float, default=0.0, help='stage 1: loss weight of each pool row = match_weight '
                   '** this, normalised to mean 1 within each label and clipped to [1/c, c] (0: every row weighs 1)')
    p.add_argument('--pool-weight-clip', type=float, default=4.0)
    p.add_argument('--shapes-scale', type=float, default=1.0, help='CPU smoke test: rows per shape x this (min 3)')
    a = p.parse_args()
    a.layers = [int(x) for x in a.layers.split(',')]; a.labels = json.loads(a.labels)
    a.ldrop = 0.0
    return a


class Runtime:
    """SPMD data parallel on the XLA device, or one CPU process."""

    def __init__(self, cpu, tp=1):
        self.cpu = cpu
        if cpu:
            self.dev, self.n = torch.device('cpu'), 1
            return
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.spmd as xs
        import torch_xla.runtime as xr
        xr.use_spmd()
        self.xm, self.xs = xm, xs
        self.n = xr.global_runtime_device_count()
        assert self.n % tp == 0, (self.n, tp)
        self.dp = self.n // tp
        self.mesh = xs.Mesh(np.arange(self.n), (self.dp, tp), ('data', 'model'))
        self.dev = torch_xla.device() if hasattr(torch_xla, 'device') else xm.xla_device()

    def put(self, t):
        t = t.to(self.dev)
        if not self.cpu:   # batch rows over the chips
            self.xs.mark_sharding(t, self.mesh, ('data',) + (None,) * (t.dim() - 1))
        return t

    def step(self):
        if not self.cpu:
            self.xm.mark_step()


COLUMN = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj')   # output features split over 'model'
ROW = ('o_proj', 'down_proj')                                      # input features split over 'model'


def tp_spec(name, p):
    """Megatron-style split of a decoder projection's base weight or LoRA factor, or None (replicated)."""
    parts = name.split('.')
    proj = next((x for x in parts if x in COLUMN + ROW), None)
    if proj is None or p.dim() != 2:
        return None
    if proj in COLUMN and 'lora_A' not in parts:    # base [out, in] and lora_B [out, r]: split out
        return ('model', None)
    if proj in ROW and 'lora_B' not in parts:       # base [out, in] and lora_A [r, in]: split in
        return (None, 'model')
    return None


def shard_tp(model, rt):
    """Annotates every decoder projection (base weight and LoRA factor on the split side); XLA's SPMD partitioner
    inserts the all-reduces. Heads (q 40 / kv 8 for Qwen3-14B) must divide by the tp degree."""
    n = 0
    for name, p in model.named_parameters():
        spec = tp_spec(name, p)
        if spec is not None:
            rt.xs.mark_sharding(p, rt.mesh, spec); n += 1
    return n


def place(net, rt, tp):
    """Moves the model to the device. With tensor parallelism one parameter at a time, each split right after its
    transfer: an unannotated transfer is replicated on every chip, and Qwen3-14B's 27 layers (19 GB in bf16) do not fit
    one chip's 16 GB."""
    if rt.cpu or tp == 1:
        return net.to(rt.dev), 0
    n = 0
    for name, p in net.named_parameters():
        p.data = p.data.to(rt.dev)
        spec = tp_spec(name, p)
        if spec is not None:
            rt.xs.mark_sharding(p, rt.mesh, spec); n += 1
    for mod in net.modules():
        for k, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[k] = b.to(rt.dev)
    return net, n


class Net(W.Net):
    """l24_worker's Net; forward also takes per-row loss weights (stage 1's pool weighting), which multiply the class
    weights in the plain loss and the same-group mask in the centred one."""

    def forward(self, ids, att, y, same, cw, rw=None):
        if rw is None:
            return super().forward(ids, att, y, same, cw)
        z = self.features(ids, att)
        zt = z.view(z.shape[0] // 3, 3, -1)
        zc = (zt - zt.mean(1, keepdim=True)).reshape(z.shape[0], -1)
        return W.wce(self.plain(self.drop(z)), y, cw, rw) + W.wce(self.centered(self.drop(zc)), y, cw, same.repeat_interleave(3) * rw)


def pool_weights(pool, p, c):
    """match_weight ** p, mean 1 within each label, clipped to [1/c, c], mean 1 again (the class balance is kept)."""
    w = np.clip(pool.match_weight.astype(float).fillna(1.0).to_numpy(), 1e-3, None) ** p
    lab = pool.model.to_numpy()
    norm = lambda v: pd.Series(v).groupby(lab).transform(lambda s: s / s.mean()).to_numpy()
    return norm(np.clip(norm(w), 1 / c, c)).astype(np.float32)


def load_base(a, dtype):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.base); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = a.keep; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:a.keep]
    try:
        m = AutoModel.from_pretrained(a.base, config=cfg, dtype=dtype)
    except TypeError:   # transformers < 4.56 calls it torch_dtype
        m = AutoModel.from_pretrained(a.base, config=cfg, torch_dtype=dtype)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return tok, m, tc.hidden_size


def xla_checkpointing(model):
    """Swap the checkpoint function of every layer that checkpoints for torch_xla's (optimisation barriers)."""
    import inspect
    from torch_xla.utils.checkpoint import checkpoint
    f = functools.partial(checkpoint, use_reentrant=True) if 'use_reentrant' in inspect.signature(checkpoint).parameters \
        else checkpoint   # older torch_xla: always the reentrant form
    n = 0
    for m in model.modules():
        if getattr(m, 'gradient_checkpointing', False) is True:
            m._gradient_checkpointing_func = f; n += 1
    assert n > 0, 'no layer checkpoints'
    return n


def scaled(shapes, s):
    return [(L, max(3, int(round(R * s / 3)) * 3)) for L, R in shapes]


def buckets(trips, enc, shapes, rng):
    """Micro-batches (tokens, triplets): each triplet in the smallest bucket that holds its longest row."""
    by = {L: [] for L, _ in shapes}
    for t in trips:
        m = max(len(enc[i]) for i in t[0])
        by[next((L for L, _ in shapes if m <= L), shapes[-1][0])].append(t)
    out = []
    for L, R in shapes:
        k = R // 3
        out += [(L, by[L][s:s + k]) for s in range(0, len(by[L]) - k + 1, k)]
    return [out[i] for i in rng.permutation(len(out))]


def tensors(enc, idx, L, pad):
    ids = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x = enc[i][:L]
        ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
    return ids, att


def train_loop(net, params, batches, enc, y_all, cw, pad, rt, lr, head_lr, warmup, hours=None, tag='', rw_all=None):
    opt = torch.optim.AdamW([{'params': [p for p in net.model.parameters() if p.requires_grad], 'lr': lr},
                             {'params': list(net.norms.parameters()) + list(net.plain.parameters())
                              + list(net.centered.parameters()), 'lr': head_lr}], weight_decay=0.01)
    total = len(batches); warm = max(1, int(warmup * total)); losses = []
    t_train = time.time(); t_cal = None; step = 0
    net.train()
    for L, mb in batches:
        idx = [i for t, _ in mb for i in t]
        ids, att = tensors(enc, idx, L, pad)
        y = rt.put(torch.as_tensor(y_all[idx]))
        same = torch.tensor([s for _, s in mb], dtype=torch.float32).to(rt.dev)
        if rw_all is None:
            loss = net(rt.put(ids), rt.put(att), y, same, cw)
        else:
            loss = net(rt.put(ids), rt.put(att), y, same, cw, rt.put(torch.as_tensor(rw_all[idx])))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g, base in zip(opt.param_groups, (lr, head_lr)):
            g['lr'] = base * W.lr_mult(step, total, warm)
        opt.step(); opt.zero_grad(set_to_none=True); rt.step(); step += 1
        losses.append(loss.detach())
        if step == 20:
            t_cal = time.time()
        if hours is not None and step == 120:   # the time box, at the speed measured past the compiles
            sps = (time.time() - t_cal) / 100
            fit = step + int((hours * 3600 - (time.time() - t_train)) / sps)
            if fit < total:
                total = max(step + 1, fit); warm = min(warm, max(1, total // 20))
            log(f'{tag} speed {sps:.2f} s/step: {total} of {len(batches)} steps fit {hours} h')
        if step % 100 == 0 or step >= total:
            log(f'{tag} step {step}/{total} loss {float(torch.stack(losses[-100:]).mean()):.4f} '
                f'lr {opt.param_groups[0]["lr"]:.2e} ({(time.time() - t_train) / 3600:.2f} h)')
        if step >= total:
            break
    del opt
    return dict(steps=step, planned=len(batches), hours=round((time.time() - t_train) / 3600, 3),
                final_loss=float(torch.stack(losses[-100:]).mean()) if losses else float('nan'))


@torch.no_grad()
def predict(net, enc, rows, pad, rt, shapes):
    net.eval()
    out = np.zeros((len(rows), net.plain.out_features), np.float32)
    by = {}
    for p, r in enumerate(rows):
        by.setdefault(next((L for L, _ in shapes if len(enc[r]) <= L), shapes[-1][0]), []).append(p)
    for L, R in shapes:
        ps = by.get(L, [])
        for s in range(0, len(ps), R):
            chunk = ps[s:s + R]
            idx = [rows[p] for p in chunk]
            ids, att = tensors(enc, idx + [idx[0]] * (R - len(idx)), L, pad)
            z = net.plain(net.features(rt.put(ids), rt.put(att)))
            lp = torch.log_softmax(z.float(), -1)
            rt.step()
            out[chunk] = lp.cpu().numpy()[:len(chunk)]
    net.train()
    return out


def save(model, net, out, meta):
    sd = {n: p.detach().cpu().float() for n, p in model.named_parameters() if 'lora_' in n}
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, state_dict=sd)
    H = {}
    for i, nm in enumerate(net.norms):
        H[f'norms.{i}.weight'] = nm.weight.detach().float().cpu().clone(); H[f'norms.{i}.bias'] = nm.bias.detach().float().cpu().clone()
    for h in ('plain', 'centered'):
        H[f'{h}.weight'] = getattr(net, h).weight.detach().float().cpu().clone()
        H[f'{h}.bias'] = getattr(net, h).bias.detach().float().cpu().clone()
    torch.save(H, os.path.join(out, 'heads.pt'))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)


def bacc(lp, y):
    return float(np.mean([(lp[y == c].argmax(1) == c).mean() for c in np.unique(y)]))


def main():
    a = parse()
    rt = Runtime(a.cpu, a.tp)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    s1_shapes, s2_shapes, ev_shapes = (S1_SHAPES, S2_SHAPES, EVAL_SHAPES) if a.shapes_scale == 1 else \
        (scaled(S1_SHAPES, a.shapes_scale), scaled(S2_SHAPES, a.shapes_scale), scaled(EVAL_SHAPES, a.shapes_scale))
    if not a.cpu:
        assert all(R % (3 * rt.dp) == 0 for _, R in s1_shapes + s2_shapes) and all(R % rt.dp == 0 for _, R in ev_shapes)
    pool = pd.read_parquet(a.pool); tr = pd.read_parquet(a.train)
    L2I = {l: i for i, l in enumerate(a.labels)}
    yp = pool.model.map(L2I); assert yp.notna().all(), 'pool label outside --labels'
    yp = yp.to_numpy(int); yt = tr.label.to_numpy(int); folds = tr.fold.to_numpy(int)
    tok, m, hid = load_base(a, torch.float32 if a.cpu and not os.environ.get('TPM_CPU_BF16') else torch.bfloat16)
    model, n_lora = W.add_lora(m, a)
    net = Net(model, hid, a.layers, len(a.labels))
    net.train()   # from_pretrained returns eval mode, where the layers skip checkpointing
    n_ck = W.checkpointed(model) if a.cpu else xla_checkpointing(model)
    net, n_tp = place(net, rt, a.tp)
    if n_tp:
        log('tensor-parallel weights:', n_tp)
    params = [p for p in net.parameters() if p.requires_grad]
    log(f'{rt.n} devices, LoRA params {n_lora}, checkpointed layers {n_ck}, pool {len(pool)}, official {len(tr)}')
    enc_p = W.encode(tok, pool.text.astype(str).tolist(), a.max_len)
    enc_t = W.encode(tok, tr.text.astype(str).tolist(), a.max_len)
    log('tokenised; pool tokens', int(sum(len(x) for x in enc_p)))
    counts = np.bincount(yp, minlength=len(a.labels)).astype(float)
    cw = torch.tensor(len(yp) / (len(a.labels) * np.maximum(counts, 1)), dtype=torch.float32).to(rt.dev)
    rw = pool_weights(pool, a.pool_weight, a.pool_weight_clip) if a.pool_weight > 0 else None
    if rw is not None:
        log(f'pool rows weighted by match_weight ** {a.pool_weight} (clip {a.pool_weight_clip}): effective share '
            f'{float(rw.sum() ** 2 / (rw.astype(np.float64) ** 2).sum() / len(rw)):.3f}')
    rng = np.random.RandomState(a.seed)
    batches = []
    for e in range(int(math.ceil(a.epochs))):
        ep = buckets(W.triplets(pool.prompt_key.astype(str).values, np.arange(len(pool)), rng), enc_p, s1_shapes, rng)
        batches += ep[:int(len(ep) * min(1.0, a.epochs - e))]
    base_meta = dict(base=a.base, keep=a.keep, layers=a.layers, max_len=a.max_len, labels=a.labels, r=a.r, alpha=a.alpha,
                     device='cpu' if a.cpu else f'tpu x{rt.n} spmd', dtype='float32' if a.cpu else 'bfloat16',
                     pool_rows=len(pool), seed=a.seed, pool_weight=a.pool_weight, pool_weight_clip=a.pool_weight_clip)
    log(f'stage 1: {len(batches)} batches over {a.epochs} epochs, shapes {s1_shapes}')
    r1 = train_loop(net, params, batches, enc_p, yp, cw, tok.pad_token_id, rt, a.lr, a.head_lr, a.warmup, a.hours, 'stage 1', rw)
    s1_meta = dict(base_meta, stage=1, lr=a.lr, head_lr=a.head_lr, epochs=a.epochs, **r1,
                   epochs_done=round(a.epochs * r1['steps'] / max(1, r1['planned']), 3))
    s1_dir = a.out.rstrip('/') + '_s1'
    save(model, net, s1_dir, s1_meta)
    lp1 = predict(net, enc_t, list(range(len(tr))), tok.pad_token_id, rt, ev_shapes)
    np.save(os.path.join(os.path.dirname(a.out.rstrip('/')) or '.', 's1_on_train.npy'), lp1)
    s1_meta['official_balanced_accuracy'] = bacc(lp1, yt)
    json.dump(s1_meta, open(os.path.join(s1_dir, 'meta.json'), 'w'), indent=1)
    log(f"stage 1 done {r1}; pool-only model on the official rows: balanced accuracy {s1_meta['official_balanced_accuracy']:.4f}")
    keep = {n: p.detach().cpu().clone() for n, p in net.named_parameters() if p.requires_grad}
    oof = np.zeros((len(tr), len(a.labels)), np.float32); seen = np.zeros(len(tr), bool)
    summary = dict(stage1=s1_meta, jobs={})
    groups = tr.prompt_id.astype(str).values
    for job in a.jobs.split(','):
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in keep:
                    p.copy_(keep[n].to(p.device))
        if a.tp > 1 and not a.cpu:   # re-annotate the reset factors
            shard_tp(net.model, rt)
        rows = np.arange(len(tr)) if job == 'full' else np.where(folds != int(job))[0]
        jr = np.random.RandomState(a.seed + (99 if job == 'full' else int(job)))
        b2 = []
        for e in range(int(math.ceil(a.epochs2))):
            ep = buckets(W.triplets(groups, rows, jr), enc_t, s2_shapes, jr)
            b2 += ep[:int(len(ep) * min(1.0, a.epochs2 - e))]
        r2 = train_loop(net, params, b2, enc_t, yt, None, tok.pad_token_id, rt, a.lr2, a.head_lr2, a.warmup2, None, f'job {job}')
        name = os.path.basename(a.out.rstrip('/')) + ('' if job == 'full' else f'_f{job}')
        out = os.path.join(os.path.dirname(a.out.rstrip('/')) or '.', name)
        check = predict(net, enc_t, list(range(40)), tok.pad_token_id, rt, ev_shapes)
        meta = dict(base_meta, stage=2, job=job, lr=a.lr2, head_lr=a.head_lr2, epochs=a.epochs2, **r2,
                    stage1=os.path.basename(s1_dir), in_sample_check_40=float((check.argmax(1) == yt[:40]).mean()))
        if job != 'full':
            v = np.where(folds == int(job))[0]
            lp = predict(net, enc_t, list(v), tok.pad_token_id, rt, ev_shapes)
            oof[v] = lp; seen[v] = True
            meta['valid_balanced_accuracy'] = bacc(lp, yt[v])
        save(model, net, out, meta)
        np.save(os.path.join(out, 'check40.npy'), check)
        summary['jobs'][job] = {k: meta[k] for k in meta if k not in base_meta}
        log(f"job {job} done: {r2}, in-sample 40 {meta['in_sample_check_40']:.2f}"
            + (f", held-out balanced accuracy {meta['valid_balanced_accuracy']:.4f}" if job != 'full' else ''))
    d = os.path.dirname(a.out.rstrip('/')) or '.'
    if seen.all():
        np.savez(os.path.join(d, 'oof.npz'), ids=np.array(tr.id.astype(str).tolist()), labels=np.array(a.labels), logits=oof)
        summary['oof_balanced_accuracy'] = bacc(oof, yt)
        log(f"out-of-fold balanced accuracy {summary['oof_balanced_accuracy']:.4f} (stage 1 alone {s1_meta['official_balanced_accuracy']:.4f})")
    json.dump(summary, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
    log('ALL DONE')


if __name__ == '__main__':
    main()
