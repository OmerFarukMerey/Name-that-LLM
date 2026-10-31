"""Two-stage LoRA24 member trainer for Name That LLM, saved in the format nlm-v27's infer.py loads.

Lucas Boesen's public LoRA24 recipe (kernel name-that-llm-train-lora24-s7) with Dhruv Tibarewal's two stages (nlm-v27:
stage 1 on his 120k public pool with class weights, stage 2 on the 2,699 official rows): the base truncated to --keep
layers in float16 (frozen), LoRA r16 / alpha 32 / dropout .05 on every linear layer of the decoder (fp32), masked mean
pool (first token dropped) of 8 depths with one LayerNorm each, concatenated, into a plain and a prompt-centred linear
head; micro-batches of same-prompt triplets; loss CE(plain(z)) + CE(centred(z - triplet mean)); AdamW (adapter and
head learning rates), weight decay .01, warmup then cosine, clip 1.0, fp16 autocast with a GradScaler, gradient
checkpointing. A multimodal checkpoint (Ministral 3) keeps its text path only; infer.py loads the same AutoModel.

    python l24_worker.py --stage 0 ...                         fp16 health, memory and speed check (one GPU)
    torchrun --nproc_per_node 2 l24_worker.py --stage 1 ...    pool stage, data-parallel over both GPUs, time-boxed
    python l24_worker.py --stage 2 --jobs 0,2,4 ...            official rows, one job per held-out fold (or 'full')

Output of stage 1 and of each stage-2 job: adapter_config.json + adapter_model.safetensors (peft), heads.pt (norms.i,
plain, centered; rows in --labels order) and meta.json; stage-2 fold jobs also write valid.npz (held-out log-probs).
"""
import argparse
import contextlib
import importlib.util as _iu
import json
import math
import os
import time

_fs = _iu.find_spec   # Kaggle ships a torchao that some peft versions reject; it is unused here
_iu.find_spec = lambda name, *a, **k: None if name.split('.')[0] == 'torchao' else _fs(name, *a, **k)
import numpy as np
import pandas as pd
import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

T0 = time.time()
TARGET = r'(.*\.)?layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|gate_up_proj|down_proj))'


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--base', required=True); p.add_argument('--out', required=True)
    p.add_argument('--labels', required=True, help='json list: head row order')
    p.add_argument('--keep', type=int, default=24); p.add_argument('--layers', default='2,4,6,8,12,16,20,24')
    p.add_argument('--max-len', type=int, default=384)
    p.add_argument('--r', type=int, default=16); p.add_argument('--alpha', type=int, default=32)
    p.add_argument('--ldrop', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--head-lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.01); p.add_argument('--warmup', type=float, default=0.03)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--tri', type=int, default=2, help='triplets per micro-batch')
    p.add_argument('--accum', type=int, default=2, help='micro-batches per optimizer step and GPU')
    p.add_argument('--hours', type=float, default=9.0, help='stage 1: training time budget')
    p.add_argument('--ckpt-min', type=float, default=60.0)
    p.add_argument('--pool', default=''); p.add_argument('--train', default='')
    p.add_argument('--init', default='', help='stage 1: an earlier stage-1 output to continue; stage 2: the stage-1 output')
    p.add_argument('--jobs', default='', help="stage 2: comma list of held-out folds and/or 'full'")
    p.add_argument('--class-weights', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--check-rows', type=int, default=48, help='stage 0: rows per check')
    a = p.parse_args()
    a.layers = [int(x) for x in a.layers.split(',')]; a.labels = json.loads(a.labels)
    return a


RANK, WORLD = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
LOCAL = int(os.environ.get('LOCAL_RANK', 0))
DEV = torch.device(f'cuda:{LOCAL}') if torch.cuda.is_available() else torch.device('cpu')
CUDA = DEV.type == 'cuda'


def log(*x):
    if RANK == 0:
        print(f'[{(time.time() - T0) / 60:7.1f} min]', *x, flush=True)


def autocast():
    return torch.autocast('cuda', dtype=torch.float16) if CUDA else contextlib.nullcontext()


def load_base(a):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(a.base); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = a.keep; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:a.keep]
    m = AutoModel.from_pretrained(a.base, config=cfg, dtype=torch.float16 if CUDA else torch.float32)
    for name in ('vision_tower', 'multi_modal_projector'):   # text only: the image path never runs
        if getattr(m, name, None) is not None:
            setattr(m, name, None)
    m.to(DEV)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return tok, m, tc.hidden_size


def add_lora(m, a, init=''):
    from peft import LoraConfig, PeftModel, get_peft_model
    if init:
        model = PeftModel.from_pretrained(m, init, is_trainable=True)
    else:
        model = get_peft_model(m, LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.ldrop, bias='none',
                                             target_modules=TARGET))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n > 0, 'no LoRA parameters: target modules did not match'
    return model, n


class Net(nn.Module):
    """LoRA decoder + one LayerNorm per pooled depth + plain and centred heads; forward returns the loss."""

    def __init__(self, model, hid, layers, k):
        super().__init__()
        self.model, self.layers = model, layers
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in layers])
        self.plain = nn.Linear(hid * len(layers), k)
        self.centered = nn.Linear(hid * len(layers), k)
        self.drop = nn.Dropout(0.1)

    def features(self, ids, att):
        with autocast():
            hs = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        m = att.unsqueeze(-1).float()
        m[:, 0] = 0
        return torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1)) for l, nm in zip(self.layers, self.norms)], -1)

    def forward(self, ids, att, y, same, cw):
        z = self.features(ids, att)
        zt = z.view(z.shape[0] // 3, 3, -1)
        zc = (zt - zt.mean(1, keepdim=True)).reshape(z.shape[0], -1)
        return wce(self.plain(self.drop(z)), y, cw) + wce(self.centered(self.drop(zc)), y, cw, same.repeat_interleave(3))


def wce(logits, y, cw, mask=None):
    nll = F.cross_entropy(logits.float(), y, reduction='none')
    w = cw[y] if cw is not None else torch.ones_like(nll)
    if mask is not None:
        w = w * mask
    return (nll * w).sum() / w.sum().clamp(min=1e-6)


def encode(tok, texts, max_len):
    """infer.py's tokenisation: default special tokens, truncation at max_len, no end token."""
    return tok(list(texts), truncation=True, max_length=max_len)['input_ids']


def batch(enc, idx, pad):
    Lm = max(len(enc[i]) for i in idx)
    ids = torch.full((len(idx), Lm), pad, dtype=torch.long); att = torch.zeros((len(idx), Lm), dtype=torch.long)
    for r, i in enumerate(idx):
        ids[r, :len(enc[i])] = torch.tensor(enc[i]); att[r, :len(enc[i])] = 1
    return ids.to(DEV), att.to(DEV)


def triplets(groups, rows, rng):
    """Same-group triplets over `rows` (a short last chunk is filled with other members of its group); rows whose
    group has one member go into mixed triplets, flagged 0 so their centred loss is skipped."""
    by = {}
    for r in rows:
        by.setdefault(groups[r], []).append(int(r))
    out, solo = [], []
    for g in sorted(by):
        m = [int(x) for x in rng.permutation(by[g])]
        if len(m) == 1:
            solo.append(m[0]); continue
        for s in range(0, len(m), 3):
            b = m[s:s + 3]
            while len(b) < 3:
                b.append(int(rng.choice([x for x in m if x not in b] or m)))
            out.append((b, 1.0))
    solo = [int(x) for x in rng.permutation(solo)]
    out += [(solo[s:s + 3], 0.0) for s in range(0, len(solo) - 2, 3)]
    return [out[i] for i in rng.permutation(len(out))]


def micro_batches(trips, enc, tri, rng, window=64):
    """`tri` triplets per micro-batch, length-sorted within windows of the shuffled list, then shuffled again."""
    mbs = []
    for s in range(0, len(trips), window):
        w = sorted(trips[s:s + window], key=lambda t: max(len(enc[i]) for i in t[0]))
        mbs += [w[i:i + tri] for i in range(0, len(w) - tri + 1, tri)]
    return [mbs[i] for i in rng.permutation(len(mbs))]


def lr_mult(step, total, warm):
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


def optimizer(net, a):
    ad = [p for p in net.model.parameters() if p.requires_grad]
    hd = list(net.norms.parameters()) + list(net.plain.parameters()) + list(net.centered.parameters())
    opt = torch.optim.AdamW([{'params': ad, 'lr': a.lr}, {'params': hd, 'lr': a.head_lr}], weight_decay=a.wd)
    return opt, ad + hd


def checkpointed(model):
    """Decoder layers that will run gradient checkpointing (the flag is set and the layer is in training mode)."""
    n = sum(1 for m_ in model.modules() if getattr(m_, 'gradient_checkpointing', False) is True and m_.training)
    assert n > 0, 'gradient checkpointing is off: the decoder would keep every activation'
    return n


def zero_loss(params):
    """A loss of exactly zero that reaches every trainable parameter: stands in for a micro-batch whose fp16 forward
    overflowed, so every rank still runs the same backward (and gradient all-reduce)."""
    return sum(p.sum() for p in params) * 0.0


@torch.no_grad()
def predict(net, enc, rows, pad, bs=16):
    net.eval()
    out = np.zeros((len(rows), net.plain.out_features), np.float32)
    order = np.argsort([len(enc[i]) for i in rows], kind='stable')
    for s in range(0, len(order), bs):
        pos = order[s:s + bs]
        ids, att = batch(enc, [rows[i] for i in pos], pad)
        out[pos] = F.log_softmax(net.plain(net.features(ids, att)), -1).float().cpu().numpy()
    net.train()
    return out


def save(peft_model, net, out, meta):
    os.makedirs(out, exist_ok=True)
    peft_model.save_pretrained(out)
    H = {}
    for i, nm in enumerate(net.norms):
        H[f'norms.{i}.weight'] = nm.weight.detach().float().cpu().clone(); H[f'norms.{i}.bias'] = nm.bias.detach().float().cpu().clone()
    for h in ('plain', 'centered'):
        H[f'{h}.weight'] = getattr(net, h).weight.detach().float().cpu().clone()
        H[f'{h}.bias'] = getattr(net, h).bias.detach().float().cpu().clone()
    torch.save(H, os.path.join(out, 'heads.pt'))
    json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'), indent=1)


def load_heads(net, d):
    H = torch.load(os.path.join(d, 'heads.pt'), map_location='cpu')
    with torch.no_grad():
        for i, nm in enumerate(net.norms):
            nm.weight.copy_(H[f'norms.{i}.weight']); nm.bias.copy_(H[f'norms.{i}.bias'])
        for h in ('plain', 'centered'):
            getattr(net, h).weight.copy_(H[f'{h}.weight']); getattr(net, h).bias.copy_(H[f'{h}.bias'])


def step_fn(net, wrapped, opt, scaler, params, a):
    """One micro-batch: forward (loss), backward; returns (loss value, overflowed)."""
    def run(mb, enc, y_all, pad, cw, sync):
        idx = [i for t, _ in mb for i in t]
        ids, att = batch(enc, idx, pad)
        y = torch.as_tensor(y_all[idx], device=DEV)
        same = torch.tensor([s for _, s in mb], device=DEV, dtype=torch.float32)
        ctx = wrapped.no_sync() if (WORLD > 1 and not sync) else contextlib.nullcontext()
        with ctx:
            loss = wrapped(ids, att, y, same, cw)
            bad = not bool(torch.isfinite(loss))
            if bad:
                loss = zero_loss(params)
            scaler.scale(loss / a.accum).backward()
        return float(loss.detach()), bad
    return run


def stage0(a):
    """fp16 health (non-finite pooled depths on the longest pool texts), peak memory of a full-length training step,
    and training speed on random triplets. Writes <out>/check.json."""
    pool = pd.read_parquet(a.pool)
    tok, m, hid = load_base(a)
    model, n_lora = add_lora(m, a)
    net = Net(model, hid, a.layers, len(a.labels)).to(DEV)
    enc = encode(tok, pool.text.astype(str).tolist(), a.max_len)
    lens = np.array([len(x) for x in enc])
    longest = [int(i) for i in np.argsort(-lens)[:a.check_rows]]
    bad_rows = 0
    with torch.no_grad():
        net.eval()
        for s in range(0, len(longest), 6):
            ids, att = batch(enc, longest[s:s + 6], tok.pad_token_id)
            with autocast():
                hs = model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
            fin = torch.stack([torch.isfinite(hs[l]).flatten(1).all(1) for l in a.layers]).all(0)
            bad_rows += int((~fin).sum())
        net.train()
    log(f'fp16 check: {bad_rows}/{len(longest)} of the longest pool texts overflow at a pooled depth; '
        f'checkpointed layers {checkpointed(model)}')
    y_all = pool.model.map({l: i for i, l in enumerate(a.labels)}).fillna(0).astype(int).values
    opt, params = optimizer(net, a)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=CUDA)
    run = step_fn(net, net, opt, scaler, params, a)
    if CUDA:
        torch.cuda.reset_peak_memory_stats(DEV)
    t = time.time()
    trips = [([longest[3 * k], longest[3 * k + 1], longest[3 * k + 2]], 1.0) for k in range(4)]
    tri = a.tri
    while True:   # full-length micro-batches: the memory high-water mark; halve the micro-batch on running out
        try:
            for k in range(2):
                run(trips[2 * k:2 * k + tri], enc, y_all, tok.pad_token_id, None, True)
            break
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            assert tri > 1, 'out of GPU memory at one triplet per micro-batch'
            log(f'out of memory at {tri} triplets per micro-batch; trying {tri // 2}'); tri //= 2
    a.tri = tri
    peak = torch.cuda.max_memory_allocated(DEV) / 2 ** 30 if CUDA else 0.0
    opt.zero_grad(set_to_none=True)
    rng = np.random.RandomState(a.seed)
    groups = pool.prompt_key.astype(str).values
    mbs = micro_batches(triplets(groups, rng.choice(len(pool), min(3000, len(pool)), replace=False), rng), enc, a.tri, rng)[:12]
    if CUDA:
        torch.cuda.synchronize()
    t = time.time(); toks = 0; bad = 0
    for mb in mbs:
        _, b = run(mb, enc, y_all, tok.pad_token_id, None, True); bad += b
        toks += sum(len(enc[i]) for t_, _ in mb for i in t_)
    if CUDA:
        torch.cuda.synchronize()
    tps = toks / (time.time() - t)
    total_tokens = int(lens.sum())
    res = dict(ok=bad_rows <= 0.02 * len(longest) and bad == 0, overflow_rows=bad_rows, rows_checked=len(longest),
               peak_gib=round(peak, 2), tokens_per_s_one_gpu=round(tps, 1), lora_params=n_lora, pool_tokens=total_tokens,
               hours_per_epoch_two_gpus=round(total_tokens / (2 * tps) / 3600, 2), train_step_overflow=bad, hidden=hid,
               tri=tri)
    log('check', json.dumps(res))
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, 'check.json'), 'w'), indent=1)


def stage1(a):
    if CUDA:
        torch.cuda.set_device(DEV)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pool = pd.read_parquet(a.pool)
    y_all = pool.model.map({l: i for i, l in enumerate(a.labels)}).values
    assert not np.isnan(y_all.astype(float)).any(), 'pool label outside --labels'
    y_all = y_all.astype(int)
    tok, m, hid = load_base(a)
    model, n_lora = add_lora(m, a, a.init)   # before the process group: once one exists, peft loads an adapter on its tensor-parallel path
    if WORLD > 1:
        torch.distributed.init_process_group('nccl' if CUDA else 'gloo')
    net = Net(model, hid, a.layers, len(a.labels)).to(DEV)
    prior = 0.0
    if a.init:   # continue an earlier stage-1 run: its adapter and heads, a new triplet order (--seed), a new cosine cycle
        load_heads(net, a.init)
        prev = json.load(open(os.path.join(a.init, 'meta.json')))
        prior = prev.get('epochs_total', prev['epochs_done'])
    net.train()   # from_pretrained returns eval mode, in which the decoder layers skip gradient checkpointing
    n_ckpt = checkpointed(model)
    wrapped = net
    if WORLD > 1:
        wrapped = nn.parallel.DistributedDataParallel(net, device_ids=[LOCAL] if CUDA else None,
                                                      gradient_as_bucket_view=True)
    enc = encode(tok, pool.text.astype(str).tolist(), a.max_len)
    counts = np.bincount(y_all, minlength=len(a.labels)).astype(float)
    cw = torch.tensor(len(y_all) / (len(a.labels) * np.maximum(counts, 1)), dtype=torch.float32, device=DEV) \
        if a.class_weights else None
    groups = pool.prompt_key.astype(str).values
    rng = np.random.RandomState(a.seed)
    mbs = []
    for e in range(int(math.ceil(a.epochs))):
        ep = micro_batches(triplets(groups, np.arange(len(pool)), rng), enc, a.tri, rng)
        if e + 1 > a.epochs:
            ep = ep[:int(len(ep) * (a.epochs - e))]
        mbs += ep
    mine = mbs[RANK::WORLD][:len(mbs) // WORLD]
    planned = len(mine) // a.accum
    log(f'stage 1: {len(pool)} pool rows, {len(mbs)} micro-batches of {a.tri} triplets over {a.epochs} epochs, '
        f'{planned} optimizer steps per rank ({WORLD} ranks x {a.accum} accumulation), LoRA params {n_lora}, '
        f'class weights {"on" if cw is not None else "off"}, checkpointed layers {n_ckpt}')
    opt, params = optimizer(net, a)
    scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=CUDA)
    run = step_fn(net, wrapped, opt, scaler, params, a)
    total, warm = planned, max(1, int(a.warmup * planned))
    budget = a.hours * 3600; t_train = time.time(); last_ckpt = time.time()
    step, losses, bad, skipped, t_cal = 0, [], 0, 0, None
    meta = dict(stage=1, base=a.base, keep=a.keep, layers=a.layers, max_len=a.max_len, labels=a.labels, r=a.r,
                alpha=a.alpha, lr=a.lr, head_lr=a.head_lr, epochs=a.epochs, tri=a.tri, accum=a.accum, world=WORLD,
                class_weights=cw is not None, pool_rows=len(pool), init=a.init, seed=a.seed)
    for i, mb in enumerate(mine):
        sync = (i + 1) % a.accum == 0
        lv, b = run(mb, enc, y_all, tok.pad_token_id, cw, sync)
        losses.append(lv); bad += b
        if not sync:
            continue
        for g, base_lr in zip(opt.param_groups, (a.lr, a.head_lr)):
            g['lr'] = base_lr * lr_mult(step, total, warm)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        before = scaler.get_scale(); scaler.step(opt); scaler.update()
        skipped += scaler.get_scale() < before
        opt.zero_grad(set_to_none=True); step += 1
        if step == 10:
            t_cal = time.time()
        if step == 60 or (step > 60 and step % 200 == 0):
            # the time box: steps that fit the remaining budget at the measured speed (rank 0 decides for all)
            sps = (time.time() - t_cal) / (step - 10)
            fit = step + int((budget - (time.time() - t_train)) / sps)
            new_total = torch.tensor([min(planned, max(step + 1, fit))], device=DEV)
            if WORLD > 1:
                torch.distributed.broadcast(new_total, 0)
            if step == 60:
                total = int(new_total.item()); warm = min(warm, max(1, total // 20))
                log(f'speed {sps:.2f} s/step: {total} of {planned} planned steps fit the {a.hours} h budget '
                    f'({total / planned * a.epochs:.2f} epochs)')
            elif int(new_total.item()) < total:
                total = int(new_total.item()); log(f'running slower: total cut to {total} steps')
        if step % 100 == 0:
            log(f'step {step}/{total} loss {np.mean(losses[-100 * a.accum:]):.4f} overflow {bad} skipped {skipped} '
                f'lr {opt.param_groups[0]["lr"]:.2e} ({(time.time() - t_train) / 3600:.2f} h)')
        if RANK == 0 and time.time() - last_ckpt > a.ckpt_min * 60:
            save(model, net, os.path.join(a.out, 'ckpt'), dict(meta, step=step, total=total)); last_ckpt = time.time()
        if step >= total:
            break
    hours = (time.time() - t_train) / 3600
    meta.update(steps=step, total_steps=total, planned_steps=planned, hours=round(hours, 3), overflow_microbatches=bad,
                skipped_steps=int(skipped), final_loss=float(np.mean(losses[-100 * a.accum:])),
                epochs_done=round(step / planned * a.epochs, 3))
    meta['epochs_total'] = round(prior + meta['epochs_done'], 3)   # pool epochs including the continued run's
    if RANK == 0:
        save(model, net, a.out, meta)
        log(f'stage 1 done: {step} steps in {hours:.2f} h, {meta["epochs_done"]} epochs, overflow micro-batches {bad}, '
            f'skipped steps {skipped}; saved {a.out}')
    if WORLD > 1:
        torch.distributed.barrier(); torch.distributed.destroy_process_group()


def stage2(a):
    torch.manual_seed(a.seed)
    os.makedirs(os.path.dirname(a.out.rstrip('/')) or '.', exist_ok=True)
    tr = pd.read_parquet(a.train)
    y_all = tr.label.astype(int).values; folds = tr.fold.astype(int).values
    tok, m, hid = load_base(a)
    model, n_lora = add_lora(m, a, a.init)
    net = Net(model, hid, a.layers, len(a.labels)).to(DEV)
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    s1 = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
    enc = encode(tok, tr.text.astype(str).tolist(), a.max_len)
    groups = tr.prompt_id.astype(str).values
    s1_meta = json.load(open(os.path.join(a.init, 'meta.json')))
    if '0' in a.jobs.split(','):   # the pool-only model on the official rows, none of which it has seen
        lp = predict(net, enc, list(range(len(tr))), tok.pad_token_id)
        s1_meta['official_balanced_accuracy'] = float(np.mean([(lp[y_all == c].argmax(1) == c).mean() for c in np.unique(y_all)]))
        np.save(a.out + '_stage1_on_train.npy', lp)
        log(f"stage-1 model on the official rows: balanced accuracy {s1_meta['official_balanced_accuracy']:.4f}")
    for job in a.jobs.split(','):
        t0 = time.time()
        set_peft_model_state_dict(model, s1); load_heads(net, a.init)
        rows = np.arange(len(tr)) if job == 'full' else np.where(folds != int(job))[0]
        rng = np.random.RandomState(a.seed + (99 if job == 'full' else int(job)))
        mbs = []
        for e in range(int(math.ceil(a.epochs))):
            mbs += micro_batches(triplets(groups, rows, rng), enc, a.tri, rng)
        total = len(mbs) // a.accum; warm = max(1, int(a.warmup * total))
        opt, params = optimizer(net, a)
        scaler = torch.amp.GradScaler('cuda', init_scale=256.0, enabled=CUDA)
        run = step_fn(net, net, opt, scaler, params, a)
        net.train(); checkpointed(model)   # see stage1: gradient checkpointing only runs in training mode
        step, losses, bad, skipped = 0, [], 0, 0
        for i, mb in enumerate(mbs[:total * a.accum]):
            lv, b = run(mb, enc, y_all, tok.pad_token_id, None, True)
            losses.append(lv); bad += b
            if (i + 1) % a.accum:
                continue
            for g, base_lr in zip(opt.param_groups, (a.lr, a.head_lr)):
                g['lr'] = base_lr * lr_mult(step, total, warm)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            before = scaler.get_scale(); scaler.step(opt); scaler.update()
            skipped += scaler.get_scale() < before
            opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                log(f'job {job} step {step}/{total} loss {np.mean(losses[-50 * a.accum:]):.4f} overflow {bad} skipped {skipped}')
        name = os.path.basename(a.out.rstrip('/')) + ('' if job == 'full' else f'_f{job}')
        out = os.path.join(os.path.dirname(a.out.rstrip('/')), name)
        check = predict(net, enc, list(range(40)), tok.pad_token_id)
        meta = dict(s1_meta, stage=2, job=job, steps=step, epochs=a.epochs, lr=a.lr, head_lr=a.head_lr, tri=a.tri,
                    accum=a.accum, overflow_microbatches=bad, skipped_steps=int(skipped), init=a.init,
                    final_loss=float(np.mean(losses[-50 * a.accum:])), minutes=round((time.time() - t0) / 60, 2),
                    in_sample_check_40=float((check.argmax(1) == y_all[:40]).mean()))
        if job != 'full':
            valid = np.where(folds == int(job))[0]
            lp = predict(net, enc, list(valid), tok.pad_token_id)
            meta['valid_accuracy'] = float((lp.argmax(1) == y_all[valid]).mean())
            meta['valid_balanced_accuracy'] = float(np.mean([(lp[y_all[valid] == c].argmax(1) == c).mean()
                                                            for c in np.unique(y_all[valid])]))
        save(model, net, out, meta)
        np.save(os.path.join(out, 'check40.npy'), check)
        if job != 'full':
            np.savez(os.path.join(out, 'valid.npz'), ids=np.array(tr.id.astype(str).values[valid].tolist()), logits=lp)
        log(f"job {job} done: {step} steps, {meta['minutes']} min, in-sample 40 {meta['in_sample_check_40']:.2f}"
            + (f", held-out accuracy {meta['valid_accuracy']:.4f} balanced {meta['valid_balanced_accuracy']:.4f}"
               if job != 'full' else '') + f'; saved {out}')


if __name__ == '__main__':
    args = parse()
    {0: stage0, 1: stage1, 2: stage2}[args.stage](args)
