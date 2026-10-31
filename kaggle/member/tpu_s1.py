"""Stage 1 of a LoRA24 member on a Kaggle TPU v3-8: l24_worker.py's stage 1 (same model, triplets, loss, class weights,
learning-rate schedule, time box and output layout), one xmp.spawn process per TPU core instead of torchrun over two
T4s, so the separate 20 h weekly TPU quota can train members next to the 45 h GPU quota.

    PJRT_DEVICE=TPU python tpu_s1.py --stage 1 --base DIR --pool pool.parquet --labels JSON --out DIR [--init DIR] ...
    python tpu_s1.py --cpu --stage 1 ...       one process on the CPU, without torch_xla (local smoke test)

What XLA changes: the frozen base runs in bf16 (no fp16 autocast or GradScaler, so no overflow skips); each micro-batch
is right-padded to a multiple of 64 tokens, so only max_len / 64 shapes compile (right padding never changes a real
token's hidden state, and the pool masks pads out); gradients are averaged over the cores (xm.reduce_gradients) before
clipping; the time box is agreed as a min over cores. Core 0 writes what l24_worker's stage 2 loads.
"""
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


class Cpu:
    """The torch_xla calls used below, for one CPU process."""
    world, rank = 1, 0

    def device(self): return torch.device('cpu')
    def mark_step(self): pass
    def reduce_gradients(self, opt): pass
    def min_all(self, tag, x): return x
    def rendezvous(self, tag): pass


class Xla:
    def __init__(self):
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        self.xm, self.world, self.rank = xm, xr.world_size(), xr.global_ordinal()

    def device(self): return self.xm.xla_device()
    def mark_step(self): self.xm.mark_step()
    def reduce_gradients(self, opt): self.xm.reduce_gradients(opt)
    def min_all(self, tag, x): return self.xm.mesh_reduce(tag, x, min)
    def rendezvous(self, tag): self.xm.rendezvous(tag)


def log(rt, *x):
    if rt.rank == 0:
        print(f'[{(time.time() - T0) / 60:7.1f} min]', *x, flush=True)


def load_base(a, dtype):
    """l24_worker.load_base on the host, in `dtype`: the text path of the base truncated to --keep layers."""
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
    for name in ('vision_tower', 'multi_modal_projector'):
        if getattr(m, name, None) is not None:
            setattr(m, name, None)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return tok, m, tc.hidden_size


def bucket(enc, idx, pad, max_len, step=64):
    """l24_worker.batch on the host, right-padded to the next multiple of `step` tokens (at most max_len)."""
    L = min(max_len, step * math.ceil(max(len(enc[i]) for i in idx) / step))
    ids = torch.full((len(idx), L), pad, dtype=torch.long); att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        ids[r, :len(enc[i])] = torch.tensor(enc[i]); att[r, :len(enc[i])] = 1
    return ids, att


def save(model, net, out, meta):
    """l24_worker.save from device tensors: the adapter (peft layout) from host copies of the LoRA weights."""
    sd = {n: p.detach().cpu() for n, p in model.named_parameters() if 'lora_' in n}
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


def train(index, a):
    rt = Cpu() if a.cpu else Xla()
    dev = rt.device()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    pool = pd.read_parquet(a.pool)
    y_all = pool.model.map({l: i for i, l in enumerate(a.labels)}).values
    assert not np.isnan(y_all.astype(float)).any(), 'pool label outside --labels'
    y_all = y_all.astype(int)
    tok, m, hid = load_base(a, torch.float32 if a.cpu else torch.bfloat16)
    model, n_lora = W.add_lora(m, a, a.init)
    net = W.Net(model, hid, a.layers, len(a.labels))
    prior = 0.0
    if a.init:   # continue an earlier stage-1 run (GPU or TPU): its adapter and heads, a new triplet order and cosine cycle
        W.load_heads(net, a.init)
        prev = json.load(open(os.path.join(a.init, 'meta.json')))
        prior = prev.get('epochs_total', prev['epochs_done'])
    net = net.to(dev)
    net.train()   # from_pretrained returns eval mode, in which the decoder layers skip gradient checkpointing
    n_ckpt = W.checkpointed(model)
    enc = W.encode(tok, pool.text.astype(str).tolist(), a.max_len)
    counts = np.bincount(y_all, minlength=len(a.labels)).astype(float)
    cw = torch.tensor(len(y_all) / (len(a.labels) * np.maximum(counts, 1)), dtype=torch.float32).to(dev) \
        if a.class_weights else None
    groups = pool.prompt_key.astype(str).values
    rng = np.random.RandomState(a.seed)
    mbs = []
    for e in range(int(math.ceil(a.epochs))):
        ep = W.micro_batches(W.triplets(groups, np.arange(len(pool)), rng), enc, a.tri, rng)
        if e + 1 > a.epochs:
            ep = ep[:int(len(ep) * (a.epochs - e))]
        mbs += ep
    mine = mbs[rt.rank::rt.world][:len(mbs) // rt.world]
    planned = len(mine) // a.accum
    log(rt, f'stage 1 on {"cpu" if a.cpu else "tpu"}: {len(pool)} pool rows, {len(mbs)} micro-batches of {a.tri} '
        f'triplets over {a.epochs} epochs, {planned} optimizer steps per core ({rt.world} cores x {a.accum} accumulation), '
        f'LoRA params {n_lora}, class weights {"on" if cw is not None else "off"}, checkpointed layers {n_ckpt}')
    opt, params = W.optimizer(net, a)
    total, warm = planned, max(1, int(a.warmup * planned))
    budget = a.hours * 3600; t_train = time.time(); last_ckpt = time.time()
    step, losses, t_cal, last = 0, [], None, float('nan')
    meta = dict(stage=1, base=a.base, keep=a.keep, layers=a.layers, max_len=a.max_len, labels=a.labels, r=a.r,
                alpha=a.alpha, lr=a.lr, head_lr=a.head_lr, epochs=a.epochs, tri=a.tri, accum=a.accum, world=rt.world,
                class_weights=cw is not None, pool_rows=len(pool), init=a.init, seed=a.seed,
                device='cpu' if a.cpu else 'tpu-v3-8', dtype='float32' if a.cpu else 'bfloat16')
    for i, mb in enumerate(mine):
        idx = [r for t, _ in mb for r in t]
        ids, att = bucket(enc, idx, tok.pad_token_id, a.max_len)
        y = torch.as_tensor(y_all[idx]).to(dev)
        same = torch.tensor([s for _, s in mb], dtype=torch.float32).to(dev)
        loss = net(ids.to(dev), att.to(dev), y, same, cw)
        (loss / a.accum).backward()
        losses.append(loss.detach())
        if (i + 1) % a.accum:
            rt.mark_step()   # cut the graph between accumulated micro-batches; a full step is one graph
            continue
        rt.reduce_gradients(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        for g, base_lr in zip(opt.param_groups, (a.lr, a.head_lr)):
            g['lr'] = base_lr * W.lr_mult(step, total, warm)
        opt.step(); opt.zero_grad(set_to_none=True); rt.mark_step(); step += 1
        if step == 10:
            t_cal = time.time()
        if step == 60 or (step > 60 and step % 200 == 0):
            # the time box: steps that fit the remaining budget at the measured speed, the same on every core
            sps = (time.time() - t_cal) / (step - 10)
            fit = rt.min_all(f'fit{step}', step + int((budget - (time.time() - t_train)) / sps))
            new_total = min(planned, max(step + 1, fit))
            if step == 60:
                total = new_total; warm = min(warm, max(1, total // 20))
                log(rt, f'speed {sps:.2f} s/step: {total} of {planned} planned steps fit the {a.hours} h budget '
                    f'({total / planned * a.epochs:.2f} epochs)')
            elif new_total < total:
                total = new_total; log(rt, f'running slower: total cut to {total} steps')
        if step % 100 == 0 or step >= total:
            losses = losses[-100 * a.accum:]
            last = float(torch.stack(losses).mean())   # one host sync per 100 steps
            log(rt, f'step {step}/{total} loss {last:.4f} lr {opt.param_groups[0]["lr"]:.2e} '
                f'({(time.time() - t_train) / 3600:.2f} h)')
        if rt.rank == 0 and time.time() - last_ckpt > a.ckpt_min * 60:
            save(model, net, os.path.join(a.out, 'ckpt'), dict(meta, step=step, total=total)); last_ckpt = time.time()
        if step >= total:
            break
    hours = (time.time() - t_train) / 3600
    meta.update(steps=step, total_steps=total, planned_steps=planned, hours=round(hours, 3), overflow_microbatches=0,
                skipped_steps=0, final_loss=last, epochs_done=round(step / planned * a.epochs, 3))
    meta['epochs_total'] = round(prior + meta['epochs_done'], 3)   # pool epochs including the continued run's
    if rt.rank == 0:
        save(model, net, a.out, meta)
        log(rt, f'stage 1 done: {step} steps in {hours:.2f} h, {meta["epochs_done"]} epochs '
            f'({meta["epochs_total"]} in all); saved {a.out}')
    rt.rendezvous('saved')


if __name__ == '__main__':
    cpu = '--cpu' in sys.argv
    if cpu:
        sys.argv.remove('--cpu')
    a = W.parse()
    a.cpu = cpu
    assert a.stage == 1, 'tpu_s1.py runs stage 1 only'
    if cpu:
        train(0, a)
    else:
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(train, args=(a,))
