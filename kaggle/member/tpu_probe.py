"""TPU probe for the LoRA24 members: can Kaggle's TPU v3-8 (8 cores x 16 GB) train Lucas Boesen's LoRA24 recipe on an
8B base, and how fast? On T4 x2 one 120k-pool epoch of Ministral-3-8B takes ~10 h of the 45 h weekly GPU quota; the
20 h weekly TPU quota is separate and unused.

    PJRT_DEVICE=TPU python tpu_probe.py --base DIR --nprocs 1|8 --lens 256,384 --batch 6 --steps 6

One xmp.spawn process per TPU core, each holding the frozen bf16 base (its first --keep layers) and a LoRA copy;
xm.optimizer_step all-reduces the LoRA and head gradients (data parallel). Random token ids at fixed shapes (each
shape compiles once) through l24_worker.py's model: LoRA r16 on every decoder linear layer (fp32), gradient
checkpointing, masked mean pool of 8 depths with one LayerNorm each, plain and prompt-centred heads on triplets.
Prints step times, tokens/s over all cores and the device memory.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from torch import nn

T0 = time.time()
TARGET = r'(.*\.)?layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|gate_up_proj|down_proj))'
LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]


def log(*x):
    print(f'[{time.time() - T0:7.1f} s]', *x, flush=True)


def load_base(base, keep):
    """l24_worker.load_base in bf16 on the host: the text path of the base, truncated to `keep` layers."""
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(base); tc = cfg.text_config if hasattr(cfg, 'text_config') else cfg
    tc.num_hidden_layers = keep; tc.use_cache = False
    if getattr(tc, 'layer_types', None) is not None:
        tc.layer_types = tc.layer_types[:keep]
    try:
        m = AutoModel.from_pretrained(base, config=cfg, dtype=torch.bfloat16)
    except TypeError:   # transformers < 4.56 calls it torch_dtype
        m = AutoModel.from_pretrained(base, config=cfg, torch_dtype=torch.bfloat16)
    for name in ('vision_tower', 'multi_modal_projector'):
        if getattr(m, name, None) is not None:
            setattr(m, name, None)
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return m, tc.hidden_size


class Net(nn.Module):
    def __init__(self, model, hid, k=18):
        super().__init__()
        self.model = model
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in LAYERS])
        self.plain = nn.Linear(hid * len(LAYERS), k)
        self.centered = nn.Linear(hid * len(LAYERS), k)

    def forward(self, ids, att, y):
        hs = self.model(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
        m = att.unsqueeze(-1).float()
        z = torch.cat([nm((hs[l].float() * m).sum(1) / m.sum(1)) for l, nm in zip(LAYERS, self.norms)], -1)
        zt = z.view(z.shape[0] // 3, 3, -1)
        zc = (zt - zt.mean(1, keepdim=True)).reshape(z.shape[0], -1)
        return F.cross_entropy(self.plain(z), y) + F.cross_entropy(self.centered(zc), y)


def run(index, a):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from peft import LoraConfig, get_peft_model
    dev = xm.xla_device()
    rank, world = xr.global_ordinal(), xr.world_size()
    t = time.time()
    m, hid = load_base(a.base, a.keep)
    model = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', target_modules=TARGET))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    net = Net(model, hid).to(dev)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    if rank == 0:
        log(f'world {world}: base loaded and moved in {time.time() - t:.0f} s, {sum(p.numel() for p in params):,} '
            f'trainable', xm.get_memory_info(dev))
    g = torch.Generator().manual_seed(rank)
    for L in a.lens:
        ids = torch.randint(1000, 30000, (a.batch, L), generator=g).to(dev)
        att = torch.ones(a.batch, L, dtype=torch.long).to(dev)
        y = torch.randint(0, 18, (a.batch,), generator=g).to(dev)
        times = []
        for s in range(a.steps):
            t = time.time()
            opt.zero_grad(set_to_none=True)
            loss = net(ids, att, y)
            loss.backward()
            xm.optimizer_step(opt)
            xm.mark_step()
            xm.wait_device_ops()
            times.append(time.time() - t)
            if rank == 0:
                log(f'L {L} step {s}: {times[-1]:.2f} s, loss {loss.item():.3f}')
        if rank == 0:
            steady = sorted(times[2:])[len(times[2:]) // 2]
            log(f'RESULT L {L} batch {a.batch} x {world} cores: {steady:.2f} s/step, '
                f'{a.batch * L * world / steady:,.0f} tokens/s in all', json.dumps(xm.get_memory_info(dev), default=str))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True); p.add_argument('--keep', type=int, default=24)
    p.add_argument('--nprocs', type=int, default=8); p.add_argument('--lens', default='256,384')
    p.add_argument('--batch', type=int, default=6); p.add_argument('--steps', type=int, default=6)
    a = p.parse_args()
    a.lens = [int(x) for x in a.lens.split(',')]
    import torch_xla.distributed.xla_multiprocessing as xmp
    xmp.spawn(run, args=(a,), nprocs=1 if a.nprocs == 1 else None)
