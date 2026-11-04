# ---- driver cell (inserted by kaggle/member/build_tpu_member.py, which fills CONFIG)
# Reload check of TPU-trained members on the T4: infer.py's own l24_predict path (fp16 base truncated to `keep` layers,
# LoRA merged into the weights, masked mean pool of 8 depths, LayerNorms, plain head) on the first 40 official rows,
# against the trainer's own bf16 predictions (check40.npy) and the true labels.
import glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

CONFIG = None
T0 = time.time()
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train = pd.read_csv(DATA + '/train.csv').iloc[:40]
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
L = [2, 4, 6, 8, 12, 16, 20, 24]
ok = True
for mem in CONFIG["members"]:
    base = [os.path.dirname(c) for c in cfgs if all(p in c.lower() for p in mem["base_parts"])][0]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(base); cfg.num_hidden_layers = mem["keep"]; cfg.use_cache = False
    m = AutoModel.from_pretrained(base, config=cfg, dtype=torch.float16, device_map='auto').eval()
    mods = dict(m.named_modules())
    for tag in mem["dirs"]:
        ad = os.path.dirname(glob.glob(f'/kaggle/input/**/{tag}/heads.pt', recursive=True)[0])
        acfg = json.load(open(os.path.join(ad, 'adapter_config.json'))); scale = acfg['lora_alpha'] / acfg['r']
        sd = load_file(os.path.join(ad, 'adapter_model.safetensors')); deltas = []
        for k, v in sd.items():
            if '.lora_A.' not in k:
                continue
            name = k.replace('base_model.model.', '', 1).split('.lora_A.')[0]
            if name not in mods and ('model.' + name) in mods:
                name = 'model.' + name
            B = sd[k.replace('.lora_A.', '.lora_B.')]; W = mods[name].weight
            with torch.no_grad():
                W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)
            deltas.append((W, B, v))
        H = torch.load(os.path.join(ad, 'heads.pt'), map_location='cpu'); lab = json.load(open(os.path.join(ad, 'meta.json')))['labels']
        hid = H['norms.0.weight'].shape[0]; norms = [torch.nn.LayerNorm(hid) for _ in L]
        for i, nm in enumerate(norms):
            nm.load_state_dict({'weight': H[f'norms.{i}.weight'], 'bias': H[f'norms.{i}.bias']})
        enc = tok(list(train.text), truncation=True, max_length=384)['input_ids']
        Z = []
        with torch.no_grad():
            for e in enc:
                ids = torch.tensor([e]).to(m.device); att = torch.ones_like(ids)
                hs = m(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states
                if not all(torch.isfinite(hs[l]).all() for l in L):
                    m.to(torch.bfloat16); hs = m(input_ids=ids, attention_mask=att, output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16)
                mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
                Z.append(torch.cat([nm(((hs[l].float() * mask.to(hs[l].device)).sum(1) / mask.sum(1).to(hs[l].device)).cpu()) for l, nm in zip(L, norms)], -1))
        Z = torch.cat(Z).numpy()
        z = Z @ H['plain.weight'].float().numpy().T + H['plain.bias'].float().numpy()
        lp = z - z.max(1, keepdims=True); lp -= np.log(np.exp(lp).sum(1, keepdims=True))
        ref = np.load(os.path.join(ad, 'check40.npy'))
        y = train.model.map({l: i for i, l in enumerate(lab)}).values
        agree = float((lp.argmax(1) == ref.argmax(1)).mean()); acc = float((lp.argmax(1) == y).mean())
        print(f"{tag}: merged {len(deltas)} LoRA pairs; argmax agreement with the TPU's own predictions {agree:.3f}; "
              f"in-sample accuracy GPU {acc:.3f} / TPU {float((ref.argmax(1) == y).mean()):.3f}; "
              f"max |log-prob diff| {float(np.abs(lp - ref).max()):.3f}", flush=True)
        ok &= agree >= 0.9
        for W, B, A_ in deltas:
            with torch.no_grad():
                W -= ((B.to(W.device, torch.float32) @ A_.to(W.device, torch.float32)) * scale).to(W.dtype)
    del m; torch.cuda.empty_cache()
print('CHECK', 'PASS' if ok else 'FAIL', f'{time.time() - T0:.0f} s', flush=True)
assert ok
