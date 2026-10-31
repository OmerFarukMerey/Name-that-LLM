# CPU-only timing probe (no GPU quota), upstream nlm-v27's Docker image: load Llama-3.1-8B (24 of 32 layers, fp16, as
# infer.py's l24 loader) from avazaku's notebook-output mount. mode=mount: from_pretrained straight from the mount
# (what nlm-v27 does; ~2,140 s in our GPU commit run). mode=copy: one sequential copy to local disk, then load.
import glob, os, shutil, time, torch
from transformers import AutoConfig, AutoModel
MODE = 'mount'
BL = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if '/llama-3.1/' in p and '/8b' in p))
print(MODE, BL, sorted(os.listdir(BL)), flush=True)
t0 = time.time()
if MODE == 'copy':
    shutil.copytree(BL, '/tmp/llama8b', dirs_exist_ok=True); BL = '/tmp/llama8b'
    print(f'copy {time.time() - t0:.0f} s', flush=True)
cfg = AutoConfig.from_pretrained(BL); cfg.num_hidden_layers = 24
t = time.time()
m = AutoModel.from_pretrained(BL, config=cfg, dtype=torch.float16).eval()
n = sum(p.numel() for p in m.parameters())
print(f'{MODE}: from_pretrained {time.time() - t:.0f} s, total {time.time() - t0:.0f} s, {n/1e9:.2f} B params, '
      f'checksum {float(sum(p.float().abs().sum() for p in m.parameters())):.6e}', flush=True)
