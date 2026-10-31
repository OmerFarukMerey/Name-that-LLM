# CPU-only timing probe (no GPU quota), upstream nlm-v27's Docker image: load Lucas Boesen's GLM-4-9B trunc25 (the
# complementary stack's glm9b backbone) from its notebook-output mount (mode=mount) or after one copy to /tmp (copy).
import glob, os, shutil, time, torch
from transformers import AutoModel
MODE = 'mount'
P = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/config.json', recursive=True) if 'glm4_9b_trunc25' in p))
files = sorted(os.listdir(P)); print(MODE, P, files, sum(os.path.getsize(os.path.join(P, f)) for f in files) / 1e9, 'GB', flush=True)
t0 = time.time()
if MODE == 'copy':
    shutil.copytree(P, '/tmp/glm4_9b_trunc25', dirs_exist_ok=True); P = '/tmp/glm4_9b_trunc25'
    print(f'copy {time.time() - t0:.0f} s', flush=True)
t = time.time()
m = AutoModel.from_pretrained(P, dtype=torch.float16).eval()
print(f'{MODE}: from_pretrained {time.time() - t:.0f} s, total {time.time() - t0:.0f} s, {sum(p.numel() for p in m.parameters())/1e9:.2f} B params', flush=True)
# safetensors maps fp16 checkpoints lazily: touch every parameter, as the copy to the GPUs does
t = time.time(); acc = 0.0
for i, (n, q) in enumerate(m.named_parameters()):
    acc += float(q.data.view(-1)[::997].float().sum()) + float(q.data.sum())
    if i % 100 == 0: print(f'  touched {i} tensors, {time.time() - t:.0f} s', flush=True)
print(f'{MODE}: touched all parameters in {time.time() - t:.0f} s, total {time.time() - t0:.0f} s ({acc:.3e})', flush=True)
