"""Sibling likelihood for qwen3.8-max (exp052): every train passage scored under the open Qwen3.8-27B chat model (NF4,
lightningv08's public inference assets, the same weights the 0.694 reads), in its no-think assistant turn, with the
per-token outputs ntl-segrun gives for the GLM models (log p of the true token, entropy, E_p[log p ^ 2], rank) on the
first 160 tokens. GLM-5.3-Flash is to GLM-5.3 what Qwen3.8-27B may be to qwen3.8-max: a smaller open sibling.

    python kaggle/glmll/build_q38ll.py          # writes kaggle/glmll/ntl-q38ll/
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
ASSETS = "lightningv08/ntl-olmo2-native24-final-inference-assets"
NAME = "ntl-q38ll"

SCORE = '''import glob, json, os, sys, time
import numpy as np, pandas as pd
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

CUT = 160
PREFIX = "<|im_start|>user\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"   # chat template, no think
path, csv, dest, batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
print("transformers", transformers.__version__, "torch", torch.__version__, flush=True)
df = pd.read_csv(csv, dtype={"id": str})
texts = df.text.fillna("").astype(str).tolist()
tok = AutoTokenizer.from_pretrained(path)
qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=["model.visual", "lm_head", "mtp"])
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(path, quantization_config=qc, dtype=torch.float16,
                                                    device_map="auto").eval()
n4 = 0
for m in model.modules():   # T4 has no bf16 tensor cores: dequantize to fp16 whatever the checkpoint's config says
    if hasattr(m, "compute_dtype"):
        m.compute_dtype = torch.float16
        n4 += 1
print(f"{n4} 4-bit linears, compute fp16", flush=True)
print(f"loaded in {time.time() - t0:.0f} s; devices {sorted(set(str(p.device) for p in model.parameters()))}", flush=True)
pre = tok(PREFIX, add_special_tokens=False).input_ids
enc = [pre + tok(t, add_special_tokens=False).input_ids[:CUT] for t in texts]
P = len(pre)
pad = tok.pad_token_id if tok.pad_token_id is not None else 248044
dev = model.get_input_embeddings().weight.device
out = [None] * len(enc)
order = np.argsort([len(e) for e in enc], kind="stable")
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        L = max(len(enc[i]) for i in idx)
        x = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :len(enc[i])] = torch.tensor(enc[i])
            att[r, :len(enc[i])] = 1
        logits = model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits[:, P - 1:-1].float()
        lpa = torch.log_softmax(logits, -1)
        del logits
        tgt = x[:, P:].to(lpa.device)
        lp = lpa.gather(-1, tgt[..., None])[..., 0]
        p = lpa.exp()
        h = -(p * lpa).sum(-1)
        m2 = (p * lpa * lpa).sum(-1)
        rank = (lpa > lp[..., None]).sum(-1).float()
        del p, lpa
        s = torch.stack([lp, h, m2, rank], -1).cpu().numpy().astype(np.float32)
        for r, i in enumerate(idx):
            out[i] = s[r, :len(enc[i]) - P]
        if b % (40 * batch) == 0:
            print(f"{b + len(idx)}/{len(order)} texts, {time.time() - t0:.0f} s", flush=True)
off = np.concatenate([[0], np.cumsum([len(s) for s in out])])
allsc = np.concatenate(out)
bad = int((~np.isfinite(allsc)).any(1).sum())
np.savez(dest, ids=np.array(df.id.tolist(), dtype="U64"), offsets=off, scores=allsc)
print(f"{len(texts)} texts, {len(allsc)} tokens, non-finite rows {bad}, mean lp {allsc[:, 0].mean():.3f}, "
      f"{time.time() - t0:.0f} s", flush=True)
'''

RUN = f'''import glob, os, subprocess, sys, time
import pandas as pd
ROOT = next(p for p in glob.glob('/kaggle/input/**/name-that-llm-artifacts', recursive=True) if os.path.isdir(p + '/qwen38-27b-nf4'))
WHEELS = sorted(glob.glob(ROOT + '/qwen38-offline-linux/*.whl'))
print(ROOT, [os.path.basename(w) for w in WHEELS], flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--disable-pip-version-check',
                '--target', '/kaggle/temp/q38py', *WHEELS], check=True)
TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
os.makedirs('/kaggle/working/out', exist_ok=True)
env = dict(os.environ, PYTHONPATH='/kaggle/temp/q38py')
t = time.time()
subprocess.run([sys.executable, '/tmp/q38score.py', ROOT + '/qwen38-27b-nf4', TRAIN, '/kaggle/working/out/q38_scores.npz', '4'],
               env=env, check=True)
print(f'done in {{time.time() - t:.0f}} s', flush=True)
'''


def build():
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              f"# {NAME}\n\nPer-token likelihood of every train passage under the open Qwen3.8-27B chat model (NF4, no-think "
              "assistant turn, first 160 tokens): a sibling likelihood for qwen3.8-max (exp052)."},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": "%%writefile /tmp/q38score.py\n" + SCORE},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": RUN}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = HERE / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [ASSETS], "competition_sources": ["name-that-llm"],
            "kernel_sources": [], "model_sources": [], "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4",
            "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {NAME}: {len(cells)} cells")


if __name__ == "__main__":
    build()
