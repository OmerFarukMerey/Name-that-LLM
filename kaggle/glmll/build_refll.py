"""Reference likelihoods for exp050: every train passage scored under Qwen3-4B-Base and Qwen3-8B-Base (attached to the
0.694 offline), with the per-token outputs ntl-segrun gives for the GLM models (log p of the true token, entropy,
E_p[log p ^ 2], rank), on the first 160 tokens. GLM-5.3 imitates Claude, so Claude passages are likely under GLM too;
the ratio to a neutral model separates "likely under GLM" from "easy to predict".

    python kaggle/glmll/build_refll.py          # writes kaggle/glmll/ntl-refll/
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
MODELS = ["qwen-lm/qwen-3/Transformers/4b-base/1", "qwen-lm/qwen-3/Transformers/8b-base/1"]
NAME = "ntl-refll"

REFLL = '''import glob, json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CUT = 160


def score_texts(path, texts, batch=8):
    """Per text an (n, 4) float32 array: log p(true token), entropy, E_p[log p ^ 2], rank (tokens above the true one),
    for the first CUT tokens after an <|endoftext|> context token."""
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto").eval()
    bos = tok.convert_tokens_to_ids("<|endoftext|>")
    enc = [[bos] + tok(t, add_special_tokens=False).input_ids[:CUT] for t in texts]
    dev = model.get_input_embeddings().weight.device
    out = [None] * len(enc)
    order = np.argsort([len(e) for e in enc], kind="stable")
    with torch.no_grad():
        for b in range(0, len(order), batch):
            idx = order[b:b + batch]
            L = max(len(enc[i]) for i in idx)
            x = torch.full((len(idx), L), bos, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                x[r, :len(enc[i])] = torch.tensor(enc[i])
                att[r, :len(enc[i])] = 1
            logits = model(input_ids=x.to(dev), attention_mask=att.to(dev)).logits[:, :-1].float()
            lpa = torch.log_softmax(logits, -1)
            del logits
            lp = lpa.gather(-1, x[:, 1:].to(lpa.device)[..., None])[..., 0]
            p = lpa.exp()
            h = -(p * lpa).sum(-1)
            m2 = (p * lpa * lpa).sum(-1)
            rank = (lpa > lp[..., None]).sum(-1).float()
            del p, lpa
            s = torch.stack([lp, h, m2, rank], -1).cpu().numpy().astype(np.float32)
            for r, i in enumerate(idx):
                out[i] = s[r, :len(enc[i]) - 1]
    del model
    torch.cuda.empty_cache()
    return out
'''

RUN = '''import glob, json, os, time
import numpy as np, pandas as pd
exec(open('/tmp/refll.py').read())
TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
df = pd.read_csv(TRAIN)
ids, texts = df.id.astype(str).tolist(), df.text.fillna('').astype(str).tolist()
os.makedirs('/kaggle/working/out', exist_ok=True)
for tag, size in (('q4b', '4b-base'), ('q8b', '8b-base')):
    path = os.path.dirname(sorted(glob.glob(f'/kaggle/input/**/{size}/**/config.json', recursive=True))[0])
    t = time.time()
    sc = score_texts(path, texts)
    off = np.concatenate([[0], np.cumsum([len(s) for s in sc])])
    allsc = np.concatenate(sc)
    bad = int((~np.isfinite(allsc)).any(1).sum())
    np.savez(f'/kaggle/working/out/{tag}_scores.npz', ids=np.array(ids, dtype='U64'), offsets=off, scores=allsc)
    print(f'{tag}: {path}, {len(texts)} texts, {len(allsc)} tokens, non-finite rows {bad}, {time.time() - t:.0f} s', flush=True)
'''


def build():
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              f"# {NAME}\n\nPer-token likelihood of every train passage under Qwen3-4B-Base and Qwen3-8B-Base (first 160 "
              "tokens): a neutral reference for the GLM-5.3 likelihood features of exp050."},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": "%%writefile /tmp/refll.py\n" + REFLL},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": RUN}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = HERE / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": [], "competition_sources": ["name-that-llm"],
            "kernel_sources": [], "model_sources": MODELS, "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4",
            "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {NAME}: {len(cells)} cells")


if __name__ == "__main__":
    build()
