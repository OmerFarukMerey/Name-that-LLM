"""Build ntl-l24-dp-diag: where an l24 batch's time goes on Kaggle's T4 x2 (exp018 follow-up).

    python kaggle/nlm/build_dp_diag.py

ntl-l24-dp-check found the copy-per-GPU l24_predict exact but only 1.14-1.17x faster, and Llama-3.1-8B (13% more
FLOPs than Qwen3-8B per token) no slower than Qwen3-8B, so something besides the GPU sets the pace. On 256 train rows
in upstream's length order, Llama-3.1-8B truncated to 24 layers (no adapter: the timing does not depend on it), batch
16, upstream's pooling, this times five ways of running the same batches:
  split_cpu  upstream: one copy split over both GPUs, masked means on the CPU (8 x ~100 MB copies per batch)
  split_gpu  the same copy, masked means on the GPU (8 x 64 KB copies)
  one_cpu    one full copy on GPU 0 alone, masked means on the CPU
  dp_cpu     a copy per GPU, alternate batches in two threads, masked means on the CPU
  dp_gpu     a copy per GPU, alternate batches in two threads, masked means on the GPU
and reports seconds per batch, the GPU-busy share (CUDA events around the forward pass), allocator retries and the
largest feature difference from split_cpu.
Output: kaggle/nlm/ntl-l24-dp-diag/ (notebook + kernel-metadata.json).
"""
import json

import build_nlm as B

KERNEL = "ntl-l24-dp-diag"
CODE = r'''import glob, os, json, shutil, time, gc, threading
import numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoConfig, AutoModel, AutoTokenizer
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False   # as infer.py
cfgs = sorted(glob.glob('/kaggle/input/**/config.json', recursive=True))
BL = next(os.path.dirname(x) for x in cfgs if '/llama-3.1/' in x.lower() and '/8b/' in x.lower())
_c = time.time(); shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'
print('llama copied in', round(time.time() - _c), 's', torch.__version__, os.cpu_count(), 'cpus', flush=True)
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
texts = pd.read_csv(DATA + '/train.csv').text.values[np.random.RandomState(0).choice(2699, 256, replace=False)]
tok = AutoTokenizer.from_pretrained(BL); tok.pad_token = tok.pad_token or tok.eos_token
enc = [x for x in tok(list(texts), truncation=True, max_length=384)['input_ids']]
order = np.argsort([len(x) for x in enc]); bs = 16; L = [2, 4, 6, 8, 12, 16, 20, 24]
batches = [order[i:i + bs] for i in range(0, len(order), bs)]
print('tokens per row: mean', np.mean([len(x) for x in enc]).round(1), 'max', max(len(x) for x in enc), flush=True)
cfg = AutoConfig.from_pretrained(BL); cfg.num_hidden_layers = 24; cfg.use_cache = False
def load(dm): return AutoModel.from_pretrained(BL, config=cfg, dtype=torch.float16, device_map=dm).eval()
norms = [torch.nn.LayerNorm(4096) for _ in L]   # stand-in heads' LayerNorms (default init; same for every mode)

@torch.no_grad()
def run(mm, bl, Z, gpu_mean, stats):
    dev = next(mm.parameters()).device
    for b in bl:
        Lm = max(len(enc[j]) for j in b)
        ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)
        for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1
        t0 = time.time(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(torch.cuda.current_stream(dev))
        hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states
        e1.record(torch.cuda.current_stream(dev))
        assert all(torch.isfinite(hs[l]).all() for l in L)
        t1 = time.time()
        mask = att.unsqueeze(-1).float(); mask[:, 0] = 0
        if gpu_mean:
            mg = mask.to(hs[L[0]].device)
            Z[b] = torch.cat([nm(((hs[l].float() * mg.to(hs[l].device)).sum(1) / mg.sum(1).to(hs[l].device)).cpu()) for l, nm in zip(L, norms)], -1).numpy()
        else:
            Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()
        t2 = time.time()
        stats.append((t2 - t0, e0.elapsed_time(e1) / 1000, t2 - t1))
        del hs

def mode(name, models, gpu_mean):
    Z = np.zeros((len(texts), 4096 * len(L)), np.float32); stats = [[] for _ in models]
    for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
    r0 = [torch.cuda.memory_stats(d).get('num_alloc_retries', 0) for d in range(torch.cuda.device_count())]
    for mm in models: run(mm, batches[:2], Z, gpu_mean, [])   # warm-up on every copy, not timed
    t = time.time()
    if len(models) == 1: run(models[0], batches, Z, gpu_mean, stats[0])
    else:
        with ThreadPoolExecutor(len(models)) as ex:
            for f in [ex.submit(run, models[k], batches[k::len(models)], Z, gpu_mean, stats[k]) for k in range(len(models))]: f.result()
    wall = time.time() - t
    retries = [torch.cuda.memory_stats(d).get('num_alloc_retries', 0) - r0[d] for d in range(torch.cuda.device_count())]
    peak = [round(torch.cuda.max_memory_allocated(d) / 2**30, 2) for d in range(torch.cuda.device_count())]
    s = np.array([x for st in stats for x in st])
    res = dict(wall=round(wall, 1), per_batch=round(wall / len(batches), 2), mean_batch_wall=round(float(s[:, 0].mean()), 2),
               mean_forward_gpu=round(float(s[:, 1].mean()), 2), mean_pool=round(float(s[:, 2].mean()), 2),
               retries=retries, peak_gib=peak)
    print(name, json.dumps(res), flush=True)
    return Z, res

R, F = {}, {}
m = load('auto'); print('split', {str(p.device) for p in m.parameters()}, flush=True)
for name, g in (('split_cpu', False), ('split_gpu', True)):
    F[name], R[name] = mode(name, [m], g)
del m; gc.collect(); torch.cuda.empty_cache()
ms = [load({'': 0}), load({'': 1})]
F['one_cpu'], R['one_cpu'] = mode('one_cpu', ms[:1], False)
for name, g in (('dp_cpu', False), ('dp_gpu', True)):
    F[name], R[name] = mode(name, ms, g)
for name in F:
    R[name]['maxdiff_vs_split_cpu'] = float(np.abs(F[name] - F['split_cpu']).max())
print('DP DIAG DONE', json.dumps(R), flush=True)
json.dump(R, open('/kaggle/working/dp_diag.json', 'w'), indent=1)
'''


def main():
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": f"# {KERNEL}\n\nexp018 follow-up: where an l24 "
                     "batch's time goes on T4 x2 (split vs a copy per GPU, CPU vs GPU masked means). Private; no submission."},
                    {"cell_type": "code", "metadata": {}, "source": CODE, "execution_count": None, "outputs": []}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    meta_in = json.loads((B.HERE / "upstream_nlm_v27.meta.json").read_text(encoding="utf-8"))
    out = B.HERE / KERNEL
    out.mkdir(exist_ok=True)
    (out / f"{KERNEL}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{KERNEL}", "title": KERNEL, "code_file": f"{KERNEL}.ipynb", "language": "python",
            "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": [], "kernel_sources": [B.LLAMA_CACHE], "competition_sources": ["name-that-llm"],
            "docker_image": meta_in["docker_image"], "machine_shape": meta_in["machine_shape"], "is_private": True,
            "keywords": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    compile(CODE, KERNEL, "exec")
    print(f"built {KERNEL}")


if __name__ == "__main__":
    main()
