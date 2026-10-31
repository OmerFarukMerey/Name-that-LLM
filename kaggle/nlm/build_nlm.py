"""Build our private forks of Dhruv Tibarewal's nlm-v27 (public 0.651), and since 2026-10-28 of mercurydust's 0.660
notebook ("V31" on Roman Tamrazov's 0.654; see BASES and the exp025 block).

    python kaggle/nlm/build_nlm.py [kernel ...]        (default: every variant)

upstream_nlm_v27.ipynb is version 1 of dhruvtibarewal/nlm-v27, pulled 2026-10-25 23:45 UTC and kept byte-for-byte
(sha256 below); `kernels pull` returns whatever the latest version is, so never refresh it from the API.

nlm-v27 mounts metaresearch/llama-3.1/Transformers/8b/2 for its ll8v5 member (5 fold LoRA24 adapters plus a full
fit, weight 1.0 of the six members). That model is gated and this account gets 403 on it.
  ntl-nlm27-nollama   cell 0 no longer looks up the Llama base (an assert fails without it), and MODELS loses its
                      ll8v5 entry. infer.py sums the members' weighted log-probabilities, so the other five keep their
                      weights and the blend is a little less sharp (total member weight 10.5 -> 9.5). Scored 0.643.
  ntl-nlm27-mb        Llama-3.1-8B comes back from avazaku's public "Verified Llama Asset Cache" notebook output
                      (NousResearch's ungated copy of meta-llama/Llama-3.1-8B, hash-checked there, Llama 3.1 Community
                      License; the user chose this source on 2026-10-26). It is the input of avazaku's 0.651
                      reproduction, which ran on the same updated datasets as our forks. On top:
                      - Dhruv's ModernBERT-large two-stage member (dhruvtibarewal/nlm-enc-v5, weight 1.5), as in his
                        newer "Name That LLM Fully Reproducible" notebook (+0.6 OOF there);
                      - ATLAS keeps the rows it has scored when the time budget runs short (upstream drops ATLAS
                        entirely when the projected finish passes the budget);
                      - a guard for any em dash: gemini-3.1-pro-preview, gemini-3.6-flash and gemini-3.8-flash never
                        use one (0 of 1,052 train and 42,968 pool passages; experiments/exp012_typography_guards.py);
                      - the TF-IDF member, whose weight is 0 (--tfidf_w 0), is no longer fitted (about 6 min).
The commit run is a full end-to-end run on the 3 public test rows (about 2.5 GPU h), so it exercises every change
before the hidden rerun. Output: kaggle/nlm/<kernel>/ (notebook + kernel-metadata.json).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE, BASE_SHA256 = "upstream_nlm_v27.ipynb", "c57238c44d772ad591ae527262dc3f2d63ab92d32c2c7fb31e4d3ab5f9a5076f"
LLAMA = "metaresearch/llama-3.1/Transformers/8b/2"
LLAMA_CACHE = "avazaku/name-that-llm-verified-llama-asset-cache"
MODERNBERT = "dhruvtibarewal/nlm-enc-v5"
LL8V5 = ('  {"name": "ll8v5", "type": "l24", "path": BL, "adapters": ALL8V5, "keep": 24, "layers": L, "w": 1.0, '
         '"bs": 16, "max_len": 384, "twin": 0.35},\n')
NO_LLAMA = [  # (old, new): each old string must occur exactly once in the code cells
    ("BL = mdir('/llama-3.1/', '/8b/')", "BL = None  # fork: Llama-3.1-8B is gated for this account, so ll8v5 is dropped"),
    (LL8V5, ""),
]

# sizes of the four weight shards as avazaku's cache notebook verified them (its PROTOCOL)
SHARDS = {"model-00001-of-00004.safetensors": 4976698672, "model-00002-of-00004.safetensors": 4999802720,
          "model-00003-of-00004.safetensors": 4915916176, "model-00004-of-00004.safetensors": 1168138808}
MIRROR_LLAMA = (
    "BL = mdir('/llama-3.1/', '/8b/')",
    "BL = mdir('/llama-3.1/', '/8b/')\n"
    f"_SHARDS = {SHARDS!r}\n"
    "assert 'name-that-llm-verified-llama-asset-cache' in BL and all(os.path.getsize(os.path.join(BL, f)) == n "
    "for f, n in _SHARDS.items()), BL   # fork: Llama-3.1-8B from avazaku's public cache (NousResearch copy)")
# from_pretrained straight from the notebook-output mount takes ~2,050 s for Llama-3.1-8B; one sequential copy to local
# disk takes ~130 s and the load from there ~60 s (CPU probes ntl-llama-load-mount / -copy on upstream's image,
# 2026-10-27). Same bytes, so the same predictions; the ~31 min go to ATLAS's fixed time budget. (The comp stack's
# GLM-4-9B, one fp16 file, reads fast from its mount: ntl-glm-load-*, so it is left alone.)
LLAMA_COPY = [("print('llama', BL, AL)\n",
               "import shutil, time as _tc; _c = _tc.time()\n"
               "shutil.copytree(BL, '/tmp/llama-3.1/8b', dirs_exist_ok=True); BL = '/tmp/llama-3.1/8b'   # fork: see the intro\n"
               "print('llama copied to local disk in', round(_tc.time() - _c), 's')\n"
               "print('llama', BL, AL)\n")]
ADD_MODERNBERT = [
    ("import shutil; shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/kaggle/working/deberta_v3_large'\n",
     "import shutil; shutil.copytree(DEB, '/kaggle/working/deberta_v3_large', dirs_exist_ok=True); DEB = '/kaggle/working/deberta_v3_large'\n"
     "MB = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/meta.json', recursive=True) if '/nlm-enc-v5/' in p))\n"
     "shutil.copytree(MB, '/tmp/mb_v5', dirs_exist_ok=True); MB = '/tmp/mb_v5'   # fork: Dhruv's ModernBERT-large member\n"
     "print('modernbert', MB, sorted(os.listdir(MB)))\n"),
    ('  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]',
     '  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n  {"name": "mb", "type": "enc", "ckpt": MB, "w": 1.5},\n]'),
]
SKIP_ZERO_TFIDF = [
    ("for kw in [dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000),",
     "for kw in [] if a.perpassage and a.tfidf_w == 0 else [dict(analyzer='char_wb', ngram_range=(2, 5), min_df=3, max_features=60000),"),
    ("if a.perpassage:\n    LP['tfidf'] = ",
     "if a.perpassage and a.tfidf_w == 0:   # fork: weight 0 multiplied its log-probabilities by 0, so skip the fit\n"
     "    log('tfidf skipped (weight 0)')\nelif a.perpassage:\n    LP['tfidf'] = "),
]
ATLAS_PARTIAL = [
    ("        for _b in range(0, len(_ord), a.atlas_bs):\n",
     "        _done = 0\n        for _b in range(0, len(_ord), a.atlas_bs):\n"),
    ("                if _done > 50 and _proj > budget: raise RuntimeError('atlas would overrun the time budget')\n",
     "                if _done > 50 and _proj > budget:   # fork: keep the rows scored so far instead of dropping ATLAS\n"
     "                    log(f'atlas stops early: {_done}/{len(_ord)} rows, projected finish {_proj:.0f}s > budget {budget:.0f}s'); break\n"),
    ("        assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-4)\n"
     "        return out[:, [_alab.index(l) for l in LABELS]]\n",
     "        _got = np.sort(_ord[:_done]); out = out[_got]   # fork: positions in `rows` that were scored\n"
     "        assert np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-4)\n"
     "        return _got, out[:, [_alab.index(l) for l in LABELS]]\n"),
    ("                A_ = atlas_scores(_bd, rows, a.atlas_budget)\n"
     "                mixed[rows] = mixed[rows] - _w * P[rows] + _w * A_; log(f'atlas {_bd} mixed at {_w}', gpu_mem())\n",
     "                _pos, A_ = atlas_scores(_bd, rows, a.atlas_budget); _ar = rows[_pos]\n"
     "                mixed[_ar] = mixed[_ar] - _w * P[_ar] + _w * A_; log(f'atlas {_bd} mixed at {_w} on {len(_ar)}/{len(rows)} rows', gpu_mem())\n"),
]
# ATLAS_PARTIAL stops at the first checkpoint whose projection overruns, which can be after 104 rows with most of the
# time unused. ATLAS_DEADLINE keeps scoring (shortest rows first) until the budget itself runs out, so with time to
# spare it is upstream's ATLAS exactly, and short of time it keeps as many rows as fit instead of none.
ATLAS_DEADLINE = [
    ("        for _b in range(0, len(_ord), a.atlas_bs):\n",
     "        _done = 0\n        for _b in range(0, len(_ord), a.atlas_bs):\n"
     "            if time.time() - T0 > budget:   # fork: stop at the deadline, keep the rows scored so far\n"
     "                log(f'atlas stops at its deadline: {_done}/{len(_ord)} rows'); break\n"),
    ("                if _done > 50 and _proj > budget: raise RuntimeError('atlas would overrun the time budget')\n", ""),
] + ATLAS_PARTIAL[2:]
EM_DASH_ALLOWED = ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5',
                   'gemini-3.7-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol',
                   'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
EM_DASH_GUARD = [
    ("'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'kimi-k3'})]\n",
     "'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'kimi-k3'}),\n"
     "          ('\\u2014', {" + ", ".join(repr(m) for m in sorted(EM_DASH_ALLOWED)) + "})]"
     "   # fork: any em dash (exp012: 0 violations in train and the 120k pool)\n"),
]


# The commit run sees only the 3 public test rows, and running the full pipeline on them costs ~2.5 GPU h of the
# weekly quota. With FAST_COMMIT the commit writes a placeholder instead; Kaggle's hidden rerun (~4,000 rows) runs
# every cell unchanged. Only for variants whose code paths an earlier full commit (or a flag) already covers.
DATA_LINE = ("DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] "
             "if os.path.exists(p + '/train.csv'))\n")
FAST_COMMIT = [
    (DATA_LINE, DATA_LINE + "import pandas as _pd\n"
     "FAST_COMMIT = len(_pd.read_csv(DATA + '/test.csv')) < 50   # fork: the 3-row commit run skips the heavy cells; "
     "the hidden rerun runs everything\nprint('FAST_COMMIT', FAST_COMMIT)\n"),
    ("!python comp_stack.py 2>&1 | grep -vE 'Warning|warn'\n",
     "if not FAST_COMMIT:\n    !python comp_stack.py 2>&1 | grep -vE 'Warning|warn'\n"),
    ("!python infer.py --data {DATA}",
     "if FAST_COMMIT:   # fork: placeholder from the commit run; Kaggle reruns the notebook on the hidden test\n"
     "    _test_ph = pd.read_csv(DATA + '/test.csv')\n"
     "    pd.DataFrame({'id': _test_ph.id, 'model': 'gpt-5.5'}).to_parquet('submission.parquet', index=False)\n"
     "else:\n    !python infer.py --data {DATA}"),
]
# ATLAS (10% mix of a source-conditioned likelihood on rows with top-2 margin < 0.3) has to finish inside a fixed
# 8.3 h budget, else it is dropped whole; on the hidden test it is close to that edge. Off = a deterministic run.
ATLAS_OFF = [("--atlas_w 0.10 ", "--atlas_w 0 ")]

# exp018: l24_predict loads each member with device_map='auto', which splits even a 24-layer 8B model (~11 GiB in fp16)
# across both T4s, so one GPU idles while the other runs its half of the layers, and it pools on the CPU: per batch
# eight ~100 MB hidden-state copies leave the GPU (23% of a batch's time, ntl-l24-dp-diag). With "dp" in a member's spec
# it loads a full copy on each GPU (Ministral 3's unused image encoder dropped; if the largest batch fits with 1 GiB
# to spare, else upstream's split copy) and runs upstream's batches, in upstream's order, alternately on the two copies
# in two threads; every member pools on the GPU. On Llama-3.1-8B (256 train rows): 1.81 s per batch upstream, 1.47 s
# pooling on the GPU, 1.08 s with a copy per GPU too (1.68x); features within 4e-6 of upstream's. q8v5, ll8v5 and
# mn8v1 fit a T4; the 14B members and ATLAS's 36-layer Qwen3-8B do not.
L24_DP_FN = [
    ("    from safetensors.torch import load_file\n    adapters = spec.get('adapters') or [spec['adapter']]\n",
     "    from safetensors.torch import load_file\n    from concurrent.futures import ThreadPoolExecutor\n"
     "    adapters = spec.get('adapters') or [spec['adapter']]\n"),
    ("    m = AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()\n"
     "    if spec.get('no_final_norm'): m.norm = torch.nn.Identity()\n"
     "    mods = dict(m.named_modules()); first = next(m.parameters()).device; L = spec['layers']\n",
     "    _ms = []   # fork (exp018): with spec['dp'], a full copy on each GPU if the largest batch fits with 1 GiB to spare\n"
     "    if spec.get('dp') and torch.cuda.device_count() >= 2:\n"
     "        try:\n"
     "            for _d in range(2):\n"
     "                _ms.append(AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map={'': _d}).eval())\n"
     "                for _n in ('vision_tower', 'multi_modal_projector'):   # Ministral 3's image path; text passes never use it\n"
     "                    if getattr(_ms[-1], _n, None) is not None: setattr(_ms[-1], _n, None)\n"
     "                gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(_d)\n"
     "                _ms[-1](input_ids=torch.full((spec.get('bs', 16), spec.get('max_len', 384)), tok.pad_token_id, device=f'cuda:{_d}'), output_hidden_states=True, use_cache=False)\n"
     "                assert torch.cuda.max_memory_allocated(_d) < torch.cuda.get_device_properties(_d).total_memory - (1 << 30), 'under 1 GiB to spare'\n"
     "        except Exception as _e:   # out of memory, or anything else: upstream's way\n"
     "            log(f'  l24 {spec[\"name\"]}: one split copy instead of one per GPU ({_e!r:.80})'); _ms = []\n"
     "        gc.collect(); torch.cuda.empty_cache()\n"
     "    if not _ms: _ms = [AutoModel.from_pretrained(spec['path'], config=cfg, dtype=torch.float16, device_map='auto').eval()]\n"
     "    m = _ms[0]\n"
     "    if spec.get('no_final_norm'):\n"
     "        for _m in _ms: _m.norm = torch.nn.Identity()\n"
     "    mods = dict(m.named_modules()); first = next(m.parameters()).device; L = spec['layers']\n"
     "    _mods = [dict(_m.named_modules()) for _m in _ms]\n"),
    ("    P_acc, C_acc = 0.0, 0.0; FEATS = None; OVF = [0]\n", "    P_acc, C_acc = 0.0, 0.0; FEATS = None; OVF = [0, 0]\n"),
    ("        for i in range(0, len(order_), bs):\n"
     "            b = order_[i:i + bs]; Lm = max(len(enc_[j]) for j in b)\n"
     "            ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)\n"
     "            for r, j in enumerate(b): ids[r, :len(enc_[j])] = torch.tensor(enc_[j]); att[r, :len(enc_[j])] = 1\n"
     "            hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states\n"
     "            if not all(torch.isfinite(hs[l]).all() for l in L):\n"
     "                m.to(torch.bfloat16); hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16); OVF[0] += 1\n"
     "            mask = att.unsqueeze(-1).float(); mask[:, 0] = 0\n"
     "            Z_[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()\n"
     "        return Z_\n",
     "        def _run(k, n):   # fork: upstream's batches in upstream's order, batch i on copy i % n, one thread per copy\n"
     "            mm = _ms[k]; dev = next(mm.parameters()).device\n"
     "            with torch.no_grad():   # grad mode is per thread\n"
     "                for i in range(k * bs, len(order_), bs * n):\n"
     "                    b = order_[i:i + bs]; Lm = max(len(enc_[j]) for j in b)\n"
     "                    ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)\n"
     "                    for r, j in enumerate(b): ids[r, :len(enc_[j])] = torch.tensor(enc_[j]); att[r, :len(enc_[j])] = 1\n"
     "                    hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states\n"
     "                    if not all(torch.isfinite(hs[l]).all() for l in L):   # fp16 overflow (un-normed deep states): redo this batch in bf16\n"
     "                        mm.to(torch.bfloat16); hs = mm(input_ids=ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True, use_cache=False).hidden_states; mm.to(torch.float16); OVF[k] += 1\n"
     "                    mask = att.unsqueeze(-1).float(); mask[:, 0] = 0; _mg = {d_: mask.to(d_) for d_ in {hs[l].device for l in L}}\n"
     "                    Z_[b] = torch.cat([nm(((hs[l].float() * _mg[hs[l].device]).sum(1) / _mg[hs[l].device].sum(1)).cpu()) for l, nm in zip(L, norms)], -1).numpy()   # fork: masked means on the GPU\n"
     "        if len(_ms) == 1: _run(0, 1)\n"
     "        else:\n"
     "            try:\n"
     "                with ThreadPoolExecutor(len(_ms)) as _ex:\n"
     "                    for _f in [_ex.submit(_run, k, len(_ms)) for k in range(len(_ms))]: _f.result()\n"
     "            except Exception as _e:   # never worse than one GPU: drop the second copy, every batch again on the first\n"
     "                log(f'  l24 {spec[\"name\"]}: two-copy pass failed ({_e!r:.80}), one copy from here on'); del _ms[1:], _mods[1:]\n"
     "                gc.collect(); torch.cuda.empty_cache(); _run(0, 1)\n"
     "        return Z_\n"),
    ("            W = mods[name].weight; B = sd[k.replace('.lora_A.', '.lora_B.')]\n"
     "            W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)\n"
     "            deltas.append((W, B, v))   # keep only the low-rank factors; the dense delta is recomputed to unmerge\n",
     "            B = sd[k.replace('.lora_A.', '.lora_B.')]\n"
     "            for _md in _mods:   # fork: every copy\n"
     "                W = _md[name].weight\n"
     "                W += ((B.to(W.device, torch.float32) @ v.to(W.device, torch.float32)) * scale).to(W.dtype)\n"
     "                deltas.append((W, B, v))   # keep only the low-rank factors; the dense delta is recomputed to unmerge\n"),
    ("        Z = np.zeros((len(texts), hid * len(L)), np.float32)\n"
     "        for i in range(0, len(order), bs):\n"
     "            b = order[i:i + bs]; Lm = max(len(enc[j]) for j in b)\n"
     "            ids = torch.full((len(b), Lm), tok.pad_token_id); att = torch.zeros((len(b), Lm), dtype=torch.long)\n"
     "            for r, j in enumerate(b): ids[r, :len(enc[j])] = torch.tensor(enc[j]); att[r, :len(enc[j])] = 1\n"
     "            hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states\n"
     "            if not all(torch.isfinite(hs[l]).all() for l in L):   # fp16 overflow (un-normed deep states): redo this batch in bf16\n"
     "                m.to(torch.bfloat16); hs = m(input_ids=ids.to(first), attention_mask=att.to(first), output_hidden_states=True, use_cache=False).hidden_states; m.to(torch.float16); OVF[0] += 1\n"
     "            mask = att.unsqueeze(-1).float(); mask[:, 0] = 0\n"
     "            Z[b] = torch.cat([nm((hs[l].float().cpu() * mask).sum(1) / mask.sum(1)) for l, nm in zip(L, norms)], -1).numpy()\n",
     "        Z = featurize(enc, norms, hid)   # fork: upstream's loop, now shared with the Gaussian head's train features\n"),
    ("done (bf16 redo batches {OVF[0]})')\n", "done (bf16 redo batches {sum(OVF)})')\n"),
    ("    del m, mods; gc.collect(); torch.cuda.empty_cache()\n    if train_texts is not None: return P_acc",
     "    del m, mods, _ms, _mods; gc.collect(); torch.cuda.empty_cache()\n    if train_texts is not None: return P_acc"),
]
L24_DP = L24_DP_FN + [   # after add_l24, whose anchor is the end of ll8v5's line
    ('"w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44}', '"w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44, "dp": True}'),
    (LL8V5, LL8V5.replace('"twin": 0.35}', '"twin": 0.35, "dp": True}')),
]

# exp019: a second, independent ATLAS bundle on the same low-margin rows, mixed after the first: harshitakumari256's
# ntl-atlas-bundle (Qwen3-4B-Base with a learned prefix and a LoRA up-projection per source; readout = softmax of a
# fitted temperature x mean token log-likelihood; public out-of-fold balanced accuracy 0.427 on train). In exp015's
# replica (no ATLAS there) it adds +0.0074 [+0.0004, +0.0145] at 10% on rows with top-2 margin < 0.3. The 4B fits a T4,
# so after ATLAS-8B one worker process per GPU (atlas2_worker.py, a new cell) scores half the rows, shortest first,
# with the bundle's own atlas_core.score_texts, until ATLAS's deadline; the rows scored by then are mixed.
ATLAS2_DATASET = "harshitakumari256/ntl-atlas-bundle"
ATLAS2_WORKER = r'''%%writefile atlas2_worker.py
"""fork (exp019): one GPU's share of the second ATLAS bundle (harshitakumari256/ntl-atlas-bundle, Qwen3-4B-Base), scored
with the bundle's own atlas_core.score_texts, shortest texts first, chunk by chunk until --deadline (a time.time()
value), saving after every chunk; `rows` in --out are positions in --texts."""
import argparse, os, sys, time
import numpy as np, pandas as pd, torch
p = argparse.ArgumentParser()
for k in ('--bundle', '--base', '--texts', '--out'): p.add_argument(k, required=True)
p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
p.add_argument('--deadline', type=float, default=float('inf')); p.add_argument('--chunk', type=int, default=16)
a = p.parse_args()
if time.time() > a.deadline: sys.exit(f'atlas2 shard {a.shard}: past its deadline before loading')
sys.path.insert(0, a.bundle); import atlas_core as ac; sys.path.remove(a.bundle)
t0 = time.time()
model, tok, cfg = ac.load_bundle(a.bundle, a.base, device='cuda:0', dtype=torch.float16, device_map={'': 'cuda:0'})
texts = pd.read_parquet(a.texts).text.tolist()
ntok = np.array([len(x) for x in ac.encode(tok, texts, cfg['max_len'], cfg['append_eos'])])
mine = np.argsort(ntok, kind='stable')[a.shard::a.nshards]
print(f'atlas2 shard {a.shard}: bundle loaded in {time.time() - t0:.0f}s, {len(mine)} texts, {ntok[mine].sum()} tokens', flush=True)
K = len(ac.LABELS); S = np.zeros((len(mine), K), np.float32); LL = np.zeros((len(mine), K), np.float32)
done, last, t1 = 0, 0.0, time.time()
while done < len(mine):
    if time.time() + 1.5 * last > a.deadline:
        print(f'atlas2 shard {a.shard}: stops at its deadline, {done}/{len(mine)} texts', flush=True); break
    c0, idx = time.time(), mine[done:done + a.chunk]
    S[done:done + len(idx)], LL[done:done + len(idx)] = ac.score_texts(
        model, tok, [texts[i] for i in idx], max_len=cfg['max_len'], append_eos=cfg['append_eos'],
        temps=tuple(cfg['temps']), texts_per_batch=2, device='cuda:0', log_every=0)
    done += len(idx); last = time.time() - c0
    np.savez(a.out + '.tmp.npz', rows=mine[:done], S=S[:done], LL=LL[:done]); os.replace(a.out + '.tmp.npz', a.out)
    if done % (10 * a.chunk) == 0: print(f'atlas2 shard {a.shard}: {done}/{len(mine)} texts, {time.time() - t1:.0f}s', flush=True)
print(f'atlas2 shard {a.shard}: {done} texts, {ntok[mine[:done]].sum()} tokens in {time.time() - t1:.0f}s', flush=True)
'''
ATLAS2_FN = '''def atlas2_scores(rows, budget):
    """fork (exp019): the second ATLAS bundle on test rows `rows`, one atlas2_worker.py process per GPU, shortest rows
    first, until T0 + budget. Returns the positions in `rows` it scored and their probabilities (the bundle's readout:
    softmax of tau x score), in our label order."""
    import subprocess
    if T0 + budget - time.time() < 300: raise RuntimeError('atlas2: under 5 min to the deadline')   # loading takes ~2 min
    _cfg = json.load(open(os.path.join(a.atlas2_dir, 'atlas_config.json')))
    _txt = os.path.join(a.cache, 'atlas2_texts.parquet'); pd.DataFrame({'text': test.text.values[rows]}).to_parquet(_txt, index=False)
    _n = max(1, torch.cuda.device_count()); _out = [os.path.join(a.cache, f'atlas2_{k}.npz') for k in range(_n)]
    for _f in _out:
        if os.path.exists(_f): os.remove(_f)
    _w = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atlas2_worker.py')
    _pr = [subprocess.Popen([sys.executable, _w, '--bundle', a.atlas2_dir, '--base', a.atlas2_base, '--texts', _txt,
                             '--out', _out[k], '--shard', str(k), '--nshards', str(_n), '--deadline', str(T0 + budget)],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))) for k in range(_n)]
    for _p in _pr:
        try: _p.wait(timeout=max(30, T0 + budget + 120 - time.time()))
        except subprocess.TimeoutExpired: _p.kill(); _p.wait(); log('atlas2: a worker was stopped after its deadline')
    _z = [np.load(_f) for _f in _out if os.path.exists(_f)]
    if not _z: raise RuntimeError('atlas2: no worker output')
    _pos = np.concatenate([z['rows'] for z in _z])
    _X = _cfg['tau'] * np.concatenate([z['S' if _cfg['score_kind'] == 'S' else 'LL'] for z in _z]).astype(np.float64)
    _o = np.argsort(_pos); _pos, _X = _pos[_o], _X[_o]
    _A = np.exp(_X - _X.max(1, keepdims=True)); _A /= _A.sum(1, keepdims=True)
    assert np.isfinite(_A).all() and len(np.unique(_pos)) == len(_pos)
    return _pos, _A[:, [_cfg['labels'].index(l) for l in LABELS]]

'''
ATLAS2 = [
    ("print('atlas', ATLAS)\n",
     "print('atlas', ATLAS)\n"
     "ATLAS2 = [os.path.dirname(x) for x in glob.glob('/kaggle/input/**/atlas_config.json', recursive=True) if '/ntl-atlas-bundle/' in x]\n"
     "assert len(ATLAS2) == 1, ATLAS2; ATLAS2 = ATLAS2[0]; B4 = mdir('/qwen-3/', '/4b-base/')   # fork (exp019): second ATLAS bundle\n"
     "print('atlas2', ATLAS2, B4)\n"),
    ("p.add_argument('--atlas_bs', type=int, default=4); p.add_argument('--atlas_chunk', type=int, default=3); "
     "p.add_argument('--atlas_budget', type=float, default=8.3 * 3600)\n",
     "p.add_argument('--atlas_bs', type=int, default=4); p.add_argument('--atlas_chunk', type=int, default=3); "
     "p.add_argument('--atlas_budget', type=float, default=8.3 * 3600)\n"
     "p.add_argument('--atlas2_dir', default=''); p.add_argument('--atlas2_base', default=''); "
     "p.add_argument('--atlas2_w', type=float, default=0.0)   # fork (exp019): a second ATLAS bundle\n"),
    ("\nCOMP_P = None\n", "\n" + ATLAS2_FN + "COMP_P = None\n"),
    ("        P = mixed / mixed.sum(1, keepdims=True)\n",
     "        if a.atlas2_dir and a.atlas2_w > 0:   # fork (exp019): the second bundle, after the first, on the same rows\n"
     "            try:\n"
     "                _pos, A2_ = atlas2_scores(rows, a.atlas_budget); _ar = rows[_pos]\n"
     "                mixed[_ar] = mixed[_ar] - a.atlas2_w * P[_ar] + a.atlas2_w * A2_; log(f'atlas2 mixed at {a.atlas2_w} on {len(_ar)}/{len(rows)} rows', gpu_mem())\n"
     "            except Exception as _e:\n"
     "                log('ATLAS2 SKIPPED:', repr(_e))\n"
     "        P = mixed / mixed.sum(1, keepdims=True)\n"),
    ("--atlas_budget {ATLAS_BUDGET} ", "--atlas_budget {ATLAS_BUDGET} --atlas2_dir {ATLAS2} --atlas2_base {B4} --atlas2_w 0.10 "),
]
# the worker cell goes right after the cell that writes comp_stack.py
ATLAS2_CELLS = [("%%writefile comp_stack.py", ATLAS2_WORKER)]


def fast_member_commit(tags):
    """FAST_COMMIT, except that the 3-row commit run still runs infer.py on our new member(s) alone: its in-sample
    check (40 train rows, accuracy > 0.5 asserted) proves that base, adapters and heads load in upstream's image."""
    return FAST_COMMIT[:2] + [
        ("!python infer.py --data {DATA}",
         "if FAST_COMMIT:   # fork: the commit run checks only our new member(s); the hidden rerun runs everything\n"
         f"    _NEW = [m_ for m_ in MODELS if m_['name'] in {sorted(tags)!r}]\n"
         "    !python infer.py --data {DATA} --models '{json.dumps(_NEW)}' --perpassage 1 --extra {EXTRA} --extra_w 0.5 "
         "--tfidf_w 0 --distort_w 0 --svc_w 0 --pool {POOL} --adapt_w 1.0 --adapt_thr 0.15 --sinkhorn 1.0 --post 1 "
         "--comp_w 0 --out submission.parquet --cache /kaggle/working/cache\n"
         "else:\n    !python infer.py --data {DATA}")]


ENC_KERNEL = "omerfarukmerey/ntl-member-enc-pool"   # our two-stage encoders (kaggle/member/enc_worker.py, exp014)


def add_encoders(weights):
    """Cell 0 copies each encoder of ntl-member-enc-pool to /tmp (as the ModernBERT one); MODELS gets an 'enc' entry."""
    lookup = "".join(
        f"ENC_{n.upper()} = os.path.dirname(next(p for p in glob.glob('/kaggle/input/**/meta.json', recursive=True) "
        f"if '/ntl-member-enc-pool/{n}/' in p))\n"
        f"shutil.copytree(ENC_{n.upper()}, '/tmp/{n}', dirs_exist_ok=True); ENC_{n.upper()} = '/tmp/{n}'   "
        f"# fork: our two-stage encoder (exp014)\n" for n in weights)
    entries = "".join(f'  {{"name": "{n}", "type": "enc", "ckpt": ENC_{n.upper()}, "w": {w}}},\n' for n, w in weights.items())
    return [("print('modernbert', MB, sorted(os.listdir(MB)))\n",
             "print('modernbert', MB, sorted(os.listdir(MB)))\n" + lookup),
            ('  {"name": "mb", "type": "enc", "ckpt": MB, "w": 1.5},\n]',
             '  {"name": "mb", "type": "enc", "ckpt": MB, "w": 1.5},\n' + entries + "]")]


ENC_WEIGHTS = {"roberta_large_pool": 1.0}   # set from exp015 before pushing (DeBERTa-v3-large ran out of GPU memory, exp014)


# our own two-stage LoRA24 members (kaggle/member/l24_worker.py): tag -> (base model parts, Kaggle model, stage-2 kernel,
# blend weight). The weight is set from exp015's out-of-fold replica before pushing.
L24_MEMBERS = {"mn8v1": (("/ministral-3/", "/ministral-3-8b-base-2512/"),
                         "mistral-ai/ministral-3/Transformers/ministral-3-8b-base-2512/1", "ntl-l24-mn8-s2", 1.0)}


def l24_entry(tag, w, dp=False):
    return (f'  {{"name": "{tag}", "type": "l24", "path": B_{tag.upper()}, "adapters": A_{tag.upper()}, "keep": 24, '
            f'"layers": L, "w": {w}, "bs": 16, "max_len": 384, "twin": 0.35' + (', "dp": True' if dp else '') + '},\n')


def add_l24(tags, weights=None, dp=False):
    """Cell 1 finds each member's base and its five fold adapters + full fit (only in our stage-2 kernel's output);
    MODELS gets an 'l24' entry after ll8v5 with infer.py's usual twin weight. `weights` overrides L24_MEMBERS'."""
    weights = {t: (weights or {}).get(t, L24_MEMBERS[t][3]) for t in tags}
    lookup = "".join(
        f"B_{t.upper()} = mdir{L24_MEMBERS[t][0]!r}\n"
        f"A_{t.upper()} = [adir(f'{t}_f{{k}}') for k in range(5)] + [adir('{t}')]\n"
        f"assert all('/{L24_MEMBERS[t][2]}/' in d for d in A_{t.upper()}), A_{t.upper()}   # fork: our two-stage LoRA24 member\n"
        f"print('{t}', B_{t.upper()}, A_{t.upper()})\n" for t in tags)
    entries = "".join(l24_entry(t, weights[t], dp) for t in tags)
    return [("print('v5', AQ14V5, ALL8V5)\n", "print('v5', AQ14V5, ALL8V5)\n" + lookup),
            ('"max_len": 384, "twin": 0.35},\n  {"name": "n14b"', '"max_len": 384, "twin": 0.35},\n' + entries + '  {"name": "n14b"')]


def check_nollama(code):
    return "ll8v5" not in code.split("MODELS = [", 1)[1] and "mdir('/llama-3.1/'" not in code


def check_mb(code):
    models = code.split("MODELS = [", 1)[1]
    return all(s in models for s in ('"name": "ll8v5"', '"name": "mb", "type": "enc", "ckpt": MB, "w": 1.5')) \
        and code.count("'\\u2014'") == 1 and "_pos, A_ = atlas_scores" in code and "raise RuntimeError('atlas would" not in code


def check_plain(code):
    """The 0.651 recipe as avazaku reproduced it: Llama from his cache, no ModernBERT, upstream ATLAS and guards."""
    models = code.split("MODELS = [", 1)[1]
    return '"name": "ll8v5"' in models and '"name": "mb"' not in models and "_SHARDS" in code \
        and "raise RuntimeError('atlas would" in code and "'\\u2014', {" not in code


VARIANTS = {
    "ntl-nlm27-noatlas": {"patches": [MIRROR_LLAMA] + FAST_COMMIT + ATLAS_OFF,
                          "check": lambda code: check_plain(code) and "--atlas_w 0 " in code
                          and code.count("FAST_COMMIT") == 5,
                          "kernels": [LLAMA_CACHE],
                          "intro": "Private fork of [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) "
                                   "(Dhruv Tibarewal, public 0.651) as avazaku reproduced it (Llama-3.1-8B from his "
                                   "public cache), with one change: ATLAS off (--atlas_w 0). Tests whether ATLAS, "
                                   "which must fit a fixed 8.3 h budget, contributes on the hidden test. The 3-row "
                                   "commit run writes a placeholder; the hidden rerun runs every cell. Built with "
                                   "Llama (Llama 3.1 Community License)."},
    "ntl-nlm27-nollama": {"patches": NO_LLAMA, "check": check_nollama,
                          "intro": "Private fork of [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) "
                                   "(Dhruv Tibarewal, public 0.651) without the Llama-3.1-8B member (ll8v5), whose "
                                   "base model is gated for this account. Everything else is unchanged."},
    "ntl-nlm27-mb": {"patches": [MIRROR_LLAMA] + ADD_MODERNBERT + SKIP_ZERO_TFIDF + ATLAS_PARTIAL + EM_DASH_GUARD,
                     "check": check_mb, "datasets": [MODERNBERT], "kernels": [LLAMA_CACHE],
                     "intro": "Private fork of [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) "
                              "(Dhruv Tibarewal, public 0.651). Llama-3.1-8B from avazaku's public cache "
                              "([Verified Llama Asset Cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache), "
                              "as in his [0.651 reproduction](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction)); "
                              "plus Dhruv's ModernBERT-large member (weight 1.5), ATLAS kept on the rows scored "
                              "before the time budget, an em-dash typography guard, and no fit for the zero-weight "
                              "TF-IDF member. Built with Llama (Llama 3.1 Community License)."},
    "ntl-nlm27-mb-enc": {"patches": [MIRROR_LLAMA] + ADD_MODERNBERT + SKIP_ZERO_TFIDF + ATLAS_PARTIAL + EM_DASH_GUARD
                         + add_encoders(ENC_WEIGHTS),
                         "check": lambda code: check_mb(code) and all(f'"name": "{n}", "type": "enc"' in code
                                                                      for n in ENC_WEIGHTS),
                         "datasets": [MODERNBERT], "kernels": [LLAMA_CACHE, ENC_KERNEL],
                         "intro": "ntl-nlm27-mb plus our own two-stage RoBERTa-large member (2 epochs on Dhruv's "
                                  "120k pool, then 3 on the official rows; kernel ntl-member-enc-pool). Built with "
                                  "Llama (Llama 3.1 Community License)."},
}
MN8_INTRO = ("The 0.651 recipe ([nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) as avazaku reproduced it) "
             "plus our own two-stage LoRA24 member on Ministral-3-8B-Base (Mistral AI, Apache 2.0): Lucas Boesen's "
             "LoRA24 recipe, stage 1 on Dhruv's 120k pool, stage 2 on the official rows, five fold adapters and a full "
             "fit (kernels ntl-l24-mn8-s1 / -s2). Its ~45 min of inference are paid for by reading Llama-3.1-8B from "
             "local disk (one ~2 min copy instead of ~34 min of scattered reads from the notebook-output mount) and "
             "by not fitting the zero-weight TF-IDF member. The 3-row commit run checks only the new member; the "
             "hidden rerun runs every cell. Built with Llama (Llama 3.1 Community License).")
# exp015 (plain base): mn8v1 adds +0.0048 where ll8v5 adds +0.0029 (ll8v5 then gave +0.008 on the LB), and on top of
# ll8v5 +0.0004 at weight 1.0, +0.0037 at 1.5, +0.0041 at 2.0, +0.0059 at 3.0 (q14v5's weight; fold 2 carries most of
# it); the LB decides between 1.0 (ll8v5's), 2.0 and 3.0
for _name, _w, _atlas in (("ntl-nlm27-mn8", 1.0, []), ("ntl-nlm27-mn8-noatlas", 1.0, ATLAS_OFF),
                          ("ntl-nlm27-mn8w2-noatlas", 2.0, ATLAS_OFF), ("ntl-nlm27-mn8w3-noatlas", 3.0, ATLAS_OFF)):
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + fast_member_commit(["mn8v1"])
        + add_l24(["mn8v1"], {"mn8v1": _w}) + _atlas,
        "check": (lambda w, atlas_off: lambda code: check_plain(code) and l24_entry("mn8v1", w) in code
                  and code.count("FAST_COMMIT") == 5 and "BL = '/tmp/llama-3.1/8b'" in code
                  and ("--atlas_w 0 " in code) == atlas_off)(_w, bool(_atlas)),
        "kernels": [LLAMA_CACHE, f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"],
        "models": [L24_MEMBERS["mn8v1"][1]],
        "intro": MN8_INTRO + f" Its blend weight is {_w} (ll8v5's is 1.0)."
        + (" ATLAS is off (--atlas_w 0), as in ntl-nlm27-noatlas, for a timing-free A/B." if _atlas else "")}

# Dhruv's newer Qwen2.5-14B LoRA24 member (dataset nlm-q25-v5, not in nlm-v27): exp015 scores its fold adapters at 0.609
# alone, and with mn8v1 at 1.0 the pair adds +0.0064 at weight 1.0 each. At 2.6 rows/s (ntl-nlm-oof-q25) one adapter
# pass over ~4,000 rows takes ~26 min, so only its full-data adapter runs, like Nishant's two single-adapter members.
# Its 8 depths were not published then; ntl-nlm-oof-q25 recovered them from that adapter (in-sample 0.91 on 256 rows),
# and its OOF used the same ones. Only without ATLAS: with it, the extra ~70 min would push ATLAS past its budget.
# 2026-10-27: Dhruv's "Name That LLM Fully Reproducible" (public 0.651, like nlm-v27) now runs this member with all six
# adapters at weight 2.0 in place of Nishant's q25_14 and depths [3, 5, 8, 11, 16, 21, 27, 32] (27 where we recovered
# 29). It did not move his LB score, so this variant is not submitted.
Q25V5 = "dhruvtibarewal/nlm-q25-v5"
Q25V5_ENTRY = ('  {"name": "q25_14v5", "type": "l24", "path": B25_14, "adapters": [A_Q25V5], "keep": 32, '
               '"layers": [3, 5, 8, 11, 16, 21, 29, 32], "w": 1.0, "bs": 8, "max_len": 384, "twin": 0.35},\n')
ADD_Q25V5 = [("print('v5', AQ14V5, ALL8V5)\n",
              "print('v5', AQ14V5, ALL8V5)\n"
              "A_Q25V5 = adir('q25_14v5'); assert '/nlm-q25-v5/' in A_Q25V5, A_Q25V5   # fork: Dhruv's newer Qwen2.5-14B member\n"
              "print('q25_14v5', B25_14, A_Q25V5)\n"),
             ('  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]',
              Q25V5_ENTRY + '  {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]')]
for _name, _w in (("ntl-nlm27-mn8-deadline", 1.0), ("ntl-nlm27-mn8w2-deadline", 2.0), ("ntl-nlm27-mn8w3-deadline", 3.0)):
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + fast_member_commit(["mn8v1"])
        + add_l24(["mn8v1"], {"mn8v1": _w}) + ATLAS_DEADLINE,
        "check": (lambda w: lambda code: check_plain(code.replace("log(f'atlas stops at its deadline", "raise RuntimeError('atlas would"))
                  and l24_entry("mn8v1", w) in code and code.count("FAST_COMMIT") == 5
                  and "BL = '/tmp/llama-3.1/8b'" in code and "--atlas_w 0.10 " in code
                  and "_pos, A_ = atlas_scores" in code and "if time.time() - T0 > budget:" in code)(_w),
        "kernels": [LLAMA_CACHE, f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"],
        "models": [L24_MEMBERS["mn8v1"][1]],
        "intro": MN8_INTRO + f" Its blend weight is {_w} (ll8v5's is 1.0). ATLAS scores rows until its deadline and "
                 "keeps those (upstream drops ATLAS whole when its projection overruns the budget)."}
# exp018: the deadline variants with the data-parallel l24_predict for q8v5, ll8v5 and mn8v1 (ntl-l24-dp-check compares
# it with upstream's on T4 x2). By the estimate in exp017, ATLAS then starts ~45 min earlier than in avazaku's 0.651
# run instead of ~11 min later, so it finishes whenever it finished there.
for _name, _w in (("ntl-nlm27-mn8w2-dp", 2.0), ("ntl-nlm27-mn8w3-dp", 3.0)):
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + fast_member_commit(["mn8v1"])
        + add_l24(["mn8v1"], {"mn8v1": _w}, dp=True) + ATLAS_DEADLINE + L24_DP,
        "check": (lambda w: lambda code: check_plain(code.replace("log(f'atlas stops at its deadline", "raise RuntimeError('atlas would"))
                  and l24_entry("mn8v1", w, dp=True) in code and code.count('"dp": True') == 3
                  and code.count("FAST_COMMIT") == 5 and "BL = '/tmp/llama-3.1/8b'" in code and "--atlas_w 0.10 " in code
                  and "_pos, A_ = atlas_scores" in code and "if time.time() - T0 > budget:" in code
                  and "for _f in [_ex.submit(_run, k, len(_ms))" in code and "Z = featurize(enc, norms, hid)" in code)(_w),
        "kernels": [LLAMA_CACHE, f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"],
        "models": [L24_MEMBERS["mn8v1"][1]],
        "intro": MN8_INTRO + f" Its blend weight is {_w} (ll8v5's is 1.0). ATLAS scores rows until its deadline and "
                 "keeps those (upstream drops ATLAS whole when its projection overruns the budget). The three 24-layer 8B "
                 "members (q8v5, ll8v5 and this one) run with a full copy on each T4 and alternate batches in two threads "
                 "instead of one copy split across both GPUs: the same computation in about half the time, which goes "
                 "to ATLAS's fixed budget."}
# exp019 (ATLAS levers), each one change on an exp018 base: "-a2" adds the second ATLAS bundle (Qwen3-4B, 10%) after
# ATLAS-8B on the same rows, "-a20" mixes ATLAS-8B at 20% instead of 10% (its author's 10% was never tuned; ATLAS is
# +0.008 at 10% here, #17). Bases: mn8w2-dp, and the plain recipe with the copy-per-GPU l24_predict and no Ministral
# (ntl-nlm27-dp), so both outcomes of 2026-10-28 have their pairs.
ATLAS_W20 = [("--atlas_w 0.10 ", "--atlas_w 0.20 ")]
DP_INTRO = ("The 0.651 recipe ([nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) as avazaku reproduced it) "
            "with exp018's faster l24_predict (a full copy of each 24-layer 8B member on each T4, alternate batches in "
            "two threads, masked means on the GPU; every log-probability within 7.6e-6 of upstream's), Llama-3.1-8B "
            "read from local disk, the zero-weight TF-IDF member not fitted, and ATLAS scored up to its deadline. The "
            "3-row commit run writes a placeholder; the hidden rerun runs every cell. Built with Llama (Llama 3.1 "
            "Community License).")
A2_INTRO = (" Also a second ATLAS bundle ([harshitakumari256/ntl-atlas-bundle](https://www.kaggle.com/datasets/harshitakumari256/ntl-atlas-bundle), "
            "Qwen3-4B-Base) mixed at 10% after ATLAS-8B on the same low-margin rows, one worker process per GPU, in the "
            "time left before ATLAS's deadline.")
A20_INTRO = " ATLAS-8B is mixed at 20% instead of 10%."


def dp_check(w, atlas_w, atlas2):
    def check(code):
        ok = (check_plain(code.replace("log(f'atlas stops at its deadline", "raise RuntimeError('atlas would"))
              and code.count('"dp": True') == (3 if w else 2) and code.count("FAST_COMMIT") == 5
              and "BL = '/tmp/llama-3.1/8b'" in code and f"--atlas_w {atlas_w} " in code
              and "_pos, A_ = atlas_scores" in code and "if time.time() - T0 > budget:" in code
              and "for _f in [_ex.submit(_run, k, len(_ms))" in code and "Z = featurize(enc, norms, hid)" in code)
        ok = ok and ((l24_entry("mn8v1", w, dp=True) in code) if w else '"name": "mn8v1"' not in code)
        return ok and (("--atlas2_w 0.10 " in code and "def atlas2_scores(rows, budget):" in code
                        and code.count("%%writefile atlas2_worker.py") == 1) == atlas2)
    return check


for _name, _w, _extra in (("ntl-nlm27-dp", None, []), ("ntl-nlm27-dp-a2", None, ATLAS2), ("ntl-nlm27-dp-a20", None, ATLAS_W20),
                          ("ntl-nlm27-mn8w2-dp-a2", 2.0, ATLAS2), ("ntl-nlm27-mn8w2-dp-a20", 2.0, ATLAS_W20)):
    _mn8, _a2 = _w is not None, _extra is ATLAS2
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF
        + (fast_member_commit(["mn8v1"]) + add_l24(["mn8v1"], {"mn8v1": _w}, dp=True) if _mn8 else FAST_COMMIT)
        + ATLAS_DEADLINE + L24_DP + _extra,
        "cells": ATLAS2_CELLS if _a2 else [],
        "check": dp_check(_w, "0.20" if _extra is ATLAS_W20 else "0.10", _a2),
        "datasets": [ATLAS2_DATASET] if _a2 else [],
        "kernels": [LLAMA_CACHE] + ([f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"] if _mn8 else []),
        "models": [L24_MEMBERS["mn8v1"][1]] if _mn8 else [],
        "intro": (MN8_INTRO + f" Its blend weight is {_w} (ll8v5's is 1.0). The three 24-layer 8B members run with a "
                  "full copy on each T4 (exp018), and ATLAS scores rows up to its deadline." if _mn8 else DP_INTRO)
        + (A2_INTRO if _a2 else A20_INTRO if _extra is ATLAS_W20 else "")}
VARIANTS["ntl-nlm27-mn8q25-noatlas"] = {
    "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + fast_member_commit(["mn8v1", "q25_14v5"])
    + add_l24(["mn8v1"]) + ADD_Q25V5 + ATLAS_OFF,
    "check": lambda code: check_plain(code) and l24_entry("mn8v1", 1.0) in code and Q25V5_ENTRY in code
    and code.count("FAST_COMMIT") == 5 and "BL = '/tmp/llama-3.1/8b'" in code and "--atlas_w 0 " in code,
    "datasets": [Q25V5], "kernels": [LLAMA_CACHE, f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"],
    "models": [L24_MEMBERS["mn8v1"][1]],
    "intro": MN8_INTRO + " Its blend weight is 1.0 (ll8v5's is 1.0). Also Dhruv Tibarewal's newer two-stage Qwen2.5-14B "
             "LoRA24 member ([nlm-q25-v5](https://www.kaggle.com/datasets/dhruvtibarewal/nlm-q25-v5); its full-data "
             "adapter, weight 1.0). ATLAS is off (--atlas_w 0), as in ntl-nlm27-noatlas."}
# exp020: Roman Tamrazov's "Name That LLM | Who am I?" (public 0.654, 2026-10-27) is nlm-v27 plus one step at the end of
# infer.py, "v38a": a typography log-likelihood (34 binary markers, per-class Bernoulli rates from train, centred,
# clipped to +-0.35, gated by (0.8 - top-2 margin) / 0.8) added to the log-probabilities after Sinkhorn, whose argmax
# replaces upstream's post step (guards and dense-cluster Hungarian; the hidden test has few answers per prompt). The
# block is copied verbatim into who_am_i_v38a.txt from the version that scored 0.654 (pulled 2026-10-28); his other
# changes (no zero-weight TF-IDF fit, ATLAS keeping the rows it scored) we make already. It runs after upstream's own
# submission.parquet is written and the notebook's `!python infer.py` ignores the exit code, so if it failed, that
# file would stand. exp020 (replica): +0.003 over no post step, -0.001 on top of the 4B ATLAS.
V38A_FILE, V38A_SHA256 = "who_am_i_v38a.txt", "48ebed37378c5bd6c8027f1244a53b32bcdcfbf969953556a8c163d050722ce5"
V38A_BLOCK = (HERE / V38A_FILE).read_text(encoding="utf-8")
assert hashlib.sha256(V38A_BLOCK.encode()).hexdigest() == V38A_SHA256, V38A_FILE
WROTE = "sub.to_parquet(a.out, index=False); log('wrote', a.out, sub.model.value_counts().to_dict())\n"
V38A = [("if a.post and a.perpassage:\n", "_v38_before_hard = tot.copy()\nif a.post and a.perpassage:\n"),
        (WROTE, WROTE + V38A_BLOCK)]
V38_INTRO = (" Plus Roman Tamrazov's typography step from [Name That LLM | Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) "
             "(public 0.654), verbatim: a train-only Bayesian likelihood of 34 typography markers, gated by the top-2 "
             "margin, added to the log-probabilities after Sinkhorn in place of the guards and dense-cluster Hungarian.")
for _name, _w, _extra in (("ntl-v38-dp", None, []), ("ntl-v38-dp-a2", None, ATLAS2), ("ntl-v38-dp-a20", None, ATLAS_W20),
                          ("ntl-v38-dp-a2-a20", None, ATLAS2 + ATLAS_W20), ("ntl-v38-mn8w2-dp", 2.0, []),
                          ("ntl-v38-mn8w2-dp-a2", 2.0, ATLAS2), ("ntl-v38-mn8w3-dp", 3.0, []),
                          ("ntl-v38-mn8w2-dp-a20", 2.0, ATLAS_W20), ("ntl-v38-mn8w3-dp-a20", 3.0, ATLAS_W20)):
    _mn8, _a2, _a20 = _w is not None, ATLAS2[0] in _extra, ATLAS_W20[0] in _extra
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF
        + (fast_member_commit(["mn8v1"]) + add_l24(["mn8v1"], {"mn8v1": _w}, dp=True) if _mn8 else FAST_COMMIT)
        + ATLAS_DEADLINE + L24_DP + _extra + V38A,
        "cells": ATLAS2_CELLS if _a2 else [],
        "check": (lambda chk: lambda code: chk(code) and code.count(V38A_BLOCK) == 1
                  and code.count("_v38_before_hard") == 2)(dp_check(_w, "0.20" if _a20 else "0.10", _a2)),
        "datasets": [ATLAS2_DATASET] if _a2 else [],
        "kernels": [LLAMA_CACHE] + ([f"omerfarukmerey/{L24_MEMBERS['mn8v1'][2]}"] if _mn8 else []),
        "models": [L24_MEMBERS["mn8v1"][1]] if _mn8 else [],
        "intro": (MN8_INTRO + f" Its blend weight is {_w} (ll8v5's is 1.0). The three 24-layer 8B members run with a "
                  "full copy on each T4 (exp018), and ATLAS scores rows up to its deadline." if _mn8 else DP_INTRO)
        + (A2_INTRO if _a2 else "") + (A20_INTRO if _a20 else "") + V38_INTRO}
# 2026-10-28 LB: ATLAS-8B at 20% scored 0.652 against the recipe's 0.651 (#21), while the second ATLAS (#20, 0.650) and
# Ministral (#18/#19/#22, -0.006 to -0.016) lost. So 2026-10-29 maps ATLAS-8B's weight on the 0.654 recipe: 30% and 40%
# (the 4B ATLAS replica peaks at 0.2-0.3 and drops from 0.4), and 30% on rows with top-2 margin below 0.45 instead of
# 0.3 (at weight w a row can only flip if its margin is below w / (1 - w), 0.43 at 30%; if time runs short, ATLAS's
# deadline keeps the shortest rows scored).
ATLAS_M45 = [("--atlas_margin 0.3 ", "--atlas_margin 0.45 ")]
for _name, _aw, _extra in (("ntl-v38-dp-a30", "0.30", []), ("ntl-v38-dp-a40", "0.40", []),
                           ("ntl-v38-dp-a30-m45", "0.30", ATLAS_M45)):
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + FAST_COMMIT + ATLAS_DEADLINE + L24_DP
        + [("--atlas_w 0.10 ", f"--atlas_w {_aw} ")] + _extra + V38A,
        "check": (lambda chk, m45: lambda code: chk(code) and code.count(V38A_BLOCK) == 1
                  and code.count("_v38_before_hard") == 2 and ("--atlas_margin 0.45 " in code) == m45)(
            dp_check(None, _aw, False), bool(_extra)),
        "kernels": [LLAMA_CACHE],
        "intro": DP_INTRO + f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%"
        + (" on rows with top-2 margin below 0.45 instead of 0.3." if _extra else ".") + V38_INTRO}
# exp024: PeterDreamHan's "Name That LLM Soft Sinkhorn 0.5 T4" (public 0.653, 2026-10-28) is avazaku's 0.651 reproduction
# with one change, --sinkhorn 1.0 -> 0.5: the final log-probabilities go half way to the balanced ones (0.5 log P +
# 0.5 log Q, i.e. half of Sinkhorn's per-class log-offset). On the same base that is +0.002, Roman's step +0.003 and
# ATLAS-8B at 20% +0.001 (#21), each at a different stage (ATLAS mix, then balancing, then the typography step), so
# 2026-10-29 stacks them on the 0.654 recipe: 0.5 and 0.25, each with ATLAS at 10% and 20%, and 0.5 with 30% (the LB
# curve below 0.5 is unknown; exp024's replica: on uneven model mixes 0.25 >= 0.5 > 1 with or without Roman's step).
SOFT_SK = "[Soft Sinkhorn 0.5](https://www.kaggle.com/code/peterdreamhan/name-that-llm-soft-sinkhorn-0-5-t4)"
for _name, _sk, _aw in (("ntl-v38-dp-sk50", "0.5", "0.10"), ("ntl-v38-dp-sk50-a20", "0.5", "0.20"),
                        ("ntl-v38-dp-sk50-a30", "0.5", "0.30"), ("ntl-v38-dp-sk25-a20", "0.25", "0.20"),
                        ("ntl-v38-dp-sk25", "0.25", "0.10")):
    VARIANTS[_name] = {
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + SKIP_ZERO_TFIDF + FAST_COMMIT + ATLAS_DEADLINE + L24_DP
        + ([("--atlas_w 0.10 ", f"--atlas_w {_aw} ")] if _aw != "0.10" else [])
        + [("--sinkhorn 1.0 ", f"--sinkhorn {_sk} ")] + V38A,
        "check": (lambda chk, sk: lambda code: chk(code) and code.count(V38A_BLOCK) == 1
                  and code.count("_v38_before_hard") == 2 and code.count(f" --sinkhorn {sk} --post 1 ") == 1
                  and "--sinkhorn 1.0" not in code)(dp_check(None, _aw, False), _sk),
        "kernels": [LLAMA_CACHE],
        "intro": DP_INTRO + (f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%." if _aw != "0.10" else "")
        + f" Sinkhorn balancing at {_sk} instead of 1.0, as in PeterDreamHan's {SOFT_SK} (public 0.653)." + V38_INTRO}
# 2026-10-28 15:14 UTC: mercurydust's "Name That LLM-Submission" took the Code tab's top score, public 0.660. It is
# Roman's 0.654 notebook with his "V31" changes: the matched-clean external rows de-duplicated against train and used,
# source-weighted with the pool, by the SVC, plus the TF-IDF (0.25) and text-distortion (0.10) members; the comp
# stack's fitted family gates applied within each family; Sinkhorn at 0.45 towards 0.65 x the predicted + 0.35 x the
# train class marginals instead of 1.0 towards even ones; a reworked typography step (37 markers, Beta-Bernoulli rates
# shrunk to the global rate, marker weights, gated by margin, length and marker count); no post step. Its version 2
# (the scored code; version 3 only rewrites the intro) is kept byte-for-byte, with its metadata (docker image
# included). Roman's code runs infer.py through subprocess.run and already stops ATLAS at its deadline, keeping the rows
# scored, but also stops it at the first checkpoint whose projected finish passes the budget; that stop is removed.
V31_FILE, V31_SHA256 = "upstream_mercury_v31.ipynb", "a6d7efe74f0988e0aa05c58fde0bce7f5c1457548213de57130df0b5f5331618"
V31_RUN = "print('RUNNING MAIN INFERENCE', flush=True)\nsubprocess.run(cmd, check=True)\n"
FAST_COMMIT_V31 = FAST_COMMIT[:2] + [
    (V31_RUN, "print('RUNNING MAIN INFERENCE', flush=True)\n"
     "if FAST_COMMIT:   # fork: placeholder from the commit run; Kaggle reruns the notebook on the hidden test\n"
     "    _test_ph = pd.read_csv(os.path.join(DATA, 'test.csv'))\n"
     "    pd.DataFrame({'id': _test_ph.id, 'model': 'gpt-5.5'}).to_parquet('/kaggle/working/submission.parquet', index=False)\n"
     "else:\n    subprocess.run(cmd, check=True)\n")]
ATLAS_TO_DEADLINE_V31 = [
    ("                if _done > 50 and _proj > budget:\n"
     "                    log(f'atlas predicted overrun; preserving {int(scored.sum())} completed rows')\n"
     "                    break\n", "")]
V31_INTRO = ("mercurydust's [Name That LLM-Submission](https://www.kaggle.com/code/mercurydust/name-that-llm-submission) "
             "(public 0.660, version 2: Roman Tamrazov's [Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) "
             "0.654 with de-duplicated, source-weighted external rows, the comp stack's family gates, a softer Sinkhorn "
             "towards a mix of predicted and train class marginals and a reworked typography step) with exp018's faster "
             "l24_predict (a full copy of each 24-layer 8B member on each T4, alternate batches in two threads, masked "
             "means on the GPU), Llama-3.1-8B read from local disk, and ATLAS scored up to its deadline (no stop on a "
             "projected overrun). The 3-row commit run writes a placeholder; the hidden rerun runs every cell. Built with "
             "Llama (Llama 3.1 Community License).")


# exp025 (replica, V31's balancing): V31's typography step adds +0.0008 over none, and with its margin gate alone (no
# length or coverage gate) +0.0026, i.e. +0.0019 [-0.0004, +0.0044] over V31's step, ahead or level on every uneven
# pool. "-mg" keeps only the margin gate.
V31_GATE = "    _gate = _confidence_gate * _length_gate * _coverage_gate\n"
TYPO_MG = [(V31_GATE, "    _gate = _confidence_gate   # fork (exp025): the margin gate alone; the length and coverage gates "
                      "cost 0.002 out of fold\n")]


def v31_check(atlas_w, mg):
    def check(code):
        return (code.count('"dp": True') == 2 and code.count("FAST_COMMIT") == 5 and "_SHARDS" in code
                and "BL = '/tmp/llama-3.1/8b'" in code and f"'--atlas_w', '{atlas_w}'" in code
                and "atlas predicted overrun" not in code and "if _b and time.time() - T0 >= budget - 60:" in code
                and "for _f in [_ex.submit(_run, k, len(_ms))" in code and "Z = featurize(enc, norms, hid)" in code
                and "'--sinkhorn', '0.45', '--sinkhorn_target_mix', '0.35', '--post', '0'" in code
                and "_apply_conditional_family_gate('claude'" in code and "hierarchical_beta_bernoulli_typography_v2" in code
                and (V31_GATE in code) != mg and ("_gate = _confidence_gate   # fork (exp025)" in code) == mg)
    return check


for _name, _aw, _mg in (("ntl-v31-dp", "0.10", False), ("ntl-v31-dp-a20", "0.20", False), ("ntl-v31-dp-a30", "0.30", False),
                        ("ntl-v31-dp-mg", "0.10", True), ("ntl-v31-dp-a20-mg", "0.20", True)):
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + ([("'--atlas_w', '0.10'", f"'--atlas_w', '{_aw}'")] if _aw != "0.10" else []) + (TYPO_MG if _mg else []),
        "check": v31_check(_aw, _mg),
        "intro": V31_INTRO + (f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%." if _aw != "0.10" else "")
        + (" The typography step keeps only its margin gate (no length or coverage gate; exp025)." if _mg else "")}

# 2026-10-29: avazaku's cache is gone (403 since ~06:00 UTC), so a new commit of the variants above has no Llama.
# alexchilton's public model alexchilton/llama-3-1-8b-base holds the same files: ntl-llama-mirror-check (12:15 UTC) found
# the sha256 of all four shards and the blob ids of config, generation config, tokenizer, index, LICENSE and USE_POLICY
# equal to those of the NousResearch copy the cache held (avazaku's PROTOCOL). "ac" variants read Llama from it.
LLAMA_AC = "alexchilton/llama-3-1-8b-base/Transformers/default/1"
MIRROR_LLAMA_AC = (
    "BL = mdir('/llama-3.1/', '/8b/')",
    "BL = mdir('/llama-3-1-8b-base/')\n"
    f"_SHARDS = {SHARDS!r}\n"
    "assert all(os.path.getsize(os.path.join(BL, f)) == n for f, n in _SHARDS.items()), BL   # fork: Llama-3.1-8B base from "
    "alexchilton's public model (the files of the NousResearch copy, sha256-checked: ntl-llama-mirror-check)")

# exp027 (2026-10-29): lightningv08's "0.661 | Qwen3.8-27B LoRA + Spectral Ensemble" (pulled 2026-10-29 12:20 UTC, kept
# byte-for-byte) is mercurydust's 0.660 recipe rebuilt from prefit caches plus his own Qwen3.8-27B GSQ3 LoRA readout
# ("GSQ": 3-bit packed base, LoRA on decoder layers 56-63, a linear head on mean-pooled states at 8 depths, full text,
# batch size 1, one worker per T4), mixed at 10% after ATLAS (10% there) and before the balancing: 0.661, +0.001. Our
# ATLAS 20% was +0.003 on the same base (#24), so these variants stack the two. His five gsq/ files run verbatim, in
# their own processes with his wheels (transformers 5.14.1), right after the comp stack; their wall time is charged to
# ATLAS's budget with the comp stack's (the 0.663 runs finished within 6.6 h, so ATLAS had ~1.5 h to spare). Any failure
# or overrun of GSQ_TIMEOUT leaves the run as it was without GSQ. His asset collection also holds configs, wheels and
# checkpoints that the pipeline's input searches would pick up (a bitsandbytes 0.50.2 wheel, Llama/Qwen bases in other
# layouts, a Qwen3.5-architecture config that would switch on the Q35 token-attention head), so HIDE_LV08 filters it out
# of glob.glob in the notebook and in comp_stack.py; the GSQ stage reaches it by explicit path.
LV08_FILE, LV08_SHA256 = "upstream_lightningv08_0661.ipynb", "c7558b61034609ead70c7d5a71a6505933d65c223c4c95e493bb229fbe58561a"
LV08_DATASET = "lightningv08/ntl-olmo2-native24-final-inference-assets"
LV08_GSQ_FILES = ("gsq_infer_model.py", "gsq_infer_layers.py", "delta_turing.py", "adapter.py", "infer.py")
GLOB_HIDE = ("import glob as _gl8; _gl8_glob = _gl8.glob\n"
             "_gl8.glob = lambda *a_, **k_: [p_ for p_ in _gl8_glob(*a_, **k_) if '/ntl-olmo2-native24-final-inference-assets/' "
             "not in p_]   # fork (exp027): lightningv08's asset collection stays out of the input searches\n")
HIDE_LV08 = [("import glob, os, subprocess, json, sys\n", "import glob, os, subprocess, json, sys\n" + GLOB_HIDE),
             ("import gc, glob, json, os, pathlib, re, time\n", "import gc, glob, json, os, pathlib, re, time\n" + GLOB_HIDE)]
GSQ_STAGE = r'''# fork (exp027): lightningv08's Qwen3.8-27B GSQ3 readout, from his "0.661 | Qwen3.8-27B LoRA + Spectral Ensemble"
# (https://www.kaggle.com/code/lightningv08/0-661-qwen3-8-27b-lora-spectral-ensemble); the next five cells are his gsq/ files.
import os, shutil, signal, subprocess, sys, time
import numpy as np
os.makedirs('/kaggle/temp/ntl-gsq-v27-runtime/gsq', exist_ok=True)
GSQ_TIMEOUT = 5400   # s from the start of the stage; past it the workers are stopped and the run goes on without GSQ
GSQ_WHEELS = ['accelerate-1.15.0-py3-none-any.whl', 'bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl',
              'huggingface_hub-1.32.0-py3-none-any.whl',
              'safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
              'tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', 'transformers-5.14.1-py3-none-any.whl']


def run_gsq(frame, out, timeout, log_path='/kaggle/working/gsq_stage.log'):
    """His two isolated T4 workers (gsq/infer.py, batch size 1, every token) on frame's texts, with his wheels in their
    own target directory as in his ensemble.py; writes ids, labels and plain logits to `out`. Each row is its own group
    (only his centred logits use groups, and his mixture uses the plain ones). Raises on any failure or after `timeout` s."""
    t0 = time.time()
    root = next(r for r in ('/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets',
                            '/kaggle/input/ntl-olmo2-native24-final-inference-assets')
                if os.path.exists(r + '/gsq3-base/qwen38-27b-gsq3/config.json'))
    final = root + '/gsq120k-trained/gsq120k-inference/gsq/'
    wheels = [root + '/name-that-llm-artifacts/qwen38-offline-linux/' + w for w in GSQ_WHEELS]
    target, work = '/kaggle/temp/gsq-python', '/kaggle/temp/gsq-work'
    os.makedirs(work, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--disable-pip-version-check', '-q',
                    '--target', target, *wheels], check=True)
    frame[['id', 'text']].to_parquet(work + '/query.parquet', index=False)
    np.save(work + '/groups.npy', np.arange(len(frame), dtype=np.int64), allow_pickle=False)
    base, free = root + '/gsq3-base/qwen38-27b-gsq3', shutil.disk_usage('/tmp').free
    if free > 20e9:   # one sequential local copy, as his stage does; random reads from the mount are slow
        shutil.copytree(base, '/tmp/gsq3-base', dirs_exist_ok=True); base = '/tmp/gsq3-base'
    copied = time.time() - t0
    env = dict(os.environ, PYTHONPATH=target, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false')
    cmd = [sys.executable, '-u', '/kaggle/temp/ntl-gsq-v27-runtime/gsq/infer.py', '--base', base,
           '--final', final + 'final.safetensors', '--metadata', final + 'final.json', '--complete', final + 'COMPLETE.json',
           '--input', work + '/query.parquet', '--groups', work + '/groups.npy', '--output', work + '/gsq.npz',
           '--work-dir', work + '/workers', '--batch-size', '1']
    try:
        with open(log_path, 'w') as lf:
            proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=max(60, timeout - (time.time() - t0)))
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait(); raise
        assert code == 0, f'gsq/infer.py exit code {code}'
    finally:
        shutil.rmtree('/tmp/gsq3-base', ignore_errors=True)
    with np.load(work + '/gsq.npz', allow_pickle=False) as z:
        assert np.array_equal(z['row_index'], np.arange(len(frame))) and np.isfinite(z['plain_logits']).all()
        np.savez(out, ids=np.array(frame.id.astype(str).tolist(), dtype=str), labels=z['labels'].astype(str),
                 plain_logits=z['plain_logits'])   # unicode arrays: np.load(allow_pickle=False) reads them
    return dict(rows=len(frame), seconds=round(time.time() - t0), copy_seconds=round(copied), tmp_free_gb=round(free / 1e9),
                workers_seconds=round(time.time() - t0 - copied))
'''


def lv08_gsq_cells():
    """[the stage cell, then his five gsq/ cells verbatim], each with the cell it follows (after comp_stack.py's)."""
    raw = (HERE / LV08_FILE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != LV08_SHA256:
        raise SystemExit(f"{LV08_FILE} is not the verified lightningv08 0.661 notebook")
    srcs = [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in json.loads(raw)["cells"]]
    cells, after = [("%%writefile comp_stack.py", GSQ_STAGE)], GSQ_STAGE.split("\n")[0]
    for f in LV08_GSQ_FILES:
        head = f"%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/{f}\n"
        hits = [s for s in srcs if s.startswith(head)]
        if len(hits) != 1:
            raise SystemExit(f"{LV08_FILE}: no single cell writes gsq/{f}")
        cells.append((after, hits[0])); after = head.rstrip("\n")
    return cells


GSQ_RUN = [("COMP_SEC = _t.time() - _c0\n",
            "GSQ_OUT = '/kaggle/working/gsq_plain.npz'   # fork (exp027): GSQ after the comp stack, inside its wall time\n"
            "try:\n"
            "    _gq = _pd.read_csv(os.path.join(DATA, 'test.csv'), dtype={'id': str}); _gq['text'] = _gq.text.fillna('').astype(str)\n"
            "    print('gsq', run_gsq(_gq, GSQ_OUT, GSQ_TIMEOUT), flush=True)\n"
            "except Exception as _e:\n"
            "    print('GSQ SKIPPED:', repr(_e), flush=True)\n"
            "    if os.path.exists(GSQ_OUT): os.remove(GSQ_OUT)\n"
            "COMP_SEC = _t.time() - _c0\n")]


def gsq_mix(w):
    return [
        ("p.add_argument('--comp_path', default=''); p.add_argument('--comp_w', type=float, default=0.0)\n",
         "p.add_argument('--comp_path', default=''); p.add_argument('--comp_w', type=float, default=0.0)\n"
         "p.add_argument('--gsq_path', default=''); p.add_argument('--gsq_w', type=float, default=0.0)   # fork (exp027)\n"),
        ("        P = mixed / mixed.sum(1, keepdims=True)\n    Q = P.copy()\n",
         "        P = mixed / mixed.sum(1, keepdims=True)\n"
         "    if a.gsq_w > 0 and a.gsq_path and os.path.exists(a.gsq_path):   # fork (exp027): lightningv08's GSQ readout, after ATLAS\n"
         "        try:\n"
         "            _gz = np.load(a.gsq_path, allow_pickle=False); _gi = {k: i for i, k in enumerate(_gz['ids'].astype(str))}\n"
         "            _gl = _gz['plain_logits'][[_gi[k] for k in test['id'].astype(str)]][:, [list(_gz['labels'].astype(str)).index(l) for l in LABELS]].astype(np.float64)\n"
         "            _G = np.exp(_gl - _gl.max(1, keepdims=True)); _G /= _G.sum(1, keepdims=True); assert np.isfinite(_G).all()\n"
         "            P = (1 - a.gsq_w) * P + a.gsq_w * _G; P /= P.sum(1, keepdims=True); log(f'gsq mixed at {a.gsq_w} on {len(_gl)} rows')\n"
         "        except Exception as _e:\n"
         "            log('GSQ SKIPPED:', repr(_e))\n"
         "    Q = P.copy()\n"),
        ("'--comp_path', '/kaggle/working/comp_logit.npy', '--comp_w', '0.15',\n",
         f"'--comp_path', '/kaggle/working/comp_logit.npy', '--comp_w', '0.15', '--gsq_path', GSQ_OUT, '--gsq_w', '{w}',\n")]


def v31ac_check(atlas_w, gsq_w):
    base = v31_check(atlas_w, False)

    def check(code):
        gsq = (code.count(GLOB_HIDE) == 2 and "def run_gsq(" in code and code.count("run_gsq(_gq, GSQ_OUT, GSQ_TIMEOUT)") == 1
               and f"'--gsq_w', '{gsq_w}'" in code and code.count("if a.gsq_w > 0 and a.gsq_path") == 1
               and code.count("%%writefile /kaggle/temp/ntl-gsq-v27-runtime/gsq/") == 5)
        return (base(code) and "mdir('/llama-3-1-8b-base/')" in code and "verified-llama-asset-cache" not in code
                and code.count("            if _pin is not None:") == 1
                and (gsq if gsq_w else ("run_gsq" not in code and GLOB_HIDE not in code)))
    return check


V31AC_INTRO = (" Llama-3.1-8B is read from alexchilton's public model (the same files, sha256-checked) since avazaku's cache was "
               "deleted.")
ATLAS_M45_V31 = [("'--atlas_margin', '0.3',", "'--atlas_margin', '0.45',")]
# Pin comp_stack's extra passages to the versions our 0.663 commit attached (2026-10-28 15:30 UTC). Pushes attach the
# current versions, and three sources have since appended batches: abdullah1afify/ntl-generations v4 (batch d1029),
# lucasboesen/ntl-free-proxy-generations v11 (v2b19-v2b20) and prvsiyan's extended dataset v7 (split new2, dataset
# version 5). Keeping only the batches each file had on 10-28 gives the old file exactly (rows, order, every column), and
# the loader's output is the 10,325 rows of 10-28 again instead of 10,599 (two public reruns of our a20 code with the new
# rows scored 0.662). A whitelist, so batches appended later are dropped too.
PIN_10_28 = [("            d = pd.read_csv(f)\n            if finish_col and finish_col in d:\n",
              "            d = pd.read_csv(f)\n"
              "            _pin = {'ntl_generations.csv': lambda d: d['batch'].astype(str).isin(\n"
              "                        {'b001', 'b001g', 'b003', 'b003a', 'd1028', 'pilot', 'pilot2', 'pilot3', 'pilotpro'}),\n"
              "                    'gen_passages.csv': lambda d: d['batch'].astype(str).isin(\n"
              "                        {f'b{i}' for i in (1, 2, *range(5, 16))} | {f'v2b{i:02d}' for i in range(1, 19)}),\n"
              "                    'passages_clean.csv': lambda d: pd.to_numeric(d['dataset_version_added'], errors='coerce').isin([1, 2, 4])\n"
              "                    }.get(os.path.basename(f))\n"
              "            if _pin is not None:   # fork: only the batches these files had on 2026-10-28 (later additions dropped)\n"
              "                try:\n"
              "                    d = d[_pin(d)]\n"
              "                except KeyError as _e:\n"
              "                    print('extra-passage pin skipped for', f, repr(_e), flush=True)\n"
              "            if finish_col and finish_col in d:\n")]
for _name, _aw, _gw, _m45 in (("ntl-v31ac-a20-gsq10", "0.20", "0.10", False), ("ntl-v31ac-a30-gsq10", "0.30", "0.10", False),
                              ("ntl-v31ac-a20-gsq20", "0.20", "0.20", False), ("ntl-v31ac-a20", "0.20", None, False),
                              ("ntl-v31ac-a40", "0.40", None, False), ("ntl-v31ac-a30-m45", "0.30", None, True)):
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA_AC] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + [("'--atlas_w', '0.10'", f"'--atlas_w', '{_aw}'")] + (ATLAS_M45_V31 if _m45 else []) + PIN_10_28
        + (HIDE_LV08 + GSQ_RUN + gsq_mix(_gw) if _gw else []),
        "cells": lv08_gsq_cells() if _gw else [],
        "check": (lambda chk, m45: lambda code: chk(code) and (("'--atlas_margin', '0.45'," in code) == m45))(
            v31ac_check(_aw, _gw), _m45),
        "models": [LLAMA_AC], "drop_kernels": [LLAMA_CACHE], "datasets": [LV08_DATASET] if _gw else [],
        "intro": V31_INTRO + f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%"
        + (" on rows with top-2 margin below 0.45 instead of 0.3." if _m45 else ".") + V31AC_INTRO
        + " The extra passages are pinned to their 2026-10-28 versions (the batches added since are dropped)."
        + (f" lightningv08's Qwen3.8-27B GSQ3 readout (his 0.661 notebook) is mixed at {round(float(_gw) * 100)}% after ATLAS, "
           "before the balancing, as there." if _gw else "")}

# exp028 (2026-10-30): exp027 left ATLAS-8B's weight on a plateau (20% and 30% both 0.663 on the 10-28 inputs; on the 10-29
# ones 40% -0.001 and GSQ -0.001 to -0.007 against 20%). The recipe's other test-time knobs were never moved on the LB, so
# 2026-10-31 brackets two of them on the pinned a20 code, one change each: the complementary stack's share of the first
# mixture (--comp_w 0.15 -> 0.10 / 0.22) and the balancing strength (--sinkhorn 0.45 -> 0.30 / 0.60; PeterDreamHan's
# softer Sinkhorn was +0.002 on nlm-v27), plus ATLAS-8B at 25%, the middle of the plateau.
KNOBS_V31 = {"comp10": ("'--comp_w', '0.15',", "'--comp_w', '0.10',"), "comp22": ("'--comp_w', '0.15',", "'--comp_w', '0.22',"),
             "sk30": ("'--sinkhorn', '0.45',", "'--sinkhorn', '0.30',"), "sk60": ("'--sinkhorn', '0.45',", "'--sinkhorn', '0.60',")}
KNOB_INTRO = {"comp10": " The complementary stack's share of the first mixture is 10% instead of 15%.",
              "comp22": " The complementary stack's share of the first mixture is 22% instead of 15%.",
              "sk30": " The balancing goes 30% of the way to the balanced log-probabilities instead of 45% (--sinkhorn 0.30).",
              "sk60": " The balancing goes 60% of the way to the balanced log-probabilities instead of 45% (--sinkhorn 0.60)."}


def knob_check(atlas_w, knob):
    base = v31ac_check(atlas_w, None)
    old, new = KNOBS_V31[knob] if knob else ("", "")

    def check(code):
        if knob and (code.count(new) != 1 or old in code):
            return False
        return base(code.replace(new, old) if knob else code) and "'--atlas_margin', '0.3'," in code
    return check


for _name, _aw, _knob in (("ntl-v31ac-a25", "0.25", None), ("ntl-v31ac-a20-comp10", "0.20", "comp10"),
                          ("ntl-v31ac-a20-comp22", "0.20", "comp22"), ("ntl-v31ac-a20-sk30", "0.20", "sk30"),
                          ("ntl-v31ac-a20-sk60", "0.20", "sk60")):
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA_AC] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + [("'--atlas_w', '0.10'", f"'--atlas_w', '{_aw}'")] + PIN_10_28 + ([KNOBS_V31[_knob]] if _knob else []),
        "check": knob_check(_aw, _knob),
        "models": [LLAMA_AC], "drop_kernels": [LLAMA_CACHE],
        "intro": V31_INTRO + f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%." + V31AC_INTRO
        + " The extra passages are pinned to their 2026-10-28 versions (the batches added since are dropped)."
        + (KNOB_INTRO[_knob] if _knob else "")}
# exp029: ATLAS-8B retrained with Joseph Adamski's code and config on every official row plus 500 open-pool passages per
# label (kernel ntl-atlas-full-ext, kaggle/atlas/), loaded from that kernel's output in place of his bundle (still
# attached, so the search names ours); everything else is the pinned a20 code. Built only if -ext wins on fold 0.
ATLAS_KERNEL = "omerfarukmerey/ntl-atlas-full-ext"
ATLAS_X = [("ATLAS = os.path.dirname(next(iter(glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True))))",
            "ATLAS = os.path.dirname(next(x for x in glob.glob('/kaggle/input/**/atlas_bundle.json', recursive=True) "
            "if 'ntl-atlas-full-ext' in x))   # fork (exp029): ATLAS-8B retrained with open-pool passages")]
for _name, _aw in (("ntl-v31ac-a20-atlasx", "0.20"), ("ntl-v31ac-a30-atlasx", "0.30")):
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA_AC] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + [("'--atlas_w', '0.10'", f"'--atlas_w', '{_aw}'")] + PIN_10_28 + ATLAS_X,
        "check": (lambda chk: lambda code: chk(code) and code.count("if 'ntl-atlas-full-ext' in x))") == 1)(
            knob_check(_aw, None)),
        "models": [LLAMA_AC], "drop_kernels": [LLAMA_CACHE], "kernels": [ATLAS_KERNEL],
        "intro": V31_INTRO + f" ATLAS-8B is mixed at {round(float(_aw) * 100)}% instead of 10%, and its adapters are "
        "our retraining with Joseph Adamski's code and configuration on the official rows plus 500 public real-model "
        "passages per label (crtproduction/name-that-llm-open-pool)." + V31AC_INTRO
        + " The extra passages are pinned to their 2026-10-28 versions (the batches added since are dropped)."}
# exp029 on fold 0: the retrained ATLAS-8B is +0.065 alone (0.406 vs 0.341 without the open-pool passages), yet the linear
# mix gives the same picks with either: ATLAS's scores are nearly flat (top ~0.067 of 18, top-2 margin 0.002), so the 20%
# mix only shrinks those rows towards uniform (+0.0037 in exp015's replica). Multiplying the mixed rows by ATLAS**k uses
# its evidence: k 3-6 is +0.015 to +0.017 on the 20% linear mix (cross-fitted k +0.009), while the retraining without the
# open-pool passages loses 0.004-0.024 the same way. Same code as -atlasx plus --atlas_pow k.
ATLAS_POW_ARG = ("p.add_argument('--atlas_margin', type=float, default=0.0, help='>0: score ATLAS only on rows whose top-2 margin is below this')",
                 "p.add_argument('--atlas_margin', type=float, default=0.0, help='>0: score ATLAS only on rows whose top-2 margin is below this')\n"
                 "p.add_argument('--atlas_pow', type=float, default=0.0, help='fork (exp029): >0 also multiplies the mixed rows by ATLAS**pow')")
ATLAS_POW_MIX = ("                    mixed[completed_rows] += _w * (A_[scored] - P[completed_rows])\n",
                 "                    mixed[completed_rows] += _w * (A_[scored] - P[completed_rows])\n"
                 "                    if a.atlas_pow > 0:   # fork (exp029): ATLAS's scores are nearly flat, so the linear mix rarely moves\n"
                 "                        mixed[completed_rows] *= A_[scored] ** a.atlas_pow   # a pick; this sharpened evidence does\n")


def atlas_pow(k):
    return [ATLAS_POW_ARG, ATLAS_POW_MIX, ("'--atlas_margin', '0.3',", f"'--atlas_margin', '0.3', '--atlas_pow', '{k}',")]


def pow_check(k):
    base = knob_check("0.20", None)

    def check(code):
        return (base(code) and code.count("if 'ntl-atlas-full-ext' in x))") == 1 and code.count("a.atlas_pow") == 2
                and code.count(f"'--atlas_margin', '0.3', '--atlas_pow', '{k}',") == 1)
    return check


for _name, _k in (("ntl-v31ac-a20-atlasx-p2", "2"), ("ntl-v31ac-a20-atlasx-p4", "4"), ("ntl-v31ac-a20-atlasx-p6", "6")):
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA_AC] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + [("'--atlas_w', '0.10'", "'--atlas_w', '0.20'")] + PIN_10_28 + ATLAS_X + atlas_pow(_k),
        "check": pow_check(_k),
        "models": [LLAMA_AC], "drop_kernels": [LLAMA_CACHE], "kernels": [ATLAS_KERNEL],
        "intro": V31_INTRO + " ATLAS-8B is mixed at 20% instead of 10%, and its adapters are our retraining with Joseph "
        "Adamski's code and configuration on the official rows plus 500 public real-model passages per label "
        f"(crtproduction/name-that-llm-open-pool). After that mix, the same rows are multiplied by ATLAS's scores to the "
        f"power {_k} (renormalised), so its evidence can move a pick (its scores are nearly flat)." + V31AC_INTRO
        + " The extra passages are pinned to their 2026-10-28 versions (the batches added since are dropped)."}
# 2026-11-01: exp028's Sinkhorn 0.30 scored 0.664 (0.45: 0.663, 0.60: 0.658), tying the Code tab's two public 0.664s,
# both forks of our 0.663: zbigniewkirchner trains the 4-view SVC on crtproduction's open pool (198k public real-model
# passages, CC BY-SA 4.0; regenerated train prompts dropped) instead of nlm_pool_v5 and raises --svc_w to 12; judith007
# runs Sinkhorn 0.60 with V31's typography evidence at x1.2 instead of x1.5. On the sk30 code: each of their changes, our
# exp029 power mix, all three together, and Sinkhorn 0.20.
OPEN_POOL = "crtproduction/name-that-llm-open-pool"
SK = {"sk30": KNOBS_V31["sk30"], "sk20": ("'--sinkhorn', '0.45',", "'--sinkhorn', '0.20',")}
OSVC_POOL = ("POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))\n",
             "POOL = next(iter(glob.glob('/kaggle/input/**/nlm_pool_v5.parquet', recursive=True)))\n"
             "# fork (zbigniewkirchner's 0.664): the SVC learns from crtproduction's open pool (198k public real-model passages,\n"
             "# CC BY-SA 4.0) instead of nlm_pool_v5, minus the passages regenerated from train prompts\n"
             "_op = _pd.read_parquet(next(iter(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))), "
             "columns=['text', 'model', 'flag_train_prompt_regen'])\n"
             "_op[~_op.flag_train_prompt_regen.astype(bool)][['text', 'model']].to_parquet('/kaggle/working/open_pool.parquet')\n"
             "POOL = '/kaggle/working/open_pool.parquet'; print('svc pool = open pool', len(_op)); del _op\n")


def osvc(w):
    return [OSVC_POOL, ("'--svc_w', '1.5',", f"'--svc_w', '{w}',")]


TYPO12 = ("    _evidence = np.clip(1.5 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)\n",
          "    _evidence = np.clip(1.2 * _evidence / max(float(_fw.sum()), 1e-6), -0.30, 0.30)   # fork: x1.2 instead of x1.5 (judith007's 0.664)\n")


def combo_check(knobs, atlas_k):
    base = pow_check(atlas_k) if atlas_k else knob_check("0.20", None)

    def check(code):
        for old, new in knobs:   # each change present once; reverted, the code must pass the pinned a20 checks
            if code.count(new) != 1 or (old not in new and old in code):
                return False
            code = code.replace(new, old)
        return base(code) and (atlas_k is not None or "atlas_pow" not in code)
    return check


for _name, _sk, _osvc, _k, _typo in (("ntl-v31ac-a20-sk30-osvc12", "sk30", "12", None, False),
                                     ("ntl-v31ac-a20-sk30-osvc15", "sk30", "1.5", None, False),
                                     ("ntl-v31ac-a20-sk30-osvc20", "sk30", "20", None, False),   # exp030: w 20 best OOF
                                     ("ntl-v31ac-a20-sk30-ax4", "sk30", None, "4", False),
                                     ("ntl-v31ac-a20-sk30-typo12", "sk30", None, None, True),
                                     ("ntl-v31ac-a20-sk30-osvc12-ax4-typo12", "sk30", "12", "4", True),
                                     ("ntl-v31ac-a20-sk30-osvc12-ax4", "sk30", "12", "4", False),
                                     ("ntl-v31ac-a20-sk20", "sk20", None, None, False)):
    _knobs = [SK[_sk]] + (osvc(_osvc) if _osvc else []) + ([TYPO12] if _typo else [])
    VARIANTS[_name] = {
        "base": "v31",
        "patches": [MIRROR_LLAMA_AC] + LLAMA_COPY + FAST_COMMIT_V31 + ATLAS_TO_DEADLINE_V31 + L24_DP
        + [("'--atlas_w', '0.10'", "'--atlas_w', '0.20'")] + PIN_10_28 + (ATLAS_X + atlas_pow(_k) if _k else []) + _knobs,
        "check": combo_check([k_ for k_ in _knobs if k_[0] != "'--svc_w', '1.5',"] if _osvc == "1.5" else _knobs, _k),
        "models": [LLAMA_AC], "drop_kernels": [LLAMA_CACHE], "kernels": [ATLAS_KERNEL] if _k else [],
        "datasets": [OPEN_POOL] if _osvc else [],
        "intro": V31_INTRO + " ATLAS-8B is mixed at 20% instead of 10%" + (
            ", and its adapters are our retraining with Joseph Adamski's code and configuration on the official rows plus "
            "500 public real-model passages per label (crtproduction/name-that-llm-open-pool); after that mix the same rows "
            f"are multiplied by ATLAS's scores to the power {_k} (renormalised)." if _k else ".")
        + f" The balancing goes {int(_sk[2:])}% of the way to the balanced log-probabilities instead of 45%."
        + (f" The 4-view SVC learns from crtproduction's open pool instead of nlm_pool_v5, at weight {_osvc} "
           "(zbigniewkirchner's 0.664)." if _osvc else "")
        + (" The typography step's evidence is scaled by 1.2 instead of 1.5 (judith007's 0.664)." if _typo else "")
        + V31AC_INTRO + " The extra passages are pinned to their 2026-10-28 versions (the batches added since are dropped)."}
BASES = {"nlm-v27": (BASE, BASE_SHA256, "upstream_nlm_v27.meta.json"),
         "v31": (V31_FILE, V31_SHA256, "upstream_mercury_v31.meta.json")}


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def build_variant(kernel):
    variant = VARIANTS[kernel]
    base = variant.get("base", "nlm-v27")
    base_file, base_sha256, base_meta = BASES[base]
    raw = (HERE / base_file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != base_sha256:
        raise SystemExit(f"{base_file} is not the verified {base} base")
    nb = json.loads(raw)
    meta_in = json.loads((HERE / base_meta).read_text(encoding="utf-8"))
    for c in nb["cells"]:
        c["source"] = text(c)
    for old, new in variant["patches"]:
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and old in c["source"]]
        if len(hits) != 1 or nb["cells"][hits[0]]["source"].count(old) != 1:
            raise SystemExit(f"{kernel}: patch must match exactly once: {old!r}")
        nb["cells"][hits[0]]["source"] = nb["cells"][hits[0]]["source"].replace(old, new)
    for after, source in variant.get("cells", []):   # new code cells, each right after the one cell starting with `after`
        hits = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c["source"].startswith(after)]
        if len(hits) != 1:
            raise SystemExit(f"{kernel}: no single cell starts with {after!r}")
        nb["cells"].insert(hits[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                         "source": source})
    code = "".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    if not variant["check"](code):
        raise SystemExit(f"{kernel}: the patched code fails its check")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": f"# {kernel}\n\n{variant['intro']}"})

    out = HERE / kernel
    out.mkdir(exist_ok=True)
    body = json.dumps(nb, indent=1, ensure_ascii=False)
    (out / f"{kernel}.ipynb").write_text(body, encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb",
            **{k: meta_in[k] for k in ("language", "kernel_type", "enable_gpu", "enable_tpu", "enable_internet",
                                       "dataset_sources", "kernel_sources", "competition_sources", "docker_image",
                                       "machine_shape")},
            "is_private": True, "keywords": [],
            "model_sources": [m for m in meta_in["model_sources"] if m != LLAMA]}
    meta["dataset_sources"] = meta["dataset_sources"] + variant.get("datasets", [])
    meta["kernel_sources"] = [k for k in meta["kernel_sources"] if k not in variant.get("drop_kernels", [])] + variant.get("kernels", [])
    if meta["enable_internet"] or len(meta["model_sources"]) != len(meta_in["model_sources"]) - (base == "nlm-v27"):
        raise SystemExit(f"{kernel}: unexpected upstream metadata")
    meta["model_sources"] = meta["model_sources"] + variant.get("models", [])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}/{kernel}.ipynb: {len(nb['cells'])} cells, {len(variant['patches'])} patches, "
          f"{len(meta['model_sources'])} models, sha256 {hashlib.sha256(body.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
        build_variant(name)
