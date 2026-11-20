"""Private fork of areebbb's public NTL-40 ("0.702 | Read the Fingerprints", pulled 2026-11-17 14:00 UTC; public 0.702 =
lightningv08's 0.702 code cells unchanged + Open Pool sibling specialists, sections 15 and 19). exp056 replicated the
specialists out of fold: +0.0038 [-0.004, +0.012] on our stand-in after the 0.702's tail, +0.037 on the public reader
OOF bank.

    ntl-702-spec        the notebook's code unchanged (a provenance cell on top)
    ntl-702-spec-paths  a CPU kernel with the same inputs: resolves every pinned path and checkpoint of
                        pinned_assets.py under this account's mount layout and runs spec_worker.py on the 3-row commit
                        test, so a version saved without a run (no GPU quota) is checked before it is submitted
    ntl-702-spec-op8    + a second sibling specialist (exp057): our TPU kernel ntl-tpu-op8's stage-1 model, a Qwen3-8B-Base
                        LoRA24 classifier trained on open pool v13 only (op8v1_s1). After the GLM stage, if at least
                        40 min of the 9 h remain, op8s1_worker.py runs infer.py's own l24_predict (extracted from
                        infer.py's source) on the test; its centred within-lab log-probs are added inside Google / OpenAI /
                        Anthropic at w 0.40 (lab mass kept), rescaled by the stack's within-lab spread against the
                        stand-in's. OOF on top of NTL-40: +0.0121 [+0.0041, +0.0204] on the stand-in (5/5 folds >= 0),
                        +0.032 on the OOF bank. Any failure leaves NTL-40's scores.
    ntl-702-spec-op8-paths  its CPU check: the pinned paths incl. op8v1_s1, and op8s1_worker.py on CPU (fp16) on 3
                        train rows against the TPU's own s1_on_train.npy
    ntl-702-gbm         NTL-40 with the GLM stage's head = exp050 gbmpair: his head (NTL-40's ll_model.json byte for
                        byte) averaged 50/50 with depth-2 boosted trees on his features + 12 per-token Flash / GLM-5.3
                        contrasts, through build_graft.py's v1_stage.py edits (as #108 graft-gbm 0.697 vs the graft's
                        0.696, #109 kiro-gbm 0.701 = kiro); a failure of the extra evidence falls back to his head
    ntl-702-gbm-c22     + lightningv08's apex-style-15 stage budget (cover 0.22, 130,000 token rows; public 0.702 alone)
    ntl-lv702-gbm       the same head on lightningv08's public 0.702 (soft typography, pulled 2026-11-18), i.e. NTL-40
                        without the Open Pool specialists: #113 (ntl-702-gbm) scored 0.690 and #111 (+ op8) 0.684, so
                        this checks whether NTL-40's own specialists cost both (its 0.702 on the Code tab can be NTL-39)
    ntl-lv702a-gbm      the same head on lightningv08's apex-style-15 (public 0.702: the soft typography + a reciprocal
                        prompt-pair adjustment and typography penalty 0.15), held until #114 (ntl-lv702-gbm) scores
    ntl-lv704-gbm       the same head on lightningv08's public 0.704 (GLM Fable Routing, pulled 2026-11-19: the soft
                        typography + stage rows ranked by GLM mass + 0.35 min(claude-fable-5 mass, GLM mass), post-stage
                        probabilities for the typography checks of GLM winners, penalty 0.15); #114 put the head at
                        +0.000 on the 0.702
    ntl-lv704-gbm-c22   + the stage at cover 0.22 / 130,000 token rows: the routed Fable rows no longer push GLM rows out
                        of the fixed 15% / 90k budget
    ntl-lv704-w125      the public 0.704 with his head (no gbmpair) and the stage's blend weight 1.25 W, sanity guard 0.32:
                        bill123987's public ntl706w (pulled 2026-11-19) made only that change to the 0.702 soft
                        typography and scored 0.704; exp058 puts W 1.25 at +0.0015 [-0.0000, +0.0033] with his head
                        under the Fable routing (cover 0.15), and flat with gbmpair
    ntl-lv704-w125-c22  + the stage at cover 0.22 / 130,000 token rows (exp058: his W 1.25 +0.0025 [-0.0000, +0.0055] at
                        cover 0.22 under the routing); submitted only if #116 (c22) beats #117 on the 0.704
    ntl-lv704-w15       the public 0.704 with the blend at 1.5 W (guard 0.32)
    ntl-lv704-w125-nof  ntl-lv704-w125 without the Fable routing (stage rows ranked by GLM mass, as in the soft 0.702 and
                        ntl706w): the 0.704's typography changes (post-stage probabilities for GLM winners, penalty
                        0.15) + 1.25 W
    ntl-lv704-w125-p05  ntl-lv704-w125 with the typography penalty 0.05 (the soft 0.702's and ntl706w's) instead of 0.15

    python kaggle/nlm/build_ntl40.py
"""
import hashlib
import json
from pathlib import Path

import build_lv694 as L
from build_graft import patched_stage
from build_l681 import text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_areebbb_ntl40_1117.ipynb"
UPSTREAM_SHA = "57e6db91e16cb672"
META = HERE / "upstream_areebbb_ntl40_1117.meta.json"
URL = "https://www.kaggle.com/code/areebbb/0-702-read-the-fingerprints"
SOFT = HERE / "upstream_lightningv08_702_soft_1118.ipynb"
SOFT_SHA = "a30c63085c331510"
SOFT_META = HERE / "upstream_lightningv08_702_soft_1118.meta.json"
SOFT_URL = "https://www.kaggle.com/code/lightningv08/0-702-soft-typography-glm-qwen3-8-27b-lora"
APEX = HERE / "upstream_lightningv08_702_apex15_1118.ipynb"
APEX_SHA = "f9948b633b22bd5e"
APEX_META = HERE / "upstream_lightningv08_702_apex15_1118.meta.json"
APEX_URL = "https://www.kaggle.com/code/lightningv08/0-702-apex-style-15-glm-qwen3-8-27b-lora"
FABLE = HERE / "upstream_lightningv08_704_fable_1119.ipynb"
FABLE_SHA = "75736a7af0b431f8"
FABLE_META = HERE / "upstream_lightningv08_704_fable_1119.meta.json"
FABLE_URL = "https://www.kaggle.com/code/lightningv08/0-704-glm-fable-routing-qwen3-8-27b-lora"
BILL = HERE / "upstream_bill123987_ntl706w_1119.ipynb"
BILL_SHA = "529d1d9036a04fa7"
BILL_URL = "https://www.kaggle.com/code/bill123987/ntl706w"

CHECK = r'''import glob, os, subprocess, sys
import pinned_assets as A
bad = []
for key in A.PATHS:
    try:
        A.asset_paths(key)
    except AssertionError as e:
        bad.append((key, str(e)[:300]))
for key in A.CHECKPOINTS:
    try:
        A.checkpoint(*key)
    except AssertionError as e:
        bad.append((str(key), str(e)[:300]))
print('PATHS', len(A.PATHS), 'CHECKPOINTS', len(A.CHECKPOINTS), 'unresolved', len(bad), flush=True)
for b in bad:
    print('  BAD', b, flush=True)
print('mount roots', sorted(os.listdir('/kaggle/input')), flush=True)
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
print('pool', glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True), flush=True)
r = subprocess.run([sys.executable, 'spec_worker.py', DATA + '/test.csv', DATA + '/train.csv', '/kaggle/working/spec_scores.npz'],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], 'spec_worker exit', r.returncode, flush=True)
import numpy as np
with np.load('/kaggle/working/spec_scores.npz') as s:
    print({k: s[k].shape for k in s.files}, flush=True)
assert not bad and r.returncode == 0
print('PATHS OK', flush=True)
'''


OP8_KERNEL = "omerfarukmerey/ntl-tpu-op8"
OP8_PINS = '''
# ours (exp057): our TPU kernel ntl-tpu-op8's stage-1 adapter, either mount layout
MOUNT_ALIASES['/kaggle/input/ntl-tpu-op8'] = '/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-op8'
PATHS['/kaggle/input/**/op8v1_s1/heads.pt'] = ['/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-op8/op8v1_s1/heads.pt']
PATHS['/kaggle/input/**/s1_on_train.npy'] = ['/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-op8/s1_on_train.npy']
'''

WORKER = r'''%%writefile op8s1_worker.py
"""ours (exp057): log-probs of ntl-tpu-op8's stage-1 model (Qwen3-8B-Base LoRA24 trained on the open pool only) on a
csv of texts, through infer.py's own l24_predict: its source is taken from infer.py at run time and run unchanged.

    python op8s1_worker.py <csv with id,text> <Qwen3-8B-Base dir> <op8v1_s1 dir> <out.npz>
"""
import ast, gc, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd, torch
from pipe import pipelined, spans_devices
from transformers import AutoTokenizer, AutoModel
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
T0 = time.time()


def log(*x):
    print(f'[{time.time() - T0:7.0f}s]', *x, flush=True)


SRC = open(os.path.join(HERE, 'infer.py')).read()
LABELS = ast.literal_eval(SRC.split('\nLABELS = ', 1)[1].split('\n', 1)[0])
_fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == 'l24_predict')
_lines = SRC.splitlines(True)   # the function with its decorators (@torch.no_grad()), exactly as infer.py has it
exec(compile(''.join(_lines[min([d.lineno for d in _fn.decorator_list] + [_fn.lineno]) - 1:_fn.end_lineno]),
             'infer.py:l24_predict', 'exec'))

csv, base, adapter, out = sys.argv[1:5]
frame = pd.read_csv(csv)
frame['text'] = frame.text.fillna('')
spec = {'name': 'op8s1', 'path': base, 'adapters': [adapter], 'keep': 24, 'layers': [2, 4, 6, 8, 12, 16, 20, 24],
        'bs': 16, 'max_len': 384, 'dp': torch.cuda.device_count() >= 2}
plain, _ = l24_predict(spec, frame.text.values, None)
assert plain.shape == (len(frame), len(LABELS)) and np.isfinite(plain).all()
np.savez(out, ids=frame.id.astype(str).values.astype('U'), labels=np.array(LABELS), logp=np.asarray(plain, np.float64))
log('op8s1 done', plain.shape)
'''

APPLY = r'''# ---- ours (exp057): a second sibling specialist, our TPU kernel ntl-tpu-op8's stage-1 model (a Qwen3-8B-Base LoRA24
# classifier trained on open pool v13 only). Its centred within-lab log-probs are added inside Google, OpenAI and
# Anthropic at OP8_W with the lab's mass kept, like the TF-IDF specialists above; OP8_W is the out-of-fold optimum on
# our stand-in stack on top of NTL-40 (+0.0121 [+0.0041, +0.0204], 5/5 folds >= 0), rescaled by the stack's within-lab
# spread against the stand-in's (OP8_REF). It runs after the GLM stage, only with 40 min of the 9 h left, and any
# failure leaves the scores as they are.
from scipy.special import log_softmax as _lsm8
OP8_W = 0.40
OP8_REF = {'gemini': 1.7077, 'gpt': 1.9925, 'claude': 2.3446}   # stand-in's mean within-lab std after NTL-40's step
OP8_OUT = '/kaggle/temp/op8s1_test.npz'
OP8_DIR = adapter_dir('op8v1_s1')
B8_OP8 = asset_path('/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1')


def op8_apply(logits_path, op8_path):
    with np.load(logits_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    labels = arrays['labels'].astype(str).tolist()
    L = np.asarray(arrays['logits'], dtype=np.float64).copy()
    with np.load(op8_path, allow_pickle=False) as s:
        assert s['ids'].astype(str).tolist() == arrays['ids'].astype(str).tolist(), 'op8 ids differ'
        S18 = np.asarray(s['logp'], np.float64)[:, [s['labels'].astype(str).tolist().index(l) for l in labels]]
    assert np.isfinite(S18).all()
    for lab, ref in OP8_REF.items():
        idx = [i for i, l in enumerate(labels) if l.startswith(lab + '-')]
        S = _lsm8(S18[:, idx], axis=1)
        S = S - S.mean(1, keepdims=True)
        inlab = np.isin(L.argmax(1), idx)
        spread = L[inlab][:, idx].std(1).mean() if inlab.sum() >= 20 else ref
        w = OP8_W * float(np.clip(spread / ref, 0.5, 2.5))
        before = L.argmax(1)
        lse = np.logaddexp.reduce(L[:, idx], axis=1, keepdims=True)
        L[:, idx] = lse + _lsm8(L[:, idx] + w * S, axis=1)
        print(f'op8 specialist {lab}: weight {w:.3f} (spread {spread:.3f}), top label changed on '
              f'{int((L.argmax(1) != before).sum())} of {len(L)} rows', flush=True)
    assert np.isfinite(L).all()
    arrays['logits'] = L.astype(np.asarray(arrays['logits']).dtype)
    return arrays


_op8_cmd = [sys.executable, 'op8s1_worker.py', os.path.join(DATA, 'test.csv'), B8_OP8, OP8_DIR, OP8_OUT]
if FAST_COMMIT:   # loads the base and the adapter on the T4s and scores the commit's test rows
    subprocess.run(_op8_cmd, check=True)
    _ids = np.load(OP8_OUT)['ids']
    _labs = sorted(pd.read_csv(os.path.join(DATA, 'train.csv')).model.unique())
    np.savez('/kaggle/temp/op8-smoke-logits.npz', ids=_ids, labels=np.array(_labs),
             logits=np.random.default_rng(0).normal(size=(len(_ids), len(_labs))))
    op8_apply('/kaggle/temp/op8-smoke-logits.npz', OP8_OUT)
    print('op8 specialist smoke ok', flush=True)
else:
    try:
        _left = 8.75 * 3600 - (time.time() - NB_T0)
        assert _left > 1800, f'only {_left:.0f}s left, op8 specialist not run'
        _r = subprocess.run(_op8_cmd, timeout=_left, capture_output=True, text=True)
        print(_r.stdout[-1500:], _r.stderr[-1500:], flush=True)
        assert _r.returncode == 0, _r.returncode
        np.savez(FINAL_LOGITS, **op8_apply(FINAL_LOGITS, OP8_OUT))
    except Exception as e:
        print('op8 specialist skipped:', repr(e)[:300], flush=True)
print('elapsed', round(time.time() - NB_T0), 's', flush=True)
'''

OP8_CHECK = r'''import glob, os, subprocess, sys, time
import numpy as np, pandas as pd
import pinned_assets as A
bad = []
for key in A.PATHS:
    try:
        A.asset_paths(key)
    except AssertionError as e:
        bad.append((key, str(e)[:300]))
for key in A.CHECKPOINTS:
    try:
        A.checkpoint(*key)
    except AssertionError as e:
        bad.append((str(key), str(e)[:300]))
print('PATHS', len(A.PATHS), 'CHECKPOINTS', len(A.CHECKPOINTS), 'unresolved', len(bad), flush=True)
for b in bad:
    print('  BAD', b, flush=True)
op8 = os.path.dirname(A.asset_paths('/kaggle/input/**/op8v1_s1/heads.pt')[0])
print('op8v1_s1', op8, sorted(os.listdir(op8)), flush=True)
B8 = A.asset_path('/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1')
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train = pd.read_csv(DATA + '/train.csv')
rows = np.argsort(train.text.fillna('').str.len().to_numpy())[:3]   # the 3 shortest train passages: CPU fp16 is slow
train.iloc[rows][['id', 'text']].to_csv('/kaggle/working/check3.csv', index=False)
t = time.time()
r = subprocess.run([sys.executable, 'op8s1_worker.py', '/kaggle/working/check3.csv', B8, op8, '/kaggle/working/check3.npz'],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], 'worker exit', r.returncode, round(time.time() - t), 's', flush=True)
if r.returncode == 0:
    z = np.load('/kaggle/working/check3.npz')
    s1 = np.load(A.asset_paths('/kaggle/input/**/s1_on_train.npy')[0])[rows]
    tpu = s1 - np.log(np.exp(s1 - s1.max(1, keepdims=True)).sum(1, keepdims=True)) - s1.max(1, keepdims=True)
    cpu = z['logp']
    print('labels', z['labels'].tolist()[:3], '...', flush=True)
    print('argmax cpu', cpu.argmax(1).tolist(), 'tpu', tpu.argmax(1).tolist(), 'max |log-prob diff|',
          float(np.abs(cpu - tpu).max()), flush=True)
assert not bad
print('PATHS OK', flush=True)
'''


Q14_KERNEL = "omerfarukmerey/ntl-tpu-q14v16"
Q14_PINS = '''
# ours (exp057): our TPU kernel ntl-tpu-q14v16's stage-1 adapter (Qwen3-14B-Base, open pool v16 only), either layout
MOUNT_ALIASES['/kaggle/input/ntl-tpu-q14v16'] = '/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-q14v16'
PATHS['/kaggle/input/**/q14op16_s1/heads.pt'] = ['/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-q14v16/q14op16_s1/heads.pt']
PATHS['/kaggle/input/**/q14op16_s1_on_train.npy'] = ['/kaggle/input/notebooks/omerfarukmerey/ntl-tpu-q14v16/q14op16_s1_on_train.npy']
'''


def _q14_worker():
    """op8s1_worker.py with ntl-tpu-q14v16's recipe (tpu_member.py --keep 27 --layers 2,...,27 --max-len 447 --eos-id
    151643, the same trainer as op8, so infer.py's default final norm); ids read as strings."""
    w = WORKER
    for old, new in [
            ("%%writefile op8s1_worker.py", "%%writefile q14s1_worker.py"),
            ("log-probs of ntl-tpu-op8's stage-1 model (Qwen3-8B-Base LoRA24 trained on the open pool only)",
             "log-probs of ntl-tpu-q14v16's stage-1 model (Qwen3-14B-Base LoRA trained on open pool v16 only)"),
            ("python op8s1_worker.py <csv with id,text> <Qwen3-8B-Base dir> <op8v1_s1 dir> <out.npz>",
             "python q14s1_worker.py <csv with id,text> <Qwen3-14B-Base dir> <q14op16_s1 dir> <out.npz>"),
            ("frame = pd.read_csv(csv)", "frame = pd.read_csv(csv, dtype={'id': str})"),
            ("spec = {'name': 'op8s1', 'path': base, 'adapters': [adapter], 'keep': 24, 'layers': [2, 4, 6, 8, 12, 16, 20, 24],\n"
             "        'bs': 16, 'max_len': 384, 'dp': torch.cuda.device_count() >= 2}",
             "spec = {'name': 'q14s1', 'path': base, 'adapters': [adapter], 'keep': 27, 'layers': [2, 4, 6, 9, 13, 17, 22, 27],\n"
             "        'bs': 8, 'max_len': 447, 'eos_id': 151643}"),
            ("log('op8s1 done', plain.shape)", "log('q14s1 done', plain.shape)")]:
        assert w.count(old) == 1, old
        w = w.replace(old, new)
    return w


Q14_WORKER = _q14_worker()

Q14_APPLY = r'''# ---- ours (exp057): the lab decision from a second pool-only model, our TPU kernel ntl-tpu-q14v16's stage-1 model (a
# Qwen3-14B-Base LoRA classifier trained on open pool v16 only). On the quarter of the rows whose two largest lab masses
# are closest, each lab's mass moves by Q14_A x its centred lab log-prob under that model; the split inside each lab and
# each row's normaliser are kept. Out of fold on our stand-in, after the GLM stage, on top of NTL-40 + op8:
# +0.0026 [+0.0004, +0.0052] (4/5 folds >= 0), and scoring only that quarter keeps all of it. Q14_A is rescaled by the
# median lab margin against the stand-in's (Q14_REF). It runs after op8, only with 40 min of the 9 h left; any failure
# leaves the scores as they are.
from scipy.special import log_softmax as _lsm14
Q14_A = 0.05
Q14_FRAC = 0.25
Q14_REF = 2.61    # stand-in's median lab margin (nats) after NTL-40 + op8
Q14_OUT = '/kaggle/temp/q14s1_test.npz'
Q14_ROWS = '/kaggle/temp/q14_rows.csv'
Q14_DIR = adapter_dir('q14op16_s1')


def q14_labs(labels):
    """column -> lab index (claude, gemini, glm, gpt, kimi, qwen3.8) and the number of labs"""
    labs = sorted({l.split('-')[0] for l in labels})
    return np.array([labs.index(l.split('-')[0]) for l in labels]), len(labs)


def q14_labmass(L, col, n):
    P = _lsm14(L, axis=1)
    return np.stack([np.logaddexp.reduce(P[:, col == j], axis=1) for j in range(n)], 1)


def q14_rows(logits_path):
    with np.load(logits_path, allow_pickle=False) as z:
        ids, labels, L = z['ids'].astype(str), z['labels'].astype(str).tolist(), np.asarray(z['logits'], np.float64)
    col, n = q14_labs(labels)
    lm = np.sort(q14_labmass(L, col, n), axis=1)
    margin = lm[:, -1] - lm[:, -2]
    return ids[margin <= np.quantile(margin, Q14_FRAC)], float(np.median(margin))


def q14_apply(logits_path, q14_path, median_margin):
    with np.load(logits_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    labels = arrays['labels'].astype(str).tolist()
    L = np.asarray(arrays['logits'], dtype=np.float64).copy()
    pos = {i: k for k, i in enumerate(arrays['ids'].astype(str).tolist())}
    with np.load(q14_path, allow_pickle=False) as s:
        rows = np.array([pos[i] for i in s['ids'].astype(str).tolist()])
        S18 = np.asarray(s['logp'], np.float64)[:, [s['labels'].astype(str).tolist().index(l) for l in labels]]
    assert np.isfinite(S18).all() and len(set(rows.tolist())) == len(rows)
    col, n = q14_labs(labels)
    lp = q14_labmass(S18, col, n)
    a = Q14_A * float(np.clip(median_margin / Q14_REF, 0.5, 2.5))
    before = L.argmax(1)
    sub = L[rows] + a * (lp - lp.mean(1, keepdims=True))[:, col]
    sub -= (np.logaddexp.reduce(sub, axis=1) - np.logaddexp.reduce(L[rows], axis=1))[:, None]
    L[rows] = sub
    print(f'q14 lab step: a {a:.3f} (median lab margin {median_margin:.3f}) on {len(rows)} rows, top label changed on '
          f'{int((L.argmax(1) != before).sum())} of {len(L)} rows', flush=True)
    assert np.isfinite(L).all()
    arrays['logits'] = L.astype(np.asarray(arrays['logits']).dtype)
    return arrays


_q14_cmd = lambda csv, out: [sys.executable, 'q14s1_worker.py', csv, B14, Q14_DIR, out]
if FAST_COMMIT:
    # loads the 14B base and the adapter on the T4s and scores the commit's test rows plus the 3 shortest train rows,
    # whose log-probs must match the TPU's own stage-1 scores (q14op16_s1_on_train.npy): the 14B path's parity check
    _te = pd.read_csv(os.path.join(DATA, 'test.csv'), dtype={'id': str})
    _tr = pd.read_csv(os.path.join(DATA, 'train.csv'), dtype={'id': str})
    _k = np.argsort(_tr.text.fillna('').str.len().to_numpy())[:3]
    pd.concat([_te[['id', 'text']], _tr.iloc[_k][['id', 'text']]]).to_csv(Q14_ROWS, index=False)
    subprocess.run(_q14_cmd(Q14_ROWS, Q14_OUT), check=True)
    with np.load(Q14_OUT) as _s:
        _ids, _labs, _lp = _s['ids'].astype(str), _s['labels'].astype(str).tolist(), np.asarray(_s['logp'], np.float64)
    _order = json.load(open(os.path.join(Q14_DIR, 'meta.json')))['labels']
    _tpu = np.load(first('/kaggle/input/**/q14op16_s1_on_train.npy'))[_k].astype(np.float64)
    _tpu = _lsm14(_tpu[:, [_order.index(l) for l in _labs]], axis=1)
    _t4 = _lp[len(_te):]
    _diff = float(np.abs(_t4 - _tpu).max())
    print('q14 parity on 3 train rows: argmax T4', _t4.argmax(1).tolist(), 'TPU', _tpu.argmax(1).tolist(),
          f'max |log-prob diff| {_diff:.3f}', flush=True)
    assert (_t4.argmax(1) == _tpu.argmax(1)).all() and _diff < 0.6, 'q14 parity failed'
    np.savez('/kaggle/temp/q14-smoke-test.npz', ids=_ids[:len(_te)], labels=np.array(_labs), logp=_lp[:len(_te)])
    np.savez('/kaggle/temp/q14-smoke-logits.npz', ids=_ids[:len(_te)], labels=np.array(_labs),
             logits=np.random.default_rng(0).normal(size=(len(_te), len(_labs))))
    _sel, _med = q14_rows('/kaggle/temp/q14-smoke-logits.npz')
    q14_apply('/kaggle/temp/q14-smoke-logits.npz', '/kaggle/temp/q14-smoke-test.npz', _med)
    print('q14 lab step smoke ok', len(_sel), 'rows selected', flush=True)
else:
    try:
        _left = 8.75 * 3600 - (time.time() - NB_T0)
        assert _left > 2400, f'only {_left:.0f}s left, q14 lab step not run'
        _sel, _med = q14_rows(FINAL_LOGITS)
        _te = pd.read_csv(os.path.join(DATA, 'test.csv'), dtype={'id': str})
        _te[_te.id.isin(set(_sel.tolist()))][['id', 'text']].to_csv(Q14_ROWS, index=False)
        print('q14 lab step: scoring', len(_sel), 'rows', flush=True)
        _r = subprocess.run(_q14_cmd(Q14_ROWS, Q14_OUT), timeout=_left, capture_output=True, text=True)
        print(_r.stdout[-1500:], _r.stderr[-1500:], flush=True)
        assert _r.returncode == 0, _r.returncode
        np.savez(FINAL_LOGITS, **q14_apply(FINAL_LOGITS, Q14_OUT, _med))
    except Exception as e:
        print('q14 lab step skipped:', repr(e)[:300], flush=True)
print('elapsed', round(time.time() - NB_T0), 's', flush=True)
'''

Q14_CHECK = r'''import glob, json, os, subprocess, sys, time
import numpy as np, pandas as pd
from scipy.special import log_softmax
import pinned_assets as A
bad = []
for key in A.PATHS:
    try:
        A.asset_paths(key)
    except AssertionError as e:
        bad.append((key, str(e)[:300]))
for key in A.CHECKPOINTS:
    try:
        A.checkpoint(*key)
    except AssertionError as e:
        bad.append((str(key), str(e)[:300]))
print('PATHS', len(A.PATHS), 'CHECKPOINTS', len(A.CHECKPOINTS), 'unresolved', len(bad), flush=True)
for b in bad:
    print('  BAD', b, flush=True)
q14 = os.path.dirname(A.asset_paths('/kaggle/input/**/q14op16_s1/heads.pt')[0])
print('q14op16_s1', q14, sorted(os.listdir(q14)), flush=True)
print('op8v1_s1', sorted(os.listdir(os.path.dirname(A.asset_paths('/kaggle/input/**/op8v1_s1/heads.pt')[0]))), flush=True)
B14 = A.asset_path('/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1')
DATA = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train = pd.read_csv(DATA + '/train.csv', dtype={'id': str})
rows = np.argsort(train.text.fillna('').str.len().to_numpy())[:3]   # the 3 shortest train passages: CPU fp16 is slow
train.iloc[rows][['id', 'text']].to_csv('/kaggle/working/check3.csv', index=False)
t = time.time()
r = subprocess.run([sys.executable, 'q14s1_worker.py', '/kaggle/working/check3.csv', B14, q14, '/kaggle/working/check3.npz'],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:], 'worker exit', r.returncode, round(time.time() - t), 's', flush=True)
if r.returncode == 0:
    z = np.load('/kaggle/working/check3.npz')
    order = json.load(open(os.path.join(q14, 'meta.json')))['labels']
    labs = z['labels'].astype(str).tolist()
    tpu = log_softmax(np.load(A.asset_paths('/kaggle/input/**/q14op16_s1_on_train.npy')[0])[rows][:, [order.index(l) for l in labs]], axis=1)
    cpu = z['logp']
    print('argmax cpu', cpu.argmax(1).tolist(), 'tpu', tpu.argmax(1).tolist(), 'max |log-prob diff|',
          float(np.abs(cpu - tpu).max()), flush=True)
assert not bad
print('PATHS OK', flush=True)
'''


def meta(name, up, gpu):
    return {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": up["dataset_sources"],
            "competition_sources": up["competition_sources"], "kernel_sources": up["kernel_sources"],
            "model_sources": up["model_sources"], "docker_image": up["docker_image"],
            "machine_shape": up["machine_shape"] if gpu else None, "keywords": []}


def write(name, nb, m):
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({k: v for k, v in m.items() if v is not None}, indent=2) + "\n",
                                              encoding="utf-8")
    sha = hashlib.sha256("".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(nb['cells'])} cells, code sha256 {sha}")


def build():
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    for c in nb["cells"]:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    up = json.loads(META.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    pinned = [c for c in code if c["source"].startswith("%%writefile pinned_assets.py")]
    worker = [c for c in code if c["source"].startswith("%%writefile spec_worker.py")]
    assert len(pinned) == 1 and len(worker) == 1

    fork = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                            f"# ntl-702-spec\n\nareebbb's public [0.702 | Read the Fingerprints]({URL}) (NTL-40, pulled "
                            "2026-11-17): every code cell unchanged. It is lightningv08's 0.702 plus Open Pool sibling "
                            "specialists (sections 15 and 19). Private copy for our submission."}] + nb["cells"])
    write("ntl-702-spec", fork, meta("ntl-702-spec", up, True))

    check = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                             "# ntl-702-spec-paths\n\nCPU check of ntl-702-spec's inputs under this account's mount "
                             "layout: every pinned path and checkpoint, and the specialist worker on the commit test."},
                            pinned[0], worker[0],
                            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CHECK}])
    write("ntl-702-spec-paths", check, meta("ntl-702-spec-paths", up, False))

    # ntl-702-spec-op8: the same cells, op8's pins appended to pinned_assets.py, the worker and the apply cell after
    # NTL-40's specialist cell (so the residual / typography stage reads the scores with both specialists)
    up8 = dict(up, kernel_sources=up["kernel_sources"] + [OP8_KERNEL])
    cells = [dict(c) for c in nb["cells"]]
    pin = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith("%%writefile pinned_assets.py")]
    anchor = [i for i, c in enumerate(cells) if c["cell_type"] == "code"
              and c["source"].startswith("# Apply the sibling specialists to the final scores")]
    assert len(pin) == 1 and len(anchor) == 1
    pin[0]["source"] = pin[0]["source"].rstrip("\n") + "\n" + OP8_PINS
    code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    new = [{"cell_type": "markdown", "metadata": {}, "source":
            "### 19b. Ours: a second sibling specialist (op8, exp057)\n\nOur TPU kernel ntl-tpu-op8 trained a Qwen3-8B-Base "
            "LoRA24 classifier on open pool v13 only (stage 1, `op8v1_s1`). After the GLM stage, if at least 40 min "
            "of the 9 h remain, `op8s1_worker.py` scores the test through infer.py's own `l24_predict`, and its "
            "centred within-lab log-probs are added inside Google, OpenAI and Anthropic at w 0.40 with each lab's mass "
            "kept. Out of fold on our stand-in stack on top of NTL-40: +0.0121 [+0.0041, +0.0204], 5/5 folds >= 0; "
            "+0.032 on the public OOF bank. Any failure leaves NTL-40's scores."},
           code(WORKER), code(APPLY)]
    cells[anchor[0] + 1:anchor[0] + 1] = new
    fork8 = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                             f"# ntl-702-spec-op8\n\nareebbb's public [0.702 | Read the Fingerprints]({URL}) (NTL-40, "
                             "pulled 2026-11-17) with every code cell unchanged, plus our second sibling specialist "
                             "(section 19b, exp057). Private."}] + cells)
    write("ntl-702-spec-op8", fork8, meta("ntl-702-spec-op8", up8, True))

    pipe = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"].startswith("%%writefile pipe.py")]
    infer = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"].startswith("%%writefile infer.py")]
    assert len(pipe) == 1 and len(infer) == 1
    check8 = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                              "# ntl-702-spec-op8-paths\n\nCPU check of ntl-702-spec-op8: every pinned path incl. "
                              "op8v1_s1, and op8s1_worker.py (infer.py's l24_predict) on CPU on 3 train rows against "
                              "the TPU's own s1_on_train.npy."},
                             pin[0], pipe[0], infer[0], code(WORKER), code(OP8_CHECK)])
    write("ntl-702-spec-op8-paths", check8, meta("ntl-702-spec-op8-paths", up8, False))

    # ntl-702-spec-op8-q14lab: ntl-702-spec-op8 plus the q14 lab step right after op8's apply cell
    upq = dict(up8, kernel_sources=up8["kernel_sources"] + [Q14_KERNEL])
    cq = [dict(c) for c in cells]
    pinq = [c for c in cq if c["cell_type"] == "code" and c["source"].startswith("%%writefile pinned_assets.py")]
    after = [i for i, c in enumerate(cq) if c["cell_type"] == "code" and c["source"] == APPLY]
    assert len(pinq) == 1 and len(after) == 1
    pinq[0]["source"] = pinq[0]["source"].rstrip("\n") + "\n" + Q14_PINS
    cq[after[0] + 1:after[0] + 1] = [
        {"cell_type": "markdown", "metadata": {}, "source":
         "### 19c. Ours: the lab decision from a second pool-only model (q14, exp057)\n\nOur TPU kernel ntl-tpu-q14v16 "
         "trained a Qwen3-14B-Base LoRA classifier on open pool v16 only (stage 1, `q14op16_s1`). After op8, if at "
         "least 40 min of the 9 h remain, `q14s1_worker.py` scores the quarter of the test rows whose two largest lab "
         "masses are closest, and each lab's mass moves by 0.05 x its centred lab log-prob (the split inside each lab "
         "kept). Out of fold on our stand-in on top of NTL-40 + op8: +0.0026 [+0.0004, +0.0052], 4/5 folds >= 0. "
         "The commit run checks the 14B path against the TPU's own scores on 3 train rows. Any failure leaves the "
         "scores as they are."},
        code(Q14_WORKER), code(Q14_APPLY)]
    forkq = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                             f"# ntl-702-spec-op8-q14lab\n\nareebbb's public [0.702 | Read the Fingerprints]({URL}) "
                             "(NTL-40, pulled 2026-11-17) with every code cell unchanged, plus our sibling specialist "
                             "(section 19b) and lab step (section 19c), exp057. Private."}] + cq)
    write("ntl-702-spec-op8-q14lab", forkq, meta("ntl-702-spec-op8-q14lab", upq, True))
    checkq = dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                              "# ntl-702-spec-op8-q14lab-paths\n\nCPU check of ntl-702-spec-op8-q14lab: every pinned "
                              "path incl. op8v1_s1 and q14op16_s1, and q14s1_worker.py (infer.py's l24_predict) on CPU "
                              "on 3 train rows against the TPU's own q14op16_s1_on_train.npy."},
                             pinq[0], pipe[0], infer[0], code(Q14_WORKER), code(Q14_CHECK)])
    write("ntl-702-spec-op8-q14lab-paths", checkq, meta("ntl-702-spec-op8-q14lab-paths", upq, False))

    for name, wide in (("ntl-702-gbm", False), ("ntl-702-gbm-c22", True)):
        write(name, gbm_fork(nb, name, wide), meta(name, up, True))

    raw = SOFT.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == SOFT_SHA, "lightningv08's upstream changed"
    soft = json.loads(raw)
    for c in soft["cells"]:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    # NTL-40 = these code cells + charts + the Open Pool specialists (sections 15 and 19) + a final gate
    ntl = {c["source"] for c in nb["cells"] if c["cell_type"] == "code"}
    assert all(c["source"] in ntl for c in soft["cells"] if c["cell_type"] == "code")
    base = (f"lightningv08's public [0.702 | Soft Typography + GLM + Qwen3.8-27B LoRA]({SOFT_URL}) (pulled 2026-11-18; "
            "NTL-40 without areebbb's Open Pool specialists)")
    write("ntl-lv702-gbm", gbm_fork(soft, "ntl-lv702-gbm", False, base),
          meta("ntl-lv702-gbm", json.loads(SOFT_META.read_text(encoding="utf-8")), True))

    raw = APEX.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == APEX_SHA, "lightningv08's apex upstream changed"
    apex = json.loads(raw)
    for c in apex["cells"]:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    # apex-style-15 = the soft typography's code + a reciprocal prompt-pair adjustment and penalty 0.15 (2 cells)
    sc = {c["source"] for c in soft["cells"] if c["cell_type"] == "code"}
    assert sum(c["source"] not in sc for c in apex["cells"] if c["cell_type"] == "code") == 2
    base = (f"lightningv08's public [0.702 | Apex Style 15% + GLM + Qwen3.8-27B LoRA]({APEX_URL}) (pulled 2026-11-18: "
            "the soft typography + a reciprocal prompt-pair adjustment and typography penalty 0.15)")
    write("ntl-lv702a-gbm", gbm_fork(apex, "ntl-lv702a-gbm", False, base),
          meta("ntl-lv702a-gbm", json.loads(APEX_META.read_text(encoding="utf-8")), True))

    raw = FABLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == FABLE_SHA, "lightningv08's 0.704 upstream changed"
    fable = json.loads(raw)
    for c in fable["cells"]:
        c["source"] = text(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    # the 0.704 = the soft typography's code with the stage's row ranking and apex_postprocess changed (3 cells)
    assert sum(c["source"] not in sc for c in fable["cells"] if c["cell_type"] == "code") == 3
    base = (f"lightningv08's public [0.704 | GLM Fable Routing + Qwen3.8-27B LoRA]({FABLE_URL}) (pulled 2026-11-19)")
    for name, wide in (("ntl-lv704-gbm", False), ("ntl-lv704-gbm-c22", True)):
        write(name, gbm_fork(fable, name, wide, base), meta(name, json.loads(FABLE_META.read_text(encoding="utf-8")), True))

    raw = BILL.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == BILL_SHA, "bill123987's upstream changed"
    # ntl706w's stage = the soft typography's stage with the GLM-mass order kept and only these two lines changed
    bill = [text(c) for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
    st = [c for c in soft["cells"] if c["cell_type"] == "code" and c["source"].startswith(GBM_STAGE)][0]["source"]
    for a, b in W125:
        st = st.replace(a, b)
    assert st in bill, "ntl706w's stage is not the soft stage with W125"
    for name, kw in (("ntl-lv704-w125", {}), ("ntl-lv704-w125-c22", {"wide": True}), ("ntl-lv704-w15", {"w": 1.5}),
                     ("ntl-lv704-w125-nof", {"routing": False}), ("ntl-lv704-w125-p05", {"penalty": 0.05}),
                     ("ntl-lv704-w135", {"w": 1.35}), ("ntl-lv704-w11", {"w": 1.1}),
                     ("ntl-lv704-w125-nof-p05", {"routing": False, "penalty": 0.05})):
        write(name, w_fork(fable, name, base, **kw), meta(name, json.loads(FABLE_META.read_text(encoding="utf-8")), True))
    for cut in (224, 256, 320):
        name = f"ntl-lv704-l{cut}"
        write(name, cut_fork(fable, name, base, cut), meta(name, json.loads(FABLE_META.read_text(encoding="utf-8")), True))
    write("ntl-lv704-l256-r05", routing_fork(cut_fork(fable, "ntl-lv704-l256-r05", base, 256), 0.5),
          meta("ntl-lv704-l256-r05", json.loads(FABLE_META.read_text(encoding="utf-8")), True))
    # the 0.704 with only the routing coefficient changed
    r05 = dict(fable, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                              f"# ntl-lv704-r05\n\n{base}, every other code cell unchanged. Private."}] + fable["cells"])
    write("ntl-lv704-r05", routing_fork(r05, 0.5), meta("ntl-lv704-r05", json.loads(FABLE_META.read_text(encoding="utf-8")), True))


GBM_STAGE = "%%writefile /kaggle/temp/v1/v1_stage.py\n"
GBM_LL = "Path('/kaggle/temp/v1/ll_model.json').write_text('"
WIDE = (("    ap.add_argument('--max-cover', type=float, default=0.15, help='at most this fraction of the test is scored')\n",
         "    ap.add_argument('--max-cover', type=float, default=0.22, help='at most this fraction of the test is scored')\n"),
        ("    n_sel = min(int((cum <= 90000).sum()), int(a.max_cover * len(ids)))\n",
         "    n_sel = min(int((cum <= 130000).sum()), int(a.max_cover * len(ids)))\n"))


W125 = (("    ap.add_argument('--max-changed', type=float, default=0.25,\n",
         "    ap.add_argument('--max-changed', type=float, default=0.32,\n"),
        ("    w = model['W'] / tp\n", "    w = 1.25 * model['W'] / tp  # ours: stronger likelihood blend\n"))


ROUTING = ("    fable_mass = np.exp(zl[:, labels.index('claude-fable-5')])\n"
           "    priority = mass + 0.35 * np.minimum(fable_mass, mass)\n"
           "    order = np.argsort(-priority, kind='stable')\n",
           "    order = np.argsort(-mass, kind='stable')  # ours: no Fable routing (the soft 0.702's order)\n")
PENALTY = "strength=0.1, corrected=False, penalty=0.15)"


def w_fork(nb, name, base, w=1.25, wide=False, routing=True, penalty=0.15):
    """base's (the 0.704's) cells with the GLM stage's blend weight at w * W and the changed-predictions guard at 0.32;
    if wide, a 22% stage budget; without routing, the stage rows ranked by GLM mass alone; the typography penalty."""
    cells = [dict(c) for c in nb["cells"]]
    st = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith(GBM_STAGE)]
    assert len(st) == 1
    src = cells[st[0]]["source"]
    for a, b in W125:
        assert src.count(a) == 1, a
        src = src.replace(a, b.replace("ours", "ours (bill123987's ntl706w, exp058)").replace("1.25 * ", f"{w:g} * "))
    for a, b in (WIDE if wide else ()) + (() if routing else (ROUTING,)):
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    cells[st[0]] = dict(cells[st[0]], source=src)
    if penalty != 0.15:
        ca = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and PENALTY in c["source"]]
        assert len(ca) == 1
        cells[ca[0]] = dict(cells[ca[0]], source=cells[ca[0]]["source"].replace(
            PENALTY, f"strength=0.1, corrected=False, penalty={penalty:g})  # ours: the soft 0.702's penalty"))
    what = (f"the GLM stage blends estopadilla's likelihood head at {w:g} W (sanity guard 0.32), as bill123987's public "
            f"[ntl706w]({BILL_URL}) did at 1.25 W on the 0.702 soft typography")
    if wide:
        what += ", and the stage scores up to 22% of the test / 130,000 token rows (lightningv08's apex-style-15)"
    if not routing:
        what += ", and the stage ranks rows by GLM mass alone (no Fable routing, as the soft 0.702)"
    if penalty != 0.15:
        what += f", and the typography penalty is {penalty:g} (the soft 0.702's)"
    return dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                            f"# {name}\n\n{base}, every other code cell unchanged; {what}. Private."}] + cells)


def cut_fork(nb, name, base, cut):
    """base's (the 0.704's) cells with estopadilla's head refitted by his recipe at a `cut`-token likelihood cut (exp060)
    and the stage's token budget at 130,000 rows (so the 15% cover still binds, as in our c22 builds' time)."""
    cells = [dict(c) for c in nb["cells"]]
    model = json.loads((L.HEADS / f"his_L{cut}.json").read_text(encoding="utf-8"))
    his = json.loads((HERE.parent / "glmll" / "v1src" / "ll_model.json").read_text(encoding="utf-8"))
    assert model["max_text_tokens"] == cut and model["features"] == his["features"] and model["W"] == his["W"]
    ll = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith(GBM_LL)]
    st = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith(GBM_STAGE)]
    assert len(ll) == 1 and len(st) == 1
    assert json.loads(ll[0]["source"][len(GBM_LL):].rsplit("')", 1)[0]) == his, "the 0.704's head is not estopadilla's"
    ll[0]["source"] = (f"# ours (exp060): estopadilla's recipe refitted on the first {cut} tokens (his: 160)\n"
                       f"Path('/kaggle/temp/v1/ll_model.json').write_text({json.dumps(model)!r})\n")
    a, b = WIDE[1]
    src = cells[st[0]]["source"]
    assert src.count(a) == 1, a
    cells[st[0]] = dict(cells[st[0]], source=src.replace(a, b.rstrip("\n") + f"  # ours: room for {cut}-token rows\n"))
    what = (f"the GLM stage reads the first {cut} tokens of each passage (estopadilla's head refitted with his recipe at "
            f"that cut; his reads 160) and may score up to 130,000 token rows (15% cover unchanged)")
    return dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                            f"# {name}\n\n{base}, every other code cell unchanged; {what}. Private."}] + cells)


def routing_fork(nb, coef):
    """nb (a fork of the 0.704) with the Fable routing coefficient `coef` (the 0.704's: 0.35, exp061)."""
    a = "    priority = mass + 0.35 * np.minimum(fable_mass, mass)\n"
    cells = [dict(c) for c in nb["cells"]]
    st = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith(GBM_STAGE)]
    assert len(st) == 1 and cells[st[0]]["source"].count(a) == 1
    cells[st[0]] = dict(cells[st[0]], source=cells[st[0]]["source"].replace(
        a, f"    priority = mass + {coef:g} * np.minimum(fable_mass, mass)  # ours: routing coefficient (0.704: 0.35)\n"))
    head = cells[0]["source"]
    assert cells[0]["cell_type"] == "markdown" and head.endswith(". Private.")
    cells[0] = dict(cells[0], source=head[:-len(". Private.")] + f"; the Fable routing coefficient is {coef:g} (0.704: "
                    "0.35). Private.")
    return dict(nb, cells=cells)


def gbm_fork(nb, name, wide, base=None):
    """NTL-40's cells (or base's) with the exp050 gbmpair head in the GLM stage (and, if wide, a 22% stage budget)."""
    cells = [dict(c) for c in nb["cells"]]
    model = json.loads((L.HEADS / "gbmpair.json").read_text(encoding="utf-8"))
    assert model["format"] == "ntl-v1-ll-model-1" and model.get("gbm") and model.get("extra")
    ll = [c for c in cells if c["cell_type"] == "code" and c["source"].startswith(GBM_LL)]
    st = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith(GBM_STAGE)]
    assert len(ll) == 1 and len(st) == 1
    his = json.loads(ll[0]["source"][len(GBM_LL):].rsplit("')", 1)[0])
    assert his == model["fallback"], "NTL-40's head is not gbmpair's fallback"
    ll[0]["source"] = ("# ours (exp050): estopadilla's head averaged with boosted trees (gbmpair); his head is its fallback\n"
                       f"Path('/kaggle/temp/v1/ll_model.json').write_text({json.dumps(model)!r})\n")
    src = patched_stage(cells[st[0]]["source"])
    for a, b in WIDE if wide else ():
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    cells[st[0]] = dict(cells[st[0]], source=src)
    cells[st[0] + 1:st[0] + 1] = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                   "source": "%%writefile /kaggle/temp/v1/extra_features.py\n"
                                             + L.EXTRA_SRC.read_text(encoding="utf-8")}]
    what = ("the GLM stage's head is our exp050 gbmpair (his head averaged 50/50 with boosted trees on per-token "
            "Flash / GLM-5.3 contrasts; his head is the fallback)")
    if wide:
        what += ", and the stage scores up to 22% of the test / 130,000 token rows (lightningv08's apex-style-15)"
    base = base or f"areebbb's public [0.702 | Read the Fingerprints]({URL}) (NTL-40, pulled 2026-11-17)"
    return dict(nb, cells=[{"cell_type": "markdown", "metadata": {}, "source":
                            f"# {name}\n\n{base}, every other code cell unchanged; {what}. Private."}] + cells)


if __name__ == "__main__":
    build()
