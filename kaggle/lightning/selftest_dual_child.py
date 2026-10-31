"""ntl-v28-dual self-test child: the patched worker and run-script block, end to end on CPU with a mocked 27B classifier.

The self-test cell (e) runs this file with one JSON argument ({runtime, train_csv or null}) and reads its
DUAL_SELFTEST_REPORT line. A child process keeps torch, sklearn and the 40960-d fixture out of the notebook process.
Fixture: 10 near-duplicates of one passage (one word dropped, one token added each) planted among 50 other passages
(real train passages: 12 answers of one prompt plus one answer each of 38 other prompts; synthetic without train.csv).
Every expectation is recomputed here independently of dual_head.py (Roman Tamrazov's V22 formula, written out).
"""
import json
import sys
import tempfile
import textwrap
import types
from pathlib import Path

request = json.loads(sys.argv[1])
sys.path.insert(0, request['runtime'])
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.special import softmax  # noqa: E402
import dual_head  # noqa: E402
from reference627_common import LABELS, topic_groups  # noqa: E402

report = {'modules': {}}
rng = np.random.default_rng(20261025)
report['smoke'] = dual_head.smoke_test()

# ---- fixture: one planted 10-row near-duplicate cluster among 50 other passages
if request.get('train_csv'):
    train = pd.read_csv(request['train_csv'], dtype={'id': str}).sort_values('id', kind='stable')
    prompts = sorted(train.prompt_id.unique())
    base = train[train.prompt_id == prompts[0]].text.iloc[0].split()
    planted_texts = [' '.join(base[:k] + base[k + 1:] + [f'qzv{k}k']) for k in range(10)]
    others = (train[train.prompt_id == prompts[1]].text.iloc[:12].tolist()
              + [train[train.prompt_id == p].text.iloc[0] for p in prompts[2:40]])
    report['fixture'] = 'train.csv'
else:
    def passage(words):
        return ' '.join(''.join(rng.choice(list('bcdfghjklmnpqrstvwz'), 2)) + w for w in words)
    base = [f'atq{j}' for j in range(45)]
    planted_texts = [' '.join(base[:k] + base[k + 1:] + [f'qzv{k}k']) for k in range(10)]
    others = [passage([f'o{i}x{j}' for j in range(35)]) for i in range(50)]
    report['fixture'] = 'synthetic'
order = rng.permutation(60)
texts = [(planted_texts + others)[i] for i in order]
planted = sorted(int(np.flatnonzero(order == i)[0]) for i in range(10))
ids = [f'row{i:03d}' for i in range(60)]
reference_groups = topic_groups(texts, threshold=.90)
sizes = np.bincount(reference_groups)[reference_groups]
reference_valid = (sizes >= 8) & (sizes <= 18)
report['planted_group_exact'] = (np.flatnonzero(reference_groups == reference_groups[planted[0]]).tolist() == planted
                                 and bool(reference_valid[planted].all()))
report['fixture_valid_rows'] = int(reference_valid.sum())
report['fixture_other_valid_rows'] = int(reference_valid.sum()) - 10


# ---- the patched own_qwen_worker.py main(), with the 27B classifier replaced by CPU torch heads
class MockClassifier:
    built = []

    def __init__(self, base, adapter, *, portable=False, resident=False):
        assert (base, adapter, portable, resident) == ('mock-base', 'mock-adapter', True, True)
        torch.manual_seed(0)
        self.device = torch.device('cpu')
        self.labels = [LABELS[i] for i in np.random.default_rng(1).permutation(18)]  # worker label order != LABELS
        self.plain, self.centered = torch.nn.Linear(40960, 18), torch.nn.Linear(40960, 18)
        with torch.no_grad():
            self.plain.weight.mul_(40.)
            self.centered.weight.copy_(.5 * self.plain.weight + 20. * torch.randn(18, 40960) / 40960 ** .5)
        self.calls = []
        MockClassifier.built.append(self)

    def predict(self, texts, batch_size=3, return_features=False):
        self.calls.append((len(texts), batch_size, return_features))
        if not return_features:
            raise AssertionError('the dual worker must request the pooled features')
        offsets = np.random.default_rng(2).normal(size=(reference_groups.max() + 1, 40960)) * 2
        features = (np.random.default_rng(3).normal(size=(len(texts), 40960)) + offsets[reference_groups]).astype(np.float32)
        with torch.inference_mode():
            logits = self.plain(torch.from_numpy(features)).numpy()
        self.features = features.copy()
        return logits.astype(np.float32), features


stub = types.ModuleType('qwen_fulltext_inference')
stub.FullTextClassifier = MockClassifier
sys.modules['qwen_fulltext_inference'] = stub
torch.cuda.device_count = lambda: 2
torch.cuda.max_memory_reserved = lambda device=None: 0
import own_qwen_worker  # noqa: E402
import run_fulltext_selected_ensemble as run  # noqa: E402

for module in (dual_head, own_qwen_worker, run):
    report['modules'][module.__name__] = str(Path(module.__file__).resolve())

with tempfile.TemporaryDirectory(prefix='ntl-dual-selftest-') as tmp:
    work = Path(tmp)
    pd.DataFrame({'id': ids, 'text': texts}).to_csv(work / 'qwen-query.csv', index=False)
    (work / 'paths.json').write_text(json.dumps({'own_nf4_base': 'mock-base', 'own_fulltext_adapter': 'mock-adapter'}))
    sys.argv = ['own_qwen_worker.py', '--query', str(work / 'qwen-query.csv'), '--paths', str(work / 'paths.json'),
                '--out', str(work / 'qwen.npz')]
    own_qwen_worker.main()
    model = MockClassifier.built[-1]
    report['predict_calls'] = model.calls
    with np.load(work / 'qwen.npz', allow_pickle=False) as z:
        saved = {key: z[key] for key in z.files}
    sidecar = json.loads((work / 'qwen.npz.json').read_text())
    report['npz_keys'] = sorted(saved)
    report['npz_dtypes'] = {key: str(value.dtype) for key, value in saved.items()}
    report['sidecar_dual'] = sidecar.get('dual_head')
    histogram = np.bincount(np.bincount(reference_groups)[np.bincount(reference_groups) > 0])
    report['sidecar_matches'] = (sidecar['dual_head']['centered_valid_rows'] == int(reference_valid.sum())
                                 and sidecar['dual_head']['cluster_size_histogram']
                                 == {str(s): int(n) for s, n in enumerate(histogram) if n}
                                 and sidecar['rows'] == 60)
    report['worker_valid_matches'] = (np.array_equal(saved['centered_valid'], reference_valid.astype(np.uint8))
                                      and np.array_equal(saved['centered_groups'], reference_groups)
                                      and saved['ids'].tolist() == ids and saved['labels'].tolist() == model.labels)
    # independent centred logits: float64 cluster means of the features the mock returned, the head as float64
    weight = model.centered.weight.detach().double().numpy()
    bias = model.centered.bias.detach().double().numpy()
    expected = saved['logits'].astype(np.float64).copy()
    x = model.features.astype(np.float64)
    for group in np.unique(reference_groups[reference_valid]):
        rows = np.flatnonzero(reference_groups == group)
        expected[rows] = (x[rows] - x[rows].mean(0)) @ weight.T + bias
    report['centered_max_abs_diff'] = float(np.abs(saved['centered_logits'] - expected).max())
    report['outside_rows_identical'] = bool(np.array_equal(saved['centered_logits'][~reference_valid],
                                                           saved['logits'][~reference_valid]))

    # ---- the patched run-script block, exactly as written in run_fulltext_selected_ensemble.py
    script = Path(run.__file__).read_text(encoding='utf-8')
    start = script.index("    qwen_logits=aligned(work/'qwen.npz',selected_ids)\n")
    stop = script.index("    dual_report['qwen_dual_changed_rows']=")
    stop = script.index('\n', stop) + 1
    block = textwrap.dedent(script[start:stop])
    prior = rng.dirichlet(np.ones(18) * .3, size=60)
    scope = dict(np=np, json=json, aligned=run.aligned, blend_selected=run.blend_selected, sinkhorn=run.sinkhorn,
                 dual_head_logits=run.dual_head_logits, LABELS=run.LABELS, work=work, ids=ids, selected_ids=list(ids),
                 prior=prior, share=.10)
    exec(compile(block, 'run_fulltext_selected_ensemble.py[dual block]', 'exec'), scope)
    report['block_lines'] = block.count('\n')
    with np.load(work / 'qwen_blend_probabilities.npz', allow_pickle=False) as z:
        report['blend_npz_matches'] = bool(np.array_equal(z['probabilities'], scope['mixed']))

columns = [model.labels.index(label) for label in LABELS]
plain = softmax(saved['logits'][:, columns].astype(np.float64), axis=1)
centered = softmax(saved['centered_logits'][:, columns].astype(np.float64), axis=1)
agreement = plain.argmax(1) == centered.argmax(1)
alpha = np.where(reference_valid, np.where(agreement, .35, .18), 0.)[:, None]
dual = (1 - alpha) * plain + alpha * centered
mixed = scope['mixed']
report['block_max_abs_diff'] = float(np.abs(mixed - (.9 * prior + .1 * dual)).max())
report['block_outside_bitwise_v28'] = bool(np.array_equal(mixed[~reference_valid], scope['v28_mix'][~reference_valid]))
report['block_inside_changed'] = bool(np.abs(mixed[reference_valid] - scope['v28_mix'][reference_valid]).max() > 1e-4)
report['dual_report'] = scope['dual_report']
report['dual_report_matches'] = (scope['dual_report']['qwen_dual_valid_rows'] == int(reference_valid.sum())
                                 and scope['dual_report']['qwen_dual_agree_rows'] == int((agreement & reference_valid).sum()))

# ---- the dual formula on random data, both alpha branches, and the all-invalid identity
logits_a = rng.normal(size=(400, 18)) * 3
logits_b = np.where(np.arange(400)[:, None] < 200, logits_a + rng.normal(size=(400, 18)) * .3, rng.normal(size=(400, 18)) * 3)
p, q = softmax(logits_a, axis=1), softmax(logits_b, axis=1)
agree = p.argmax(1) == q.argmax(1)
roman = (1 - np.where(agree, .35, .18)[:, None]) * p + np.where(agree, .35, .18)[:, None] * q
ours, _ = dual_head.dual_probabilities(logits_a, logits_b, np.ones(400, dtype=bool))
out_all, _ = dual_head.dual_head_logits(logits_a, logits_b, np.ones(400, dtype=np.uint8))
out_none, rep_none = dual_head.dual_head_logits(logits_a, logits_a, np.zeros(400, dtype=np.uint8))
report['formula_max_abs_diff'] = float(max(np.abs(ours - roman).max(), np.abs(softmax(out_all, axis=1) - roman).max()))
report['formula_branches'] = [int(agree.sum()), int((~agree).sum())]
report['formula_invalid_identity'] = bool(np.array_equal(out_none, logits_a) and rep_none['qwen_dual_valid_rows'] == 0)

# ---- fail closed: a broken contract raises instead of degrading
failures = {}
tampered = saved['centered_logits'].copy()
tampered[np.flatnonzero(~reference_valid)[0], 0] += 1
shifted = np.roll(model.features, 1, axis=0)
with torch.inference_mode():
    def head(layer):
        return lambda block: layer(torch.from_numpy(block)).numpy()
    for name, call in [('centred_differs_outside', lambda: dual_head.dual_head_logits(saved['logits'], tampered, saved['centered_valid'])),
                       ('features_misaligned', lambda: dual_head.centered_head_logits(saved['logits'], shifted, reference_groups, reference_valid, head(model.centered), head(model.plain))),
                       ('partial_cluster', lambda: dual_head.centered_head_logits(saved['logits'], model.features, reference_groups, reference_valid & (np.arange(60) != planted[0]), head(model.centered), head(model.plain))),
                       ('valid_not_binary', lambda: dual_head.dual_head_logits(saved['logits'], saved['centered_logits'], saved['centered_valid'] * 2))]:
        try:
            call()
            failures[name] = 'no error'
        except (ValueError, FloatingPointError) as error:
            failures[name] = type(error).__name__
report['fail_closed'] = failures
print('DUAL_SELFTEST_REPORT ' + json.dumps(report), flush=True)
