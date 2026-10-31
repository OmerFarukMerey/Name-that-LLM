"""ntl-0641-svc-tiers self-test helper: one call of the notebook's own cached_svc, in a separate process.

The self-test cell runs this file with one JSON argument and reads its SVC_SELFTEST_REPORT line. Running it in a child
process keeps sklearn, the 78 MB readout and the sparse features out of the notebook process, which then idles
for the hours-long hidden rerun. The child inherits the notebook's environment, as the ensemble does.
"""
import json
import sys
from hashlib import blake2s
from pathlib import Path

request = json.loads(sys.argv[1])
sys.path.insert(0, request['runtime'])  # the ensemble imports reference627_common from its own directory too
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import reference627_common as common  # noqa: E402

fixture = request['fixture']
test = pd.read_csv(request['test_csv'], dtype={'id': str})
test['text'] = test.text.fillna('').astype(str)
public = test[[i in fixture and blake2s(t.encode('utf-8'), digest_size=16).hexdigest() == fixture[i]
               for i, t in zip(test.id, test.text)]]
with np.load(Path(request['root']) / 'preview.npz', allow_pickle=False) as saved:
    preview_ids, direct = saved['ids'].astype(str).tolist(), saved['direct']
train = pd.read_csv(request['train_csv'], dtype={'id': str})
train = train.iloc[np.linspace(0, len(train) - 1, 300).round().astype(int)]
texts = public.text.tolist() + train.text.fillna('').astype(str).tolist()

logits = common.cached_svc(request['root'], texts)  # a str root, exactly as run_fulltext_selected_ensemble.py passes it
labels = list(common.LABELS)
gap = None
if len(public) and direct.shape == (3, 18) and set(public.id) <= set(preview_ids):
    gap = float(np.abs(logits[:len(public)] - direct[[preview_ids.index(i) for i in public.id]]).max())
predicted = np.asarray(labels)[logits[len(public):].argmax(1)]
report = {
    'module': str(Path(common.__file__).resolve()),
    'torch_imported': 'torch' in sys.modules,
    'logits_shape': list(logits.shape),
    'finite': bool(np.isfinite(logits).all()),
    'labels': labels,
    'public_ids': public.id.tolist(),
    'preview_ids': preview_ids,
    'direct_shape': list(direct.shape),
    'gap': gap,
    'train_rows': int(len(train)),
    'train_labels_known': bool(set(train.model) <= set(labels)),
    'accuracy': float((predicted == train.model.to_numpy()).mean()),
}
print('SVC_SELFTEST_REPORT ' + json.dumps(report), flush=True)
