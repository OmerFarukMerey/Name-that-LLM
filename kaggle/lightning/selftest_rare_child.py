"""ntl-v28-rare-helper15 self-test helper: the notebook's own cached_svc on kirilltym's rare1516 readout and on V28's
ntl627-svc-prefit, on the same 300 train rows, in a separate process.

The self-test cell runs this file with one JSON argument and reads its RARE_SELFTEST_REPORT line. Running it in a child
process keeps sklearn, both readouts and the sparse features out of the notebook process, which then idles for the
hours-long hidden rerun. rare1516 ships no preview.npz, so the check is relative to ntl627 on the same rows.
"""
import json
import sys
from pathlib import Path

request = json.loads(sys.argv[1])
sys.path.insert(0, request['runtime'])  # the ensemble imports reference627_common from its own directory too
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import reference627_common as common  # noqa: E402

train = pd.read_csv(request['train_csv'], dtype={'id': str})
train = train.iloc[np.linspace(0, len(train) - 1, 300).round().astype(int)]
texts = train.text.fillna('').astype(str).tolist()
truth = train.model.to_numpy()
labels = list(common.LABELS)
report = {'module': str(Path(common.__file__).resolve()), 'labels': labels, 'train_rows': int(len(train)),
          'train_labels_known': bool(set(truth) <= set(labels))}
argmax = {}
for name in ('rare', 'ntl627'):
    logits = common.cached_svc(request[name], texts)  # a str root, exactly as run_fulltext_selected_ensemble.py passes it
    argmax[name] = logits.argmax(1)
    predicted = np.asarray(labels)[argmax[name]]
    report[name] = {
        'labels': json.loads((Path(request[name]) / 'metadata.json').read_text(encoding='utf-8'))['labels'],
        'shape': list(logits.shape),
        'finite': bool(np.isfinite(logits).all()),
        'accuracy': float((predicted == truth).mean()),
        'rare_class_hits': {label: [int(((predicted == label) & (truth == label)).sum()), int((truth == label).sum())]
                            for label in ('glm-5.3', 'glm-5.3-flash', 'kimi-k3')},
    }
report['agreement'] = float((argmax['rare'] == argmax['ntl627']).mean())
report['torch_imported'] = 'torch' in sys.modules
print('RARE_SELFTEST_REPORT ' + json.dumps(report), flush=True)
