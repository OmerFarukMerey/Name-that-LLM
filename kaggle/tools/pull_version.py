"""Pull one exact version of a public Kaggle notebook (source + metadata). Run through kglrun.py for auth.

    python kglrun.py python pull_version.py <user/slug> <V-label e.g. V22 | none for latest> <out_dir>

Writes <out_dir>/<slug>_<label>.ipynb, <slug>_<label>.meta.json and <slug>_<label>.txt (cell dump: '=====CELL i type').
"""
import json, sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ref, label, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
user, slug = ref.split('/')
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    r = ApiGetKernelRequest(); r.user_name = user; r.kernel_slug = slug
    if label != 'none':
        r.version_label = label
    resp = kaggle.kernels.kernels_api_client.get_kernel(r)
md = resp.metadata.to_dict()
src = resp.blob.source or ''
stem = f"{slug}_{label}"
(out / f"{stem}.ipynb").write_text(src, encoding='utf-8')
(out / f"{stem}.meta.json").write_text(json.dumps(md, indent=1, default=str), encoding='utf-8')
try:
    nb = json.loads(src)
    with open(out / f"{stem}.txt", 'w', encoding='utf-8') as f:
        for i, c in enumerate(nb['cells']):
            s = c['source'] if isinstance(c['source'], str) else ''.join(c['source'])
            f.write(f"\n=====CELL {i} {c['cell_type']}\n{s}")
    ncell = len(nb['cells'])
except Exception as e:
    ncell = f'not ipynb ({e.__class__.__name__})'
keys = ['currentVersionNumber', 'dockerImage', 'machineShape', 'enableGpu', 'lastRunTime']
print(ref, label, {k: md.get(k) for k in keys}, 'cells=', ncell, 'datasets=', len(md.get('datasetDataSources') or []),
      'kernels=', len(md.get('kernelDataSources') or []), 'models=', len(md.get('modelDataSources') or []))
