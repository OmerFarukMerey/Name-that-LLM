"""Print current version + last update of every nlm-v27 input, and nlm-v27's own version/run time (read-only)."""
import json, sys
sys.path.insert(0, '/Users/ofm/Desktop/Kaggle/Name-that-LLM/kaggle/lightning')
from prepush_check import api_client
meta = json.load(open('/Users/ofm/Desktop/Kaggle/Name-that-LLM/kaggle/nlm/upstream_nlm_v27.meta.json'))
api = api_client()
for ref in meta['dataset_sources']:
    owner, slug = ref.split('/')
    found = [d for d in api.dataset_list(user=owner, search=slug) if d.ref == ref]
    if not found: print(f"dataset {ref:55s} NOT FOUND"); continue
    d = found[0]; print(f"dataset {ref:55s} v{d.current_version_number:<3d} {d.last_updated.isoformat()}")
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelsRequest
with api.build_kaggle_client() as kaggle:
    for ref in meta['kernel_sources'] + ['dhruvtibarewal/nlm-v27']:
        r = ApiGetKernelRequest(); r.user_name, r.kernel_slug = ref.split('/')
        m = kaggle.kernels.kernels_api_client.get_kernel(r).metadata
        print(f"kernel  {ref:55s} v{m.current_version_number:<3d} run {getattr(m, 'last_run_time', None)}")
