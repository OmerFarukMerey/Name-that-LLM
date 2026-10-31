"""Print current version + last update of every dataset/kernel input of a kernel-metadata.json (read-only). Run via kglrun.py.

    python kglrun.py python inputs_of.py path/to/kernel-metadata.json
"""
import json, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

meta = json.load(open(sys.argv[1]))
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    for ref in meta.get('dataset_sources', []):
        r = ApiGetDatasetRequest(); r.owner_slug, r.dataset_slug = ref.split('/')
        try:
            d = kaggle.datasets.dataset_api_client.get_dataset(r)
            print(f"dataset {ref:58s} v{d.current_version_number:<3} {d.last_updated}")
        except Exception as e:
            print(f"dataset {ref:58s} ERROR {e.__class__.__name__}")
    for ref in meta.get('kernel_sources', []):
        r = ApiGetKernelRequest(); r.user_name, r.kernel_slug = ref.split('/')
        try:
            m = kaggle.kernels.kernels_api_client.get_kernel(r).metadata
            print(f"kernel  {ref:58s} v{m.current_version_number:<3} run {getattr(m, 'last_run_time', None)}")
        except Exception as e:
            print(f"kernel  {ref:58s} ERROR {e.__class__.__name__}")
    for ref in meta.get('model_sources', []):
        print(f"model   {ref}")
