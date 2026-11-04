"""Print owner, visibility and licence of every input of a kernel (read-only). Run via kglrun.py.

    python kglrun.py python visibility_of.py user/kernel-slug
"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from kagglesdk.models.types.model_api_service import ApiGetModelRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    r = ApiGetKernelRequest(); r.user_name, r.kernel_slug = sys.argv[1].split('/')
    m = kaggle.kernels.kernels_api_client.get_kernel(r).metadata
    print(f"kernel  {sys.argv[1]:58s} {'PRIVATE' if m.is_private else 'public'}")
    for ref in m.dataset_data_sources:
        r = ApiGetDatasetRequest(); r.owner_slug, r.dataset_slug = ref.split('/')
        try:
            d = kaggle.datasets.dataset_api_client.get_dataset(r)
            print(f"dataset {ref:58s} {'PRIVATE' if d.is_private else 'public'}  v{d.current_version_number:<3} {d.license_name}")
        except Exception as e:
            print(f"dataset {ref:58s} ERROR {e.__class__.__name__}")
    for ref in m.kernel_data_sources:
        r = ApiGetKernelRequest(); r.user_name, r.kernel_slug = ref.split('/')
        try:
            k = kaggle.kernels.kernels_api_client.get_kernel(r).metadata
            print(f"kernel  {ref:58s} {'PRIVATE' if k.is_private else 'public'}")
        except Exception as e:
            print(f"kernel  {ref:58s} ERROR {e.__class__.__name__}")
    for ref in m.model_data_sources:
        r = ApiGetModelRequest(); r.owner_slug, r.model_slug = ref.split('/')[:2]
        try:
            md = kaggle.models.model_api_client.get_model(r)
            print(f"model   {ref:58s} {'PRIVATE' if md.is_private else 'public'}")
        except Exception as e:
            print(f"model   {ref:58s} ERROR {e.__class__.__name__}")
