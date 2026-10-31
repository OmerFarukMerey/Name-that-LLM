"""Print the account's weekly accelerator quota (used / total hours) and its refresh time. Run through kglrun.py for auth."""
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    r = kaggle.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
print('refresh', r.quota_refresh_time)
for name in ('gpu_quota', 'tpu_quota'):
    q = getattr(r, name)
    print(name, q.to_dict() if q is not None else None)
