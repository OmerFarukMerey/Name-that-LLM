"""Print the version history (number, creation time, notes) of Kaggle datasets. Run through kglrun.py for auth.

    python kglrun.py python ds_versions.py owner/slug [owner/slug ...]
"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    for ref in sys.argv[1:]:
        r = ApiGetDatasetRequest(); r.owner_slug, r.dataset_slug = ref.split('/')
        d = kaggle.datasets.dataset_api_client.get_dataset(r)
        print(f"## {ref} current v{d.current_version_number} updated {d.last_updated}")
        for v in sorted(d.versions or [], key=lambda v: v.version_number)[-6:]:
            print(f"   v{v.version_number:<3} {v.creation_date}  {(v.version_notes or '')[:90]!r}")
