"""List the files (name, bytes) of given dataset versions. Run through kglrun.py for auth.

    python kglrun.py python ds_files.py owner/slug/N [owner/slug/N ...]
"""
import sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest

api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    for ref in sys.argv[1:]:
        owner, slug, ver = ref.split('/')
        files, token = [], None
        while True:
            r = ApiListDatasetFilesRequest(); r.owner_slug, r.dataset_slug, r.dataset_version_number = owner, slug, int(ver)
            r.page_size = 200
            if token: r.page_token = token
            resp = kaggle.datasets.dataset_api_client.list_dataset_files(r)
            files += list(resp.dataset_files or [])
            token = resp.next_page_token
            if not token: break
        print(f"## {ref}: {len(files)} files, {sum(f.total_bytes or 0 for f in files)/1e6:.1f} MB")
        for f in sorted(files, key=lambda f: f.name)[:40]:
            print(f"   {f.total_bytes:>12}  {f.name}")
