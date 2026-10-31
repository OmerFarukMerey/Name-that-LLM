"""Download one file of one dataset version (no metadata/licence call). Run through kglrun.py for auth.

    python kglrun.py python ds_download.py owner/slug/N file_name out_dir
"""
import os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

ref, name, out = sys.argv[1:4]
owner, slug, ver = ref.split('/')
api = KaggleApi(); api.authenticate()
with api.build_kaggle_client() as kaggle:
    r = ApiDownloadDatasetRequest(); r.owner_slug, r.dataset_slug, r.dataset_version_number, r.file_name = owner, slug, int(ver), name
    resp = kaggle.datasets.dataset_api_client.download_dataset(r)
    outfile = os.path.join(out, resp.request.url.split('?')[0].split('/')[-1])
    api.download_file(resp, outfile, kaggle.http_client(), True, False)
print(ref, name, '->', outfile, os.path.getsize(outfile))
