"""Save a new version of a kernel WITHOUT running it (KernelExecutionType.QUICK_SAVE), from a folder laid out like
`kaggle kernels push -p <folder>` expects (kernel-metadata.json + the notebook). Mirrors the CLI's push request
(kaggle_api_extended.kernels_push), plus the execution type the CLI never sets. Run through kglrun.py for auth:

    python kglrun.py python quick_save.py <folder> [--dry-run]

--dry-run prints the request fields (not the notebook body) and sends nothing.
"""
import json
import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

folder, dry = sys.argv[1], "--dry-run" in sys.argv
meta = json.load(open(os.path.join(folder, "kernel-metadata.json"), encoding="utf-8"))
assert meta["kernel_type"] == "notebook" and meta["language"] == "python" and len(meta["title"]) >= 5, meta
body = json.load(open(os.path.join(folder, meta["code_file"]), encoding="utf-8"))
for cell in body["cells"]:   # as the CLI does: no outputs, sources as one string
    if cell["cell_type"] == "code":
        cell["outputs"] = []
    if isinstance(cell.get("source"), list):
        cell["source"] = "".join(cell["source"])

r = ApiSaveKernelRequest()
r.slug = meta["id"]
r.new_title = meta["title"]
r.text = json.dumps(body)
r.language = "python"
r.kernel_type = "notebook"
r.is_private = bool(meta.get("is_private", True))
r.enable_gpu = bool(meta.get("enable_gpu", False))
r.enable_tpu = bool(meta.get("enable_tpu", False))
r.enable_internet = bool(meta.get("enable_internet", True))
r.dataset_data_sources = meta.get("dataset_sources", [])
r.competition_data_sources = meta.get("competition_sources", [])
r.kernel_data_sources = meta.get("kernel_sources", [])
r.model_data_sources = meta.get("model_sources", [])
r.category_ids = meta.get("keywords", [])
if meta.get("docker_image"):
    r.docker_image = meta["docker_image"]
if meta.get("machine_shape"):
    r.machine_shape = meta["machine_shape"]
r.kernel_execution_type = KernelExecutionType.QUICK_SAVE
print({k: getattr(r, k) for k in ("slug", "new_title", "is_private", "enable_gpu", "enable_internet", "machine_shape",
                                  "kernel_execution_type")}, f"{len(body['cells'])} cells, {len(r.text)} chars")
if dry:
    sys.exit(0)
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as kaggle:
    resp = kaggle.kernels.kernels_api_client.save_kernel(r)
print({k: getattr(resp, k, None) for k in ("ref", "url", "version_number", "error")})
