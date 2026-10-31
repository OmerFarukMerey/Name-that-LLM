"""Check, right before `kaggle kernels push -p kaggle/lightning/<kernel>`, that the push will attach upstream's inputs.

    /opt/anaconda3/bin/python kaggle/lightning/prepush_check.py ntl-0642-svc-tiers     (or any build_variants.py kernel)

kernel-metadata.json lists its sources without versions, as upstream's does, so a push attaches the current version
of every dataset and notebook output, and the hidden rerun keeps those. V25 (saved 2026-10-25T00:00Z) scored 0.641
with the lightning assets at v14 (sourceId 21155896) and the Kirill heads at their only version; V28 (04:52Z, 0.642)
saw the same versions. BASELINE below
was read from the Kaggle API at 2026-10-25 ~10:00Z: every upstream dataset was last updated before V25 was saved,
and each notebook output source still had the version count shown. If an author publishes a new version before
the push, our fork would silently run different weights or data next to the SVC change, and the comparison with
0.641 would be confounded. This script exits 1 in that case, and also when kernel-metadata.json drifts from the
V25 settings. It only reads from the API and writes nothing. Writing a source as owner/slug/version in
kernel-metadata.json does not pin it: the client accepts that form, but on 2026-10-26 Kaggle attached the current
versions anyway (ntl-v28-qwen-20), so a failure here can't be fixed by pinning.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
V25_SAVED = datetime(2026, 10, 25, 0, 0)
OURS = "omerfarukmerey/ntl-svc627-quality-tiers"
BASELINE_DATASETS = {  # ref: (current version, last updated, UTC)
    "dhruvtibarewal/nlm-l24-adapters": (7, "2026-10-21T16:52:23.293000"),
    "lightningv08/ntl-olmo2-native24-final-inference-assets": (14, "2026-10-24T00:22:48.223000"),
    "seyitkaangunes/ntl-extra-api-passages": (2, "2026-10-16T15:52:16.823000"),
    "abdullah1afify/ntl-generations": (2, "2026-10-20T12:48:47.563000"),
    "dophuchung/ntl-proxy-generations": (1, "2026-10-17T07:22:31.420000"),
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head": (1, "2026-10-16T21:51:36.877000"),
    "josephadamski91/ntl-atlas-qwen3-8b-v1": (3, "2026-10-20T14:22:33.173000"),
    "lucasboesen/ntl-free-proxy-generations": (7, "2026-10-22T23:55:09.390000"),
    "kirilltym/ntl-spectral-style-heads-paired-4404": (1, "2026-10-24T00:59:10.490000"),
    "nishantdahal/ntl-two-stage-lora": (1, "2026-10-19T01:34:11.527000"),
    "nishantdahal/ntl-handwriting-blend": (1, "2026-10-23T02:17:46.923000"),
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset": (6, "2026-10-22T12:41:07.410000"),
    "bbucxi/qwen35-9b-safetensors": (1, "2026-08-16T11:56:41.020000"),
    "pjmathematician/ultrabert-380m-ntl": (1, "2026-10-14T21:10:11.313000"),
}
EXTRA_DATASETS = {  # datasets a fork may add on top of upstream's; they postdate V25, so only their version is pinned
    OURS: (1, "2026-10-25T09:13:57.590000"),
    "kirilltym/ntl-rare-class-svc-1516": (1, "2026-10-25T13:24:09.303000"),
}
EXTRA_KERNELS = {  # our own notebook outputs a fork may add (ntl-v28-mistral-*): ref -> version
    "omerfarukmerey/ntl-member-mistral7b-full": 1,
}
EXTRA_MODELS = ["mistral-ai/mistral/pyTorch/7b-v0.1-hf/1"]  # ntl-v28-mistral-*
BASELINE_KERNELS = {  # ref: current version number
    "lucasboesen/name-that-llm-glm4-9b-trunc25": 1,
    "lucasboesen/name-that-llm-extract-encoders2": 1,
    "lucasboesen/name-that-llm-electra-large-weights": 1,
    "lucasboesen/name-that-llm-desklib-weights": 1,
    "lucasboesen/name-that-llm-train-lora24-s7": 1,
    "lucasboesen/name-that-llm-train-tokattn": 1,
    "lucasboesen/name-that-llm-train-tokattn2": 1,
    "vnton37nguyn/ntl-train-lora-qwen25-instyle": 2,
}
V25_SETTINGS = {
    "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
    "machine_shape": "NvidiaTeslaT4", "competition_sources": ["name-that-llm"],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "model_sources": ["qwen-lm/qwen-3/Transformers/8b-base/1", "qwen-lm/qwen-3/Transformers/14b-base/1",
                      "qwen-lm/qwen-3/Transformers/4b-base/1", "qwen-lm/qwen2.5/Transformers/14b/1",
                      "qwen-lm/qwen2.5/Transformers/7b/1"],
}


def api_client():
    """The standard Kaggle login, proven by seeing our private dataset. kagglesdk only refreshes an expired
    OAuth access token 30 min after it expires (until then calls 401 or act anonymously), so otherwise this
    retries once with a fresh token held in this process's memory only."""
    import os
    from kaggle.api.kaggle_api_extended import KaggleApi
    from requests.exceptions import HTTPError

    api = KaggleApi()
    api.authenticate()
    try:
        if any(d.ref == OURS for d in api.dataset_list(user="omerfarukmerey", search="ntl-svc627-quality-tiers")):
            return api
    except HTTPError as error:
        if error.response is None or error.response.status_code != 401:
            raise
    from kagglesdk import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials
    print("Kaggle access token expired; using a fresh in-memory token (credentials.json is not changed)")
    with KaggleClient() as client:
        os.environ["KAGGLE_API_TOKEN"] = KaggleCredentials.load(client=client).generate_access_token().token
    api = KaggleApi()
    api.authenticate()
    return api


def main(kernel):
    problems = []
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    for key, wanted in V25_SETTINGS.items():
        if key == "model_sources":
            wanted = wanted + [m for m in EXTRA_MODELS if m in meta.get(key, [])]
        if meta.get(key) != wanted:
            problems.append(f"kernel-metadata.json {key} = {meta.get(key)!r}, expected {wanted!r}")
    if meta.get("id") != f"omerfarukmerey/{kernel}" or "id_no" in meta:
        problems.append(f"kernel-metadata.json id {meta.get('id')!r} / id_no {meta.get('id_no')!r}")
    attached = set(meta["dataset_sources"])
    if not set(BASELINE_DATASETS) <= attached or not attached - set(BASELINE_DATASETS) <= set(EXTRA_DATASETS) \
            or len(attached) != len(meta["dataset_sources"]):
        problems.append(f"dataset_sources differ from BASELINE + EXTRA: {sorted(attached ^ set(BASELINE_DATASETS))}")
    pinned = {**BASELINE_DATASETS, **{ref: EXTRA_DATASETS[ref] for ref in attached & set(EXTRA_DATASETS)}}
    kernels = {**BASELINE_KERNELS, **{ref: EXTRA_KERNELS[ref] for ref in meta["kernel_sources"] if ref in EXTRA_KERNELS}}
    if sorted(meta["kernel_sources"]) != sorted(kernels):
        problems.append(f"kernel_sources differ from BASELINE + EXTRA: {sorted(set(meta['kernel_sources']) ^ set(kernels))}")

    api = api_client()
    for ref, (version, updated) in pinned.items():
        owner, slug = ref.split("/")
        found = [d for d in api.dataset_list(user=owner, search=slug) if d.ref == ref]
        if len(found) != 1:
            problems.append(f"{ref}: not found through the API ({len(found)} matches)")
            continue
        now = (found[0].current_version_number, found[0].last_updated.isoformat())
        status = "ok" if now == (version, updated) else "CHANGED"
        if now != (version, updated):
            problems.append(f"{ref}: now v{now[0]} updated {now[1]}, baseline v{version} updated {updated}")
        if ref not in EXTRA_DATASETS and found[0].last_updated >= V25_SAVED:
            problems.append(f"{ref}: updated {now[1]}, after V25 was saved")
        print(f"dataset {ref:58s} v{now[0]:<3d} {now[1]}  {status}")

    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    with api.build_kaggle_client() as kaggle:
        for ref, version in kernels.items():
            request = ApiGetKernelRequest()
            request.user_name, request.kernel_slug = ref.split("/")
            now = kaggle.kernels.kernels_api_client.get_kernel(request).metadata.current_version_number
            if now != version:
                problems.append(f"{ref}: now version {now}, baseline {version}")
            print(f"kernel  {ref:58s} v{now:<3d} {'ok' if now == version else 'CHANGED'}")

    if problems:
        print("PREPUSH FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print(f"PREPUSH OK: {len(pinned)} datasets and {len(kernels)} notebook outputs unchanged "
          f"since the baseline; kernel-metadata.json keeps the V25 settings")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepush_check.py <kernel folder in kaggle/lightning>")
    main(sys.argv[1])
