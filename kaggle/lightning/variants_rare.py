"""ntl-v28-rare-helper15 (candidate C6): lightningv08 V28 with kirilltym's rare-class SVC readout and the helper at 15%.

Two changes relative to V28, one relative to kirilltym's own V28 fork:
  1. the cached main n-gram SVC (paths['svc_prefit'], ntl627-svc-prefit) is swapped for kirilltym's rare1516 readout,
     kirilltym/ntl-rare-class-svc-1516 v1, folder rare1516_svc (CC BY 4.0): the same four views, label order and
     loader (cached_svc), refit on 79,998 rows = ntl627's 78,482 + 1,516 GLM-5.3 / GLM-5.3-flash / Kimi-K3 passages
     (Daniel Andres Miranda's CC BY 4.0 data, following JN Liang's .639 -> .640 experiment). This is exactly the
     change of kirilltym/0-639-teach-more-keep-the-signature V2/V3 (cell 4 svc_prefit). It still enters at weight 1;
  2. the Kirill spectral helper share 0.10 -> 0.15 (variant_recipes.share_variant), as in ntl-v28-helper15.
The resolver accepts exactly one of the two mount layouts and asserts it, like build.SVC_ROOT. The self-test pins
every rare1516 file by sha256 (selftest_rare_check.py) and runs the notebook's own cached_svc on the rare1516 and
ntl627 readouts in a child process (selftest_rare_child.py) on the same 300 train rows.
"""
import json

from variant_recipes import share_variant

RARE = "kirilltym/ntl-rare-class-svc-1516"
STATUS15 = "FROZEN_ALLROWS_QWEN10_KIRILL15_CANDIDATE"  # share_variant's status, made variant-specific below
STATUS = "FROZEN_ALLROWS_QWEN10_KIRILL15_RARE1516_CANDIDATE"
CHILD = "/kaggle/temp/ntl_rare_selftest_child.py"  # outside RUNTIME, so RUNTIME stays upstream's
RARE_ROOT = """
# ---- ntl-v28-rare-helper15 (kaggle/lightning/variants_rare.py): the main n-gram SVC reads kirilltym's rare-class
# readout (ntl627 pool + 1,516 GLM/Kimi passages; same four views, labels and cached_svc), as in his V28 fork V2/V3.
_rare_svc_roots = [root for root in ['/kaggle/input/datasets/kirilltym/ntl-rare-class-svc-1516/rare1516_svc',
                                     '/kaggle/input/ntl-rare-class-svc-1516/rare1516_svc'] if (Path(root) / 'readout.npz').is_file()]
assert len(_rare_svc_roots) == 1, ('Attach kirilltym/ntl-rare-class-svc-1516 exactly once', _rare_svc_roots)
RARE_SVC_ROOT = PATHS['svc_prefit'] = _rare_svc_roots[0]
del _rare_svc_roots
"""
SVC_PATCHES = [
    ("/night24_readout_banks/COMPLETE.json'}\n\nRECIPE = {",
     "/night24_readout_banks/COMPLETE.json'}\n" + RARE_ROOT + "\nRECIPE = {"),
    (" 'helper_dataset': 'kirilltym/ntl-spectral-style-heads-paired-4404/1'}",
     " 'helper_dataset': 'kirilltym/ntl-spectral-style-heads-paired-4404/1',\n"
     f" 'svc_prefit_dataset': '{RARE}/1'}}"),
    (" {'source': 'v17.paths.svc_prefit',\n"
     "  'path': '/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit'},\n",
     " {'source': 'v17.paths.svc_prefit',\n"
     "  'path': RARE_SVC_ROOT},\n"
     " {'source': 'rare1516.readout', 'path': RARE_SVC_ROOT + '/readout.npz', 'kind': 'file', 'expected_bytes': 67042500},\n"),
]
SVC_MUST = ["svc_scores=cached_svc(paths['svc_prefit'],texts)", "    total=clipped_logp(svc_scores)\n"]

_h15 = share_variant(0.10, 0.15, "lightningv08 V28 with kirilltym's rare-class SVC readout (kirilltym/ntl-rare-class-svc-"
                     "1516 v1, as in his fork 0-639-teach-more-keep-the-signature V2/V3) in place of ntl627-svc-prefit, "
                     "and the Kirill spectral helper at 15% instead of 10%; Qwen3.8-27B stays at 10% on all rows.")
_spec = json.loads(json.dumps(_h15["spec"]).replace(STATUS15, STATUS))
_spec["must"] += SVC_MUST
_spec["must_not"] += [STATUS15]
VARIANTS = {
    "ntl-v28-rare-helper15": {
        "patches": [(old, new.replace(STATUS15, STATUS)) for old, new in _h15["patches"]] + SVC_PATCHES,
        "spec": _spec,
        "datasets": [RARE],
        "cells": [("%%writefile " + CHILD + "\n", "selftest_rare_child.py")],
        "extra_check": "selftest_rare_check.py",
    },
}
