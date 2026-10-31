"""Top public notebooks of the competition by score: dump every non-empty field of the first N (read-only)."""
import sys
sys.path.insert(0, '/Users/ofm/Desktop/Kaggle/Name-that-LLM/kaggle/lightning')
from prepush_check import api_client
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest
from kagglesdk.kernels.types.kernels_enums import KernelsListSortType
api = api_client()
with api.build_kaggle_client() as kaggle:
    r = ApiListKernelsRequest(); r.competition = 'name-that-llm'; r.sort_by = KernelsListSortType.SCORE_DESCENDING; r.page_size = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    res = kaggle.kernels.kernels_api_client.list_kernels(r)
    ks = res.kernels if hasattr(res, 'kernels') else res
    for k in ks:
        d = {f: getattr(k, f) for f in dir(k) if not f.startswith('_') and not callable(getattr(k, f, None))}
        print({f: v for f, v in d.items() if v not in (None, '', 0, False, [])})
