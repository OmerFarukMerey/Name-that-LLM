"""Top public notebooks of the competition by score, with score and last run time (read-only)."""
import sys
sys.path.insert(0, '/Users/ofm/Desktop/Kaggle/Name-that-LLM/kaggle/lightning')
from prepush_check import api_client
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelsRequest
from kagglesdk.kernels.types.kernels_enums import KernelsListSortType
api = api_client()
with api.build_kaggle_client() as kaggle:
    r = ApiListKernelsRequest(); r.competition = 'name-that-llm'; r.sort_by = KernelsListSortType.SCORE_DESCENDING; r.page_size = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    res = kaggle.kernels.kernels_api_client.list_kernels(r)
    ks = res.kernels if hasattr(res, 'kernels') else res
    for k in ks:
        d = {f: getattr(k, f, None) for f in ('ref', 'best_public_score', 'last_run_time', 'total_votes', 'title')}
        print(f"{str(d['best_public_score']):>8}  {str(d['last_run_time'])[:16]}  votes {d['total_votes']:<4} {d['ref']}  | {d['title']}")
