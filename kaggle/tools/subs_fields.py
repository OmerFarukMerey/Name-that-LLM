"""Dump every non-empty field of our latest competition submissions (read-only)."""
import sys
sys.path.insert(0, '/Users/ofm/Desktop/Kaggle/Name-that-LLM/kaggle/lightning')
from prepush_check import api_client
from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
api = api_client()
with api.build_kaggle_client() as kaggle:
    r = ApiListSubmissionsRequest(); r.competition_name = 'name-that-llm'; r.page_size = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    res = kaggle.competitions.competition_api_client.list_submissions(r)
    subs = res.submissions if hasattr(res, 'submissions') else res
    for s in subs:
        d = {f: getattr(s, f, None) for f in dir(s) if not f.startswith('_') and not callable(getattr(s, f, None))}
        print({k: (str(v)[:80]) for k, v in d.items() if v not in (None, '', 0, False, [])})
