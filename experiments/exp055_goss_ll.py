"""exp055: gpt-oss-20b likelihood as an OpenAI-family reference. Every train passage scored under OpenAI's open
gpt-oss-20b (kernel ntl-gossll, kaggle/glmll/build_gossll.py: MXFP4 GGUF on llama.cpp, harmony final-channel turn,
first 160 tokens), aimed at the kimi-k3 -> gpt-5.5 confusion and the GPT classes. exp054's protocol on these scores;
when ntl-kimill's scores are also in outputs/exp054_kimi_sibling_ll/, one more set adds them (rows scored by both).

    python -I experiments/exp055_goss_ll.py
"""
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / "experiments")]
import exp054_kimi_sibling_ll as K54  # noqa: E402

OUT = Path(os.environ.get("EXP055_OUT", _ROOT / "outputs" / "exp055_goss_ll"))

if __name__ == "__main__":
    kimi = K54.OUT / "kimi_scores.npz"
    K54.main(K54.load_scores(OUT / "goss_scores.npz"), "goss",
             {"kimi": K54.load_scores(kimi)} if kimi.exists() else None)
