"""Likelihood under gpt-oss-20b (exp055): every train passage scored under OpenAI's open gpt-oss-20b (Apache 2.0; the
o200k_harmony vocabulary of the GPT-5 line), as an OpenAI-family reference for the kimi-k3 / gpt-5.5 confusion and the
GPT classes. Same kernel as ntl-kimill (build_kimill.py: llama.cpp built on a Kaggle CPU session, the C-API scorer,
first 160 tokens) with ggml-org's MXFP4 GGUF, in a harmony final-channel assistant turn after an empty user turn.

    python kaggle/glmll/build_gossll.py          # writes kaggle/glmll/ntl-gossll/
"""
from build_kimill import build

PREFIX = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
          "Knowledge cutoff: 2024-06\nCurrent date: 2026-11-16\n\nReasoning: low\n\n"
          "# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
          "<|start|>user<|message|><|end|><|start|>assistant<|channel|>final<|message|>")   # gpt-oss's chat template

READOUT = '''# first readout: per-class mean log p; OpenAI family vs rest, kimi-k3 vs gpt-5.5
from sklearn.metrics import roc_auc_score
lab = df.set_index('id').loc[ids, 'model'].to_numpy()
lpm = np.array([s[:, 0].mean() if len(s) else np.nan for s in rows])
hm = np.array([s[:, 1].mean() if len(s) else np.nan for s in rows])
ok = np.isfinite(lpm)
print(pd.DataFrame({'model': lab[ok], 'lp_mean': lpm[ok], 'ent_mean': hm[ok]}).groupby('model').mean().sort_values('lp_mean').round(3).to_string())
gpt = np.char.startswith(lab.astype(str), 'gpt-')
for name, x in (('lp_mean', lpm), ('lp_mean - ent_mean', lpm + hm)):
    print('AUC OpenAI vs rest from', name, round(roc_auc_score(gpt[ok], x[ok]), 3))
    g = ok & np.isin(lab, ['kimi-k3', 'gpt-5.5'])
    print('AUC gpt-5.5 vs kimi-k3 from', name, round(roc_auc_score(lab[g] == 'gpt-5.5', x[g]), 3), flush=True)
'''

GOSS = {"name": "ntl-gossll", "repo": "ggml-org/gpt-oss-20b-GGUF", "file": "gpt-oss-20b-MXFP4.gguf",
        "sha": "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",   # the HF LFS sha256
        "prefix": PREFIX, "work": "goss", "tag_note": "gpt-oss since b6096; the scorer is the same as ntl-kimill's",
        "out": "goss_scores.npz", "readout": READOUT,
        "md": "Per-token likelihood of every train passage under OpenAI's open gpt-oss-20b (MXFP4 GGUF, llama.cpp on "
              "CPU, harmony final-channel turn after an empty user turn, first 160 tokens): an OpenAI-family reference "
              "for the kimi-k3 / gpt-5.5 confusion (exp055)."}

if __name__ == "__main__":
    build(GOSS)
