"""Sibling likelihood for kimi-k3 (exp054): every train passage scored under the open Kimi-Linear-48B-A3B-Instruct
(MIT; Kimi-K3's text model is the same kimi_linear architecture with the same 163,840-token vocabulary and bos/eos),
in an empty-user assistant turn, with the per-token outputs ntl-segrun gives for the GLM models (log p of the true
token, entropy, E_p[log p ^ 2], rank) on the first 160 tokens.

Runs on a Kaggle CPU session with internet (no GPU quota): builds llama.cpp (newest release tag, kimi_linear needs
>= b7957) and a small scorer against its C API, downloads bartowski's Q3_K_M GGUF (sha256 checked), scores the
texts in a fixed shuffled order under a time cap (a partial run still covers every class), and prints a first
kimi-vs-rest readout.

    python kaggle/glmll/build_kimill.py          # writes kaggle/glmll/ntl-kimill/
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "ntl-kimill"
REPO = "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF"
FILE = "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_M.gguf"
SHA = "d63efc0d2824059bb1792e2f023cf3644d44173480530350ddf29cbd17dc146d"   # the HF LFS sha256 of FILE
PREFIX = "<|im_user|>user<|im_middle|><|im_end|><|im_assistant|>assistant<|im_middle|>"   # Kimi chat turns

SCORER = r'''// ntl-kll: per-token likelihood features of texts under a GGUF model, through the llama.cpp C API.
// usage: ntl-kll model.gguf texts.bin prefix out.bin cut threads budget_s [n_gpu_layers]
// texts.bin: u32 n, then n x (u32 byte length, utf-8 bytes). out.bin: per text u32 m, then m x 4 f32
// (log p of the true token, entropy, E_p[log p ^ 2], rank) for the first `cut` text tokens after the prefix.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 8) { fprintf(stderr, "usage: ntl-kll model texts.bin prefix out.bin cut threads budget_s [ngl]\n"); return 2; }
    const int cut = atoi(argv[5]), threads = atoi(argv[6]);
    const double budget = atof(argv[7]);
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = argc > 8 ? atoi(argv[8]) : 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "ntl-kll: model load failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024; cp.n_ubatch = 512; cp.n_seq_max = 1;
    cp.n_threads = threads; cp.n_threads_batch = threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ntl-kll: context failed\n"); return 4; }
    auto tokenize = [&](const std::string & s, bool special) {
        std::vector<llama_token> t(s.size() + 16);
        int n = llama_tokenize(vocab, s.data(), (int) s.size(), t.data(), (int) t.size(), false, special);
        if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, s.data(), (int) s.size(), t.data(), (int) t.size(), false, special); }
        t.resize(std::max(n, 0));
        return t;
    };
    const std::vector<llama_token> pre = tokenize(argv[3], true);
    fprintf(stderr, "ntl-kll: n_vocab %d, prefix tokens", n_vocab);
    for (llama_token x : pre) fprintf(stderr, " %d", x);
    fprintf(stderr, ", loaded in %.0f s\n", elapsed());
    FILE * fin = fopen(argv[2], "rb");
    FILE * fout = fopen(argv[4], "wb");
    uint32_t n_texts = 0;
    if (!fin || !fout || fread(&n_texts, 4, 1, fin) != 1) { fprintf(stderr, "ntl-kll: io error\n"); return 5; }
    llama_batch batch = llama_batch_init(1024, 0, 1);
    size_t tokens = 0;
    for (uint32_t k = 0; k < n_texts; k++) {
        uint32_t len = 0;
        if (fread(&len, 4, 1, fin) != 1) break;
        std::string s(len, '\0');
        if (len && fread(&s[0], 1, len, fin) != len) break;
        if (elapsed() > budget) { fprintf(stderr, "ntl-kll: budget reached after %u texts\n", k); break; }
        std::vector<llama_token> all = pre, tt = tokenize(s, false);
        if ((int) tt.size() > cut) tt.resize(cut);
        all.insert(all.end(), tt.begin(), tt.end());
        const int P = (int) pre.size(), n = (int) all.size();
        const uint32_t m = (uint32_t) (n - P);
        std::vector<float> feat(4 * (size_t) m, NAN);
        if (m > 0) {
            llama_memory_clear(llama_get_memory(ctx), true);
            for (int i = 0; i < n; i++) {
                batch.token[i] = all[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = (i >= P - 1 && i < n - 1);
            }
            batch.n_tokens = n;
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "ntl-kll: decode failed at text %u\n", k); return 6; }
            for (uint32_t j = 0; j < m; j++) {
                const float * lg = llama_get_logits_ith(ctx, P - 1 + (int) j);
                const float lt = lg[all[P + j]];
                double mx = -INFINITY;
                for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) lg[v]);
                double se = 0;
                for (int v = 0; v < n_vocab; v++) se += std::exp(lg[v] - mx);
                const double ls = std::log(se);
                double h = 0, m2 = 0;
                int rank = 0;
                for (int v = 0; v < n_vocab; v++) {
                    const double lp = (lg[v] - mx) - ls, p = std::exp(lp);
                    h -= p * lp; m2 += p * lp * lp; rank += lg[v] > lt;
                }
                feat[4 * j] = (float) ((lt - mx) - ls); feat[4 * j + 1] = (float) h;
                feat[4 * j + 2] = (float) m2; feat[4 * j + 3] = (float) rank;
            }
        }
        fwrite(&m, 4, 1, fout);
        fwrite(feat.data(), 4, feat.size(), fout);
        fflush(fout);
        tokens += m;
        if (k % 50 == 0) fprintf(stderr, "ntl-kll: %u/%u texts, %zu tokens, %.0f s\n", k + 1, n_texts, tokens, elapsed());
    }
    fprintf(stderr, "ntl-kll: done, %zu tokens, %.0f s\n", tokens, elapsed());
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    fclose(fout);
    return 0;
}
'''

RUN = f'''import glob, hashlib, os, re, shutil, struct, subprocess, sys, time
import numpy as np, pandas as pd
T0 = time.time()
W = '/kaggle/temp/kimi'
os.makedirs(W, exist_ok=True)
os.makedirs('/kaggle/working/out', exist_ok=True)
print('cpus', os.cpu_count(), 'disk free GB', shutil.disk_usage('/kaggle/temp').free // 2**30,
      'ram GB', os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // 2**30, flush=True)
assert shutil.disk_usage('/kaggle/temp').free > 30 * 2**30, 'not enough disk for the GGUF'

# llama.cpp: the newest release tag (kimi_linear landed in b7957)
refs = subprocess.run(['git', 'ls-remote', '--tags', 'https://github.com/ggml-org/llama.cpp'],
                      capture_output=True, text=True, check=True).stdout
tag = 'b%d' % max(int(x) for x in re.findall(r'refs/tags/b(\\d+)$', refs, re.M))
assert int(tag[1:]) >= 7957, tag
print('llama.cpp', tag, flush=True)
subprocess.run(['git', 'clone', '-q', '--depth', '1', '--branch', tag, 'https://github.com/ggml-org/llama.cpp', W + '/llama.cpp'], check=True)
subprocess.run(['cmake', '-S', W + '/llama.cpp', '-B', W + '/build', '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_CURL=OFF',
                '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_TOOLS=OFF',
                '-DGGML_NATIVE=ON'], check=True, stdout=subprocess.DEVNULL)
subprocess.run(['cmake', '--build', W + '/build', '--target', 'llama', '-j', str(os.cpu_count())], check=True, stdout=subprocess.DEVNULL)
lib = os.path.dirname(glob.glob(W + '/build/**/libllama.so', recursive=True)[0])
subprocess.run(['g++', '-O3', '-march=native', '-std=c++17', '/tmp/ntl-kll.cpp', '-I', W + '/llama.cpp/include',
                '-I', W + '/llama.cpp/ggml/include', '-L', lib, '-lllama', '-lggml', '-lggml-base', '-Wl,-rpath,' + lib,
                '-o', W + '/ntl-kll'], check=True)
print(f'built in {{time.time() - T0:.0f}} s', flush=True)

# the GGUF, checked against the hub's LFS sha256
from huggingface_hub import hf_hub_download
gguf = hf_hub_download({REPO!r}, {FILE!r}, local_dir=W + '/gguf')
h = hashlib.sha256()
with open(gguf, 'rb') as f:
    for blk in iter(lambda: f.read(1 << 24), b''):
        h.update(blk)
assert h.hexdigest() == {SHA!r}, h.hexdigest()
print(f'gguf ok ({{os.path.getsize(gguf) / 2**30:.1f}} GB) at {{time.time() - T0:.0f}} s', flush=True)

# texts in a fixed shuffled order, so a run cut by the time cap still covers every class
TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
df = pd.read_csv(TRAIN, dtype={{'id': str}})
order = np.random.default_rng(0).permutation(len(df))
with open(W + '/texts.bin', 'wb') as f:
    f.write(struct.pack('<I', len(order)))
    for i in order:
        b = str(df.text.iloc[i] if isinstance(df.text.iloc[i], str) else '').encode('utf-8')
        f.write(struct.pack('<I', len(b)) + b)
budget = 10.5 * 3600 - (time.time() - T0)   # the session limit is 12 h
r = subprocess.run([W + '/ntl-kll', gguf, W + '/texts.bin', {PREFIX!r}, W + '/out.bin', '160', str(os.cpu_count()),
                    str(int(budget))])
print('ntl-kll exit code', r.returncode, flush=True)

# assemble in the shuffled order, keep the scored texts
raw = open(W + '/out.bin', 'rb').read()
pos, rows, ids = 0, [], []
for i in order:
    if pos + 4 > len(raw):
        break
    (m,) = struct.unpack_from('<I', raw, pos)
    if pos + 4 + 16 * m > len(raw):
        break
    rows.append(np.frombuffer(raw, dtype='<f4', count=4 * m, offset=pos + 4).reshape(m, 4))
    ids.append(df.id.iloc[i])
    pos += 4 + 16 * m
off = np.concatenate([[0], np.cumsum([len(s) for s in rows])])
sc = np.concatenate(rows) if rows else np.zeros((0, 4), np.float32)
np.savez('/kaggle/working/out/kimi_scores.npz', ids=np.array(ids, dtype='U64'), offsets=off, scores=sc)
print(f'{{len(ids)}} / {{len(df)}} texts, {{len(sc)}} tokens, non-finite rows {{int((~np.isfinite(sc)).any(1).sum())}}, '
      f'{{time.time() - T0:.0f}} s', flush=True)

# first readout: per-class mean log p, and kimi-k3 vs rest AUC on it
from sklearn.metrics import roc_auc_score
lab = df.set_index('id').loc[ids, 'model'].to_numpy()
lpm = np.array([s[:, 0].mean() if len(s) else np.nan for s in rows])
hm = np.array([s[:, 1].mean() if len(s) else np.nan for s in rows])
ok = np.isfinite(lpm)
print(pd.DataFrame({{'model': lab[ok], 'lp_mean': lpm[ok], 'ent_mean': hm[ok]}}).groupby('model').mean().sort_values('lp_mean').round(3).to_string())
for name, x in (('lp_mean', lpm), ('lp_mean - ent_mean', lpm + hm)):
    print('AUC kimi-k3 vs rest from', name, round(roc_auc_score(lab[ok] == 'kimi-k3', x[ok]), 3))
    g = ok & np.isin(lab, ['kimi-k3', 'gpt-5.5'])
    print('AUC kimi-k3 vs gpt-5.5 from', name, round(roc_auc_score(lab[g] == 'kimi-k3', x[g]), 3), flush=True)
'''


def build():
    cells = [{"cell_type": "markdown", "metadata": {}, "source":
              f"# {NAME}\n\nPer-token likelihood of every train passage under the open Kimi-Linear-48B-A3B-Instruct "
              "(Q3_K_M GGUF, llama.cpp on CPU, empty-user assistant turn, first 160 tokens): a sibling likelihood "
              "for kimi-k3 (exp054)."},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
              "source": "%%writefile /tmp/ntl-kll.cpp\n" + SCORER},
             {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": RUN}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = HERE / NAME
    out.mkdir(exist_ok=True)
    (out / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{NAME}", "title": NAME, "code_file": f"{NAME}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": ["name-that-llm"],
            "kernel_sources": [], "model_sources": [], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {NAME}: {len(cells)} cells")


if __name__ == "__main__":
    build()
