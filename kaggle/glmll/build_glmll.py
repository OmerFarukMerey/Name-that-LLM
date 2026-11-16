"""Train likelihood features for the GLM stage of the public 0.689-0.694 line: every train passage scored under
GLM-5.3-Flash and GLM-5.3, so its likelihood head (estopadilla's ll_model.json, the same weights in all of those
notebooks) can be refit offline.

v1src/ = the files of the 0.689's V1 payload (estopadilla, public, Apache 2.0), unchanged: run_pipeline.py drives
ntl-segrun (his kernel output estopadilla/ntl-glm-likelihood-train) over jamesliebel's IQ3_XXS segment mirrors.
Same prefix, seq-batch and decode settings as the test-time stage; cut at 512 tokens instead of 160 so the per-token
scores (out/<tag>_scores.npz) rebuild the 160-token features and any longer cut offline.

    python kaggle/glmll/build_glmll.py          # writes kaggle/glmll/ntl-glmll-{flash,glm53}/
    python kaggle/glmll/build_glmll.py 160      # ntl-glmll160-{flash,glm53}/: cut as the test-time stage cuts
    python kaggle/glmll/build_glmll.py 160 1    # ntl-glmll160s1-glm53/: train rows shuffled (seed 1) first
    python kaggle/glmll/build_glmll.py pool     # ntl-glmllpool-{flash,glm53}/: exp053's open-pool rows, cut 160

The 512 cut makes each decode of 8 passages as long as its texts (up to 394 tokens); the stage cuts at 160, so most of
its decodes are 8 x 173 rows. GLM-5.3 and Flash are mixture-of-experts and their per-token scores move with the decode
shape (GLM-5.3: median 0.10 nats per token between our 512-cut run and a stage run on the same texts), so the head
gets trained on 160-cut scores to match the test.

exp051 then found the 160-cut run identical to the 512-cut one on most rows (feature gap median 0) and both still
0.10 nats per token from the stage: what moves GLM-5.3's scores is which 7 texts share a text's decode (run_pipeline
decodes the rows in input order). A seed shuffles the train rows before scoring, so every text gets new neighbours,
as on the test: a second noisy view of each row, as the test-time features are.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "v1src"
# the 0.689's pinned image: the one its test-time GLM stage runs ntl-segrun in
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
MODELS = {  # tag: (segment datasets, file pattern, segment count)
    "flash": (["jamesliebel/glm53-flash-iq3xxs-segs-mirror"], "flash_iq3xxs-seg*-of-06-*.gguf", 6),
    "glm53": (["jamesliebel/glm53-iq3xxs-segs-a-mirror", "jamesliebel/glm53-iq3xxs-segs-b-mirror"],
              "glm53_iq3xxs-seg*-of-19-*.gguf", 19),
}
MAX_TEXT_TOKENS = 512   # default cut; build(tag, cut) takes another

RUN = '''import glob, json, os, shutil, subprocess, sys, time
TAG, PATTERN, N_SEG, MAX_TEXT_TOKENS = {tag!r}, {pattern!r}, {n_seg}, {max_tok}{seed_line}
PREFIX = '[gMASK]<sop><|system|>Reasoning Effort: Max<|user|><|assistant|><think></think>'   # the test-time stage's
subprocess.run('nvidia-smi -L; nproc; free -g; df -h /kaggle/working /kaggle/temp /tmp', shell=True)
TRAIN = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
segs = {{}}
for p in sorted(glob.glob('/kaggle/input/**/' + PATTERN, recursive=True)):
    segs.setdefault(os.path.basename(p), p)   # a file may be mounted under two roots
assert len(segs) == N_SEG, sorted(segs)
SEGDIR = f'/kaggle/temp/segs_{{TAG}}'
os.makedirs(SEGDIR, exist_ok=True)
for name, p in segs.items():
    os.symlink(p, os.path.join(SEGDIR, name))
art = sorted(os.path.dirname(p) for p in glob.glob('/kaggle/input/**/ntl-bin/ntl-segrun', recursive=True))[0]
for f in ('lib/libcublas.so.13', 'lib/libcublasLt.so.13', 'lib/libcudart.so.13'):
    assert os.path.isfile(os.path.join(art, f)), f
BIN = '/kaggle/temp/ntl-bin'
os.makedirs(BIN, exist_ok=True)
shutil.copyfile(os.path.join(art, 'ntl-segrun'), os.path.join(BIN, 'ntl-segrun'))
os.chmod(os.path.join(BIN, 'ntl-segrun'), 0o755)
env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(art, 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
WORK = max(['/kaggle/temp', '/tmp'], key=lambda d: shutil.disk_usage(d).free) + '/glmll-work'
OUT = '/kaggle/working/out'
print(TRAIN, art, WORK, len(segs), 'segments', flush=True){shuffle}
cmd = [sys.executable, '-u', '/kaggle/temp/glmll/run_pipeline.py', '--bin', BIN, '--segs', SEGDIR + '/*.gguf',
       '--csv', TRAIN, '--out', OUT, '--tag', TAG, '--work', WORK, '--seq-batch', '8', '--state-dtype', 'auto',
       '--max-text-tokens', str(MAX_TEXT_TOKENS), '--prefix', PREFIX]
t = time.time()
r = subprocess.run(cmd, env=env)
print(f'{{TAG}}: exit {{r.returncode}} after {{time.time() - t:.0f}} s', flush=True)
subprocess.run(f'ls -la {{OUT}}; tail -5 {{OUT}}/{{TAG}}_run.log', shell=True)
assert r.returncode == 0
'''


SHUFFLE = '''
import pandas as pd
SHUFFLED = '/kaggle/temp/train_shuffled.csv'   # new decode neighbours for every text; the outputs list ids in this order
pd.read_csv(TRAIN, dtype={'id': str}).sample(frac=1.0, random_state=SEED).to_csv(SHUFFLED, index=False)
TRAIN = SHUFFLED'''

POOL = '''
import pandas as pd
POOL_IDS = {ids!r}
pool = pd.read_parquet(next(p for p in glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)),
                       columns=['id', 'text']).set_index('id')
missing = [i for i in POOL_IDS if i not in pool.index]
assert not missing, missing[:5]
TRAIN = '/kaggle/temp/pool_rows.csv'   # exp053's rows in pool_rows_v17.csv order (labels mixed)
pool.loc[POOL_IDS].reset_index()[['id', 'text']].to_csv(TRAIN, index=False)
print(len(POOL_IDS), 'pool rows', flush=True)'''
POOL_DATASET = "crtproduction/name-that-llm-open-pool"


def build(tag, cut=MAX_TEXT_TOKENS, seed=None, pool=False):
    datasets, pattern, n_seg = MODELS[tag]
    name = (f"ntl-glmll-{tag}" if cut == MAX_TEXT_TOKENS else f"ntl-glmll{cut}-{tag}") if seed is None else \
        f"ntl-glmll{cut}s{seed}-{tag}"
    if pool:
        import pandas as pd
        name, datasets = f"ntl-glmllpool-{tag}", datasets + [POOL_DATASET]
        ids = pd.read_csv(HERE / "pool_rows_v17.csv").id.astype(str).tolist()
    md = (f"# {name}\n\nPer-token log-likelihood of every train passage under "
          f"{'GLM-5.3-Flash' if tag == 'flash' else 'GLM-5.3'}, for refitting the likelihood head of the GLM stage "
          "offline. The three code files are estopadilla's, from the V1 payload of his public "
          "[0.689](https://www.kaggle.com/code/estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4) (Apache 2.0), "
          "unchanged; `ntl-segrun` is his build ([ntl-glm-likelihood-train]"
          "(https://www.kaggle.com/code/estopadilla/ntl-glm-likelihood-train)); the GGUF segments are jamesliebel's "
          f"mirrors. Settings as the test-time stage, cut at {cut} tokens."
          + ("" if seed is None else f" Train rows shuffled (seed {seed}) before scoring, so each passage shares its "
             "decode with other passages than in the unshuffled run.")
          + ("" if not pool else " The passages are exp053's 2,000 open-pool rows (kaggle/glmll/pool_rows_v17.csv), "
             "not the train rows."))
    cells = [{"cell_type": "markdown", "metadata": {}, "source": md}]
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": "import os\nos.makedirs('/kaggle/temp/glmll', exist_ok=True)"})
    for f in ("segio.py", "features.py", "run_pipeline.py"):
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                      "source": f"%%writefile /kaggle/temp/glmll/{f}\n" + (SRC / f).read_text(encoding="utf-8")})
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                  "source": RUN.format(tag=tag, pattern=pattern, n_seg=n_seg, max_tok=cut,
                                        seed_line="" if seed is None else f"\nSEED = {seed}",
                                        shuffle=POOL.format(ids=ids) if pool else "" if seed is None else SHUFFLE)})
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "dataset_sources": datasets, "competition_sources": ["name-that-llm"],
            "kernel_sources": ["estopadilla/ntl-glm-likelihood-train"], "model_sources": [],
            "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4", "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {name}: {len(cells)} cells")


if __name__ == "__main__":
    import sys
    cut = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "pool" else MAX_TEXT_TOKENS
    if sys.argv[1:2] == ["pool"]:
        for t in MODELS:
            build(t, 160, pool=True)
    elif len(sys.argv) > 2:   # a shuffled view: GLM-5.3 only (Flash moves 0.005 nats per token from the stage at 160)
        build("glm53", cut, int(sys.argv[2]))
    else:
        for t in MODELS:
            build(t, cut)
