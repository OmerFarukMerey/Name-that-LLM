"""Forks of lightningv08's public 0.681 (0-681-qwen3-embedding-lora-qwen3-8-27b, version 15, the scored one).

Version 15 = his weighted multi-model stack + CHEN Han's (concyclics) five-fold Qwen3-Embedding-8B native expert at a 40%
geometric share, the Qwen3.8-27B GSQ3 classifier at 6% inside the parent. Its cells bind fixed /kaggle/input paths for the
mount layout of the session that scored; heliosli's mount shim (lv681 a15, scored 0.681 with ATLAS 15%) maps them to
either layout, and this builder takes that shim unchanged. kirilltym/ntl-spectral-style-heads-paired-4404 is attached
upstream but never read (no cell names its files), so it is left out.

Knobs (all are mixing settings: no extra runtime):
    tm50   balancing target mix 0.35 -> 0.50 (aboodps's 0.669 step; +0.001 three times on our 0.66x line)
    g09    GSQ share 0.06 -> 0.09 (lightningv08: 0.03 -> 0.06 was 0.680 -> 0.681)
    g12    GSQ share 0.06 -> 0.12
    n35    native share 0.40 -> 0.35 (0.50 scored 0.679 against 0.680 at GSQ 0.03)

    python kaggle/nlm/build_l681.py [variant ...]
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_lightningv08_0681_v15.ipynb"
UPSTREAM_SHA = "334ca315934044bb"
SHIM_SRC = HERE / "upstream_heliosli_lv681_a15_v2.ipynb"
KIRILL = "kirilltym/ntl-spectral-style-heads-paired-4404"

VARIANTS = {   # name: (target mix, GSQ share, native share)
    "ntl-l681-tm50": (".50", "0.06", "0.4"),
    "ntl-l681-g09": (".35", "0.09", "0.4"),
    "ntl-l681-g09-tm50": (".50", "0.09", "0.4"),
    "ntl-l681-g12-tm50": (".50", "0.12", "0.4"),
    "ntl-l681-n35-g09-tm50": (".50", "0.09", "0.35"),
}


def text(cell):
    return cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])


def patch(cells, starts, old, new):
    """Replace `old` once inside the one code cell whose source starts with `starts`."""
    hits = [c for c in cells if c["cell_type"] == "code" and text(c).startswith(starts)]
    assert len(hits) == 1, (starts, len(hits))
    s = text(hits[0])
    assert s.count(old) == 1, (starts, old[:80], s.count(old))
    hits[0]["source"] = s.replace(old, new)


def build(name):
    tm, gsq, native = VARIANTS[name]
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
    shim = next(text(c) for c in json.loads(SHIM_SRC.read_text(encoding="utf-8"))["cells"]
                if c["cell_type"] == "code" and text(c).startswith("# Mount shim added by heliosli"))
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    cells[first_code:first_code] = [
        {"cell_type": "markdown", "metadata": {}, "source": "## Mount shim (heliosli)\n\nVersion 15's cells bind fixed input "
         "paths for one mount layout; this cell maps each of them to wherever that source is mounted in this session."},
        {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": shim}]
    # heliosli's six applications of the shim, unchanged
    patch(cells, "import csv, json, os, sys, time", "DATA = Path('/kaggle/input/competitions/name-that-llm/test.csv')",
          "DATA = Path(_ntl_fix('/kaggle/input/competitions/name-that-llm/test.csv'))")
    patch(cells, "CONFIG = {'RELEASE_READY': True", "\n(RUNTIME/'config.json').write_text(json.dumps(CONFIG),encoding='utf-8')",
          "\nCONFIG, EXPLICIT_PATHS = _ntl_fix_obj(CONFIG), _ntl_fix_obj(EXPLICIT_PATHS)\n_ntl_audit(CONFIG, EXPLICIT_PATHS)"
          "\n(RUNTIME/'config.json').write_text(json.dumps(CONFIG),encoding='utf-8')")
    patch(cells, "import shutil", "ULTRA_ROOT = Path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl')",
          "ULTRA_ROOT = Path(_ntl_fix('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl'))")
    nc = next(text(c) for c in cells if c["cell_type"] == "code" and text(c).startswith("NUMERIC_CACHE_PARTS = {"))
    first_line = nc.split("\n", 1)[0] + "\n"
    patch(cells, "NUMERIC_CACHE_PARTS = {", first_line, first_line + "NUMERIC_CACHE_PARTS = _ntl_fix_obj(NUMERIC_CACHE_PARTS)\n")
    patch(cells, "if not preview:\n    import zipfile", "    source = Path('/kaggle/input/ntl-olmo2-native24-final-inference-assets')",
          "    source = Path(_ntl_fix('/kaggle/input/ntl-olmo2-native24-final-inference-assets'))")
    last = "CONFIG['gsq_share'] = 0.06"
    ne = ("CONFIG['native_embedding'] = {'base': '/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/8b/1', 'assets': "
          "'/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679', 'share': 0.4}\n")
    patch(cells, last, ne, ne.replace("'share': 0.4}", f"'share': {native}}}")
          + "CONFIG['native_embedding'] = _ntl_fix_obj(CONFIG['native_embedding'])\n")
    # the knobs
    patch(cells, "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/core_math.py",
          "def balance_logits(P, train_labels, strength=.30, target_mix=.35):",
          f"def balance_logits(P, train_labels, strength=.30, target_mix={tm}):")
    ens = "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/ensemble.py"
    patch(cells, ens, "'sinkhorn_strength': .30, 'sinkhorn_target_mix': .35,", f"'sinkhorn_strength': .30, 'sinkhorn_target_mix': {tm},")
    patch(cells, ens, "assert config['gsq_share'] == 0.06, 'Unexpected GSQ share'", f"assert config['gsq_share'] == {gsq}, 'Unexpected GSQ share'")
    patch(cells, ens, "'experiment': 'qwen06',", f"'experiment': '{name}',")
    patch(cells, last, "CONFIG['gsq_share'] = 0.06\n", f"CONFIG['gsq_share'] = {gsq}\n")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\nlightningv08's [0.681](https://www.kaggle.com/code/lightningv08/0-681-qwen3-embedding-lora-qwen3-8-27b) "
                     f"(version 15) with heliosli's mount shim; balancing target mix {tm}, GSQ share {gsq}, native share {native}."})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_lightningv08_0681_v15.meta.json").read_text(encoding="utf-8"))
    datasets = [d for d in up["datasetDataSources"] if d != KIRILL]
    assert len(datasets) == len(up["datasetDataSources"]) - 1
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": datasets, "competition_sources": up["competitionDataSources"],
            "kernel_sources": up["kernelDataSources"], "model_sources": up["modelDataSources"],
            "docker_image": up["dockerImage"], "machine_shape": up["machineShape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256("".join(text(c) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(cells)} cells, code sha256 {sha}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        build(name)
