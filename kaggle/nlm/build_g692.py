"""Forks of the public 0.692 (mathleteutd/name-that-llm-glm-5-3-likelihood-gsq-3, version 3).

The 0.692 = estopadilla's 0.689 (alexchilton's all-public stack, i.e. lightningv08's version 15 with the Sinkhorn
target mix 0.50 of our 0.682, + his "V1" stage: GLM-5.3-Flash and GLM-5.3 likelihoods from jamesliebel's IQ3_XXS GGUF
segment mirrors on the texts the stack ranks most GLM-like, blended into final-logits.npz) with heliosli-style mount
shim and the GSQ share 0.12 -> 0.03 (+0.003 over the 0.689).

Knobs (CPU post-processing ported unchanged from the public 0.683, omkarpattanaik123; no extra GPU time, so the V1
stage keeps its time budget):
    neg   the 0.683's typographic negative evidence (-15 logits) at the end of core_math.finalize (every native fold)
    sib   the 0.683's sibling-style re-ranking (its sibling_style.py and its ensemble.py block, unchanged) on the logits
          right after the native fusion, so final-logits.npz and the V1 stage see the re-ranked logits
          (on the 0.683, removing it lost 0.005; on the 0.684 adding it was -0.001)

    python kaggle/nlm/build_g692.py [variant ...]
"""
import hashlib
import json
import sys
from pathlib import Path

from build_l681 import patch, text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_mathleteutd_692_v3.ipynb"
UPSTREAM_SHA = "517c5733bb81854e"
URL = "https://www.kaggle.com/code/mathleteutd/name-that-llm-glm-5-3-likelihood-gsq-3"
E683 = HERE / "upstream_omkarpattanaik123_apex_v1.ipynb"
RUNTIME = "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/"

VARIANTS = {   # name: (negative rules, sibling-style re-ranking)
    "ntl-g692-negsib": (True, True),
    "ntl-g692-sib": (False, True),
    "ntl-g692-neg": (True, False),
}


def e683_code():
    return [text(c) for c in json.loads(E683.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def build(name):
    neg, sib = VARIANTS[name]
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
    ecode = e683_code()
    if neg:
        ecm = next(c for c in ecode if c.startswith(RUNTIME + "core_math.py"))
        block = ecm[ecm.index("    # Soft evidence logit update"):ecm.index("    return res\n") + len("    return res\n")]
        patch(cells, RUNTIME + "core_math.py", "    return total + gate[:, None] * evidence\n", block)
    if sib:
        eens = next(c for c in ecode if c.startswith(RUNTIME + "ensemble.py"))
        i = eens.index("    # --- UPGRADE: SIBLING STYLE RESIDUAL RE-RANKING ---")
        block = eens[i:eens.index("\n\n", i) + 2]
        save = "    np.savez(work / 'final-logits.npz', ids=np.asarray(ids), labels=np.asarray(LABELS), logits=logits)\n"
        patch(cells, RUNTIME + "ensemble.py", save, block + save)
        module = next(c for c in ecode if c.startswith(RUNTIME + "sibling_style.py"))
        at = next(i for i, c in enumerate(cells) if c["cell_type"] == "code" and text(c).startswith(RUNTIME + "ensemble.py"))
        cells[at:at] = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": module}]
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\nThe public [0.692]({URL}) (version 3); the 0.683's typographic negative rules {neg}, "
                     f"sibling-style re-ranking {sib}."})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_mathleteutd_692_v3.meta.json").read_text(encoding="utf-8"))
    meta = {"id": f"omerfarukmerey/{name}", "title": name, "code_file": f"{name}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": up["datasetDataSources"], "competition_sources": up["competitionDataSources"],
            "kernel_sources": up["kernelDataSources"], "model_sources": up["modelDataSources"],
            "docker_image": up["dockerImage"], "machine_shape": up["machineShape"], "keywords": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    sha = hashlib.sha256("".join(text(c) for c in cells if c["cell_type"] == "code").encode()).hexdigest()[:16]
    print(f"built {name}: {len(cells)} cells, code sha256 {sha}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        build(name)
