"""Forks of omkarpattanaik123's public 0.683 (name-that-llm-v28-sovereign-titan-apex, version 1, the scored one).

Version 1 = lightningv08's 0.681 (version 15) with heliosli's mount shim and ATLAS 15%, CHEN Han's second native expert
(Tencent WeMM-Embedding-4B, five folds, as wired in lightningv08's five-fold-embeddings notebook) at 7% after the
Qwen3-Embedding-8B expert's 40%, plus its own two steps: a mass-preserving sibling-style re-ranking inside the model
families and hard typographic negative evidence (-15 logits). GSQ share 0.06 and balancing target mix 0.35 are
version 15's. kirilltym/ntl-spectral-style-heads-paired-4404 is attached upstream but only named in the shim's table,
never read, so it is left out.

Knobs (all mixing settings: no extra runtime):
    tm50   balancing target mix 0.35 -> 0.50 } together our 0.682 on version 15
    g12    GSQ share 0.06 -> 0.12           }
    w04 / w05 / w10   WeMM share 0.07 -> 0.04 / 0.05 / 0.10
    nosib / sib05     sibling-style re-ranking cap 0.10 -> 0 (no-op) / 0.05 (offline on the OOF bank: 0.10 -0.0007,
                      0.05 -0.0007, 0.20 -0.0045 balanced accuracy, prompt-grouped)
    a10               ATLAS share on low-margin rows 0.15 -> 0.10 (version 15's)
    oq72 / oq100      the public 0.684's change ported: the two-stage Qwen3-14B member (n14b, weight 2.5) reads Nuri
                      Dudak's q14 LoRA v4 at 0.72 + Nishant Dahal's two-stage LoRA at 0.28 (or v4 alone) instead of the
                      two-stage LoRA alone. v4 is the same recipe (Qwen3-14B-Base, keep 27, same eight layers, max_len
                      447, eos 151643; its meta: OOF plain BA 0.6361). The member's checkpoint inventory gains v4's
                      four files and core_readout weights the adapters by `adapter_weights` when a spec has them.

    python kaggle/nlm/build_e683.py [variant ...]
"""
import hashlib
import json
import sys
from pathlib import Path

from build_l681 import KIRILL, patch, text

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "upstream_omkarpattanaik123_apex_v1.ipynb"
UPSTREAM_SHA = "1702ea2b9c6101c7"
URL = "https://www.kaggle.com/code/omkarpattanaik123/name-that-llm-v28-sovereign-titan-apex"

VARIANTS = {   # name: (target mix, GSQ share, WeMM share, sibling-style cap, ATLAS share[, q14 LoRA v4 share in n14b])
    "ntl-e683-g12-tm50": (".50", "0.12", "0.07", "0.10", ".15"),
    "ntl-e683-tm50": (".50", "0.06", "0.07", "0.10", ".15"),
    "ntl-e683-g12": (".35", "0.12", "0.07", "0.10", ".15"),
    "ntl-e683-g12-tm50-w10": (".50", "0.12", "0.10", "0.10", ".15"),
    "ntl-e683-g12-tm50-w04": (".50", "0.12", "0.04", "0.10", ".15"),
    "ntl-e683-nosib": (".35", "0.06", "0.07", "0.0", ".15"),
    "ntl-e683-nosib-w05": (".35", "0.06", "0.05", "0.0", ".15"),
    "ntl-e683-nosib-a10": (".35", "0.06", "0.07", "0.0", ".10"),
    "ntl-e683-w05": (".35", "0.06", "0.05", "0.10", ".15"),
    "ntl-e683-sib05": (".35", "0.06", "0.07", "0.05", ".15"),
    "ntl-e683-oq72": (".35", "0.06", "0.07", "0.10", ".15", "0.72"),
    "ntl-e683-oq100": (".35", "0.06", "0.07", "0.10", ".15", "1.0"),
}
OQ4 = "/kaggle/input/ntl-q14-lora-v4"
OQ4_FILES = ("{'adapter/adapter_config.json': 1326, 'adapter/adapter_model.safetensors': 173456688, "
             "'heads.safetensors': 6227664, 'meta.json': 1142}")   # nuridudak/ntl-q14-lora-v4 version 1, from the API listing
N14B_FILES = ("{'adapter/adapter_config.json': 1380, 'adapter/adapter_model.safetensors': 173456688, "
              "'heads.safetensors': 6227664, 'meta.json': 1015}")


def build(name):
    tm, gsq, wemm, sib, atlas, *rest = VARIANTS[name]
    oq = rest[0] if rest else None
    raw = UPSTREAM.read_bytes()
    assert hashlib.sha256(raw).hexdigest()[:16] == UPSTREAM_SHA, "upstream changed"
    nb = json.loads(raw)
    cells = nb["cells"]
    for c in cells:
        c["source"] = text(c)
    patch(cells, "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/core_math.py",
          "def balance_logits(P, train_labels, strength=.30, target_mix=.35):",
          f"def balance_logits(P, train_labels, strength=.30, target_mix={tm}):")
    ens = "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/ensemble.py"
    patch(cells, ens, "'sinkhorn_strength': .30, 'sinkhorn_target_mix': .35,", f"'sinkhorn_strength': .30, 'sinkhorn_target_mix': {tm},")
    patch(cells, ens, "assert config['gsq_share'] == 0.06, 'Unexpected GSQ share'", f"assert config['gsq_share'] == {gsq}, 'Unexpected GSQ share'")
    patch(cells, ens, "'experiment': 'qwen06_a15',", f"'experiment': '{name}',")
    patch(cells, ens, "official_labels.to_numpy(dtype=np.int64), groups, max_alpha=0.10,",
          f"official_labels.to_numpy(dtype=np.int64), groups, max_alpha={sib},")
    keep = 1 - float(atlas)
    patch(cells, ens, "prior[atlas_rows] = .85 * prior[atlas_rows] + .15 * atlas",
          f"prior[atlas_rows] = {f'{keep:.2f}'.lstrip('0').rstrip('0')} * prior[atlas_rows] + {atlas} * atlas")
    if oq:
        cfg = "CONFIG = {'RELEASE_READY': True"
        n14b = "'adapters': ['/kaggle/input/ntl-handwriting-blend/lora_qwen3_14b'], 'keep': 27"
        if float(oq) == 1:
            patch(cells, cfg, n14b, f"'adapters': ['{OQ4}'], 'keep': 27")
            patch(cells, cfg, f"'n14b': [{N14B_FILES}]", f"'n14b': [{OQ4_FILES}]")
        else:
            patch(cells, cfg, n14b, f"'adapters': ['{OQ4}', '/kaggle/input/ntl-handwriting-blend/lora_qwen3_14b'], "
                  f"'adapter_weights': [{oq}, {1 - float(oq):.2f}], 'keep': 27")
            patch(cells, cfg, f"'n14b': [{N14B_FILES}]", f"'n14b': [{OQ4_FILES}, {N14B_FILES}]")
            rd = "%%writefile /kaggle/temp/ntl-gsq-v27-runtime/core_readout.py"
            a = "    adapters = spec['adapters'] if spec['type'] == 'l24' else [spec['ckpt']]\n"
            patch(cells, rd, a, a + "    aw = spec.get('adapter_weights')   # per-adapter weights (the public 0.684's change); equal otherwise\n"
                  "    assert aw is None or (len(aw) == len(adapters) and abs(sum(aw) - 1) < 1e-9)\n")
            patch(cells, rd, "            plain = plain + lsm(logits) / len(adapters)\n",
                  "            plain = plain + (lsm(logits) / len(adapters) if aw is None else lsm(logits) * aw[i])\n")
            c = "lsm(Zc @ data['centered_weight'].T + data['centered_bias'])"
            patch(cells, rd, f"                centered = centered + {c} / len(adapters)\n",
                  f"                centered = centered + ({c} / len(adapters) if aw is None else {c} * aw[i])\n")
    last = "CONFIG['gsq_share'] = 0.06\n"
    patch(cells, last, "'share': 0.07, 'deadline_seconds': 30240}", f"'share': {wemm}, 'deadline_seconds': 30240}}")
    patch(cells, last, last, f"CONFIG['gsq_share'] = {gsq}\n")
    cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source":
                     f"# {name}\n\nomkarpattanaik123's [0.683]({URL}) (version 1); balancing target mix {tm}, GSQ share {gsq}, "
                     f"WeMM share {wemm}, sibling-style cap {sib}, ATLAS {atlas}" + (f", n14b q14 LoRA v4 share {oq}." if oq else ".")})
    out = HERE / name
    out.mkdir(exist_ok=True)
    (out / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    up = json.loads((HERE / "upstream_omkarpattanaik123_apex_v1.meta.json").read_text(encoding="utf-8"))
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
