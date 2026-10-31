"""Build our private variants of lightningv08's V28 (public 0.642) for the 2026-10-26 submissions.

    python kaggle/lightning/build_variants.py [kernel ...]        (default: every variant)

Each variant is upstream_v28.ipynb plus:
  1. build.py's mount-path REWRITES (an API-pushed kernel mounts datasets at /kaggle/input/datasets/<owner>/<slug>);
  2. its own exact-match PATCHES (each old string must occur exactly once in the code cells, after the REWRITES);
  3. a commit-time self-test (selftest_variant_cell.py, filled with the variant's SPEC), inserted after the UltraBERT
     cell and before the run cell. On the 3-row public test the notebook writes a canned preview and skips all
     inference, so the self-test is the only thing that exercises our edits before the hidden rerun.

The variants come from the survey of every public notebook at 0.635+ (2026-10-25, scratchpad survey/result.json):
  ntl-v28-ctrl        no change: measures whether our API-pushed fork reproduces 0.642 (never published)
  ntl-v28-helper15    Kirill spectral helper share 0.10 -> 0.15 (cell 67: prior = .85*prior + .15*helper)
  ntl-v28-helper-125  the same at 0.125, a smaller step
  ntl-v28-qwen-15     Qwen3.8-27B probability share 0.10 -> 0.15 (recipe value, read by blend_selected)
  ntl-v28-qwen-20     the same at 0.20 (added 2026-10-26: the Qwen share is the only lever the public LB supports).
                      Pushed after upstream's inputs moved on (lightning assets v14 -> v15; lucasboesen's
                      gen_passages.csv, which V28 reads at runtime, v7 3.8 MB -> v8 5.1 MB). Pinning them as
                      owner/slug/version in kernel-metadata.json did not work: Kaggle attached the new versions
                      anyway (input 145.95 GB against the control's 134.1 GB). Not comparable with V28's 0.642.
  ntl-v28-mistral-5 / -10 (variants_mistral.py) our full-fit Mistral-7B member at 5% / 10% of the prior; they add
                      a notebook-output source and a model source (variant keys "kernels" and "models").
(ntl-v28-qwen15 and ntl-v28-helper125 are dead kernels: their first push was refused at the 2-GPU-session limit and every
later commit ran without the competition data mounted, so they were re-created under these names.)
Variants defined in variants_*.py modules next to this file (VARIANTS dicts of the same shape) are added too.

The build writes kaggle/lightning/<kernel>/: the notebook and kernel-metadata.json (kernel-metadata.base.json, without
our SVC-tiers dataset, plus the variant's extra datasets). Run prepush_check.py <kernel> right before pushing.
"""
import hashlib
import importlib
import json
import sys
from pathlib import Path

import build
from variant_recipes import GUARD, HELPER_LINE, STATUS, share_variant

HERE = Path(__file__).resolve().parent
BASE, BASE_SHA256 = "upstream_v28.ipynb", build.BASES["v28"][1]
SVC_TIERS = "omerfarukmerey/ntl-svc627-quality-tiers"

CTRL_SPEC = {"status": STATUS, "qwen": 0.10, "helper": 0.10,
             "must": [f"if recipe.get('status')!='{STATUS}':", GUARD, HELPER_LINE, "probabilities=sinkhorn(mixed)"],
             "must_not": []}
VARIANTS = {
    "ntl-v28-ctrl": {"patches": [], "spec": CTRL_SPEC, "datasets": []},
    "ntl-v28-helper15": share_variant(0.10, 0.15, "lightningv08 V28 with the Kirill spectral helper at 15% instead of "
                                      "10%; Qwen3.8-27B stays at 10% on all rows."),
    "ntl-v28-helper-125": share_variant(0.10, 0.125, "lightningv08 V28 with the Kirill spectral helper at 12.5% instead "
                                       "of 10%; Qwen3.8-27B stays at 10% on all rows."),
    "ntl-v28-qwen-15": share_variant(0.15, 0.10, "lightningv08 V28 with Qwen3.8-27B at 15% instead of 10% on all rows; "
                                    "the Kirill spectral helper stays at 10%."),
    "ntl-v28-qwen-20": share_variant(0.20, 0.10, "lightningv08 V28 with Qwen3.8-27B at 20% instead of 10% on all rows; "
                                    "the Kirill spectral helper stays at 10%."),
}
for module in sorted(HERE.glob("variants_*.py")):
    extra = importlib.import_module(module.stem).VARIANTS
    if set(extra) & set(VARIANTS):
        raise SystemExit(f"{module.name} redefines {sorted(set(extra) & set(VARIANTS))}")
    VARIANTS.update(extra)

ANCHOR = "ULTRA_ROOT = Path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl')"
RUN = "if not preview:\n    import subprocess, traceback"


def build_variant(kernel):
    variant = VARIANTS[kernel]
    raw = (HERE / BASE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise SystemExit(f"{BASE} is not the verified V28")
    nb = json.loads(raw)
    cells = nb["cells"]
    for old, new, expected in build.REWRITES:
        found, already = build.count(cells, old), build.count(cells, new)
        if found != expected or already:
            raise SystemExit(f"rewrite {old!r}: {found} hits (expected {expected}), {already} already rewritten")
        for c in cells:
            c["source"] = c["source"].replace(old, new)
    for old, new in variant["patches"]:
        code = [c for c in cells if c["cell_type"] == "code"]
        if build.count(code, old) != 1:
            raise SystemExit(f"{kernel}: patch matched {build.count(code, old)} times: {old!r}")
        i = build.find_cell(cells, old)
        cells[i]["source"] = cells[i]["source"].replace(old, new)
    for header, name in variant.get("cells", []):  # extra cells (e.g. a %%writefile helper), inserted before the self-test
        cells.insert(build.find_cell(cells, ANCHOR) + 1, build.code_cell(header + (HERE / name).read_text(encoding="utf-8").rstrip("\n")))
    template = (HERE / "selftest_variant_cell.py").read_text(encoding="utf-8").rstrip("\n")
    spec = dict(variant["spec"], kernel=kernel)
    if template.count("SPEC = None") != 1:
        raise SystemExit("selftest_variant_cell.py must contain 'SPEC = None' once")
    test_source = template.replace("SPEC = None", "SPEC = " + json.dumps(spec, indent=1)).replace("{KERNEL}", kernel)
    extra_check = variant.get("extra_check")
    if extra_check:
        if test_source.count("    # (e) variant-specific checks\n") != 1:
            raise SystemExit("selftest_variant_cell.py lacks the (e) marker")
        body = "".join("    " + line if line.strip() else line
                       for line in (HERE / extra_check).read_text(encoding="utf-8").splitlines(keepends=True))
        test_source = test_source.replace("    # (e) variant-specific checks\n", "    # (e) variant-specific checks\n" + body)
    position = max(build.find_cell(cells, ANCHOR), *[i for i, c in enumerate(cells)
                                                     if c["cell_type"] == "code" and c["source"].startswith("%%writefile")])
    cells.insert(position + 1, build.code_cell(test_source))

    test = build.find_cell(cells, "def _ntl_variant_selftest():")
    run = build.find_cell(cells, RUN)
    writefiles = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith("%%writefile")]
    if not (max(writefiles) < test < run):
        raise SystemExit(f"self-test cell {test} must follow every %%writefile ({max(writefiles)}) and precede run {run}")
    for line in spec["must"]:
        if sum(c["source"].count(line) for c in cells if c["cell_type"] == "code" and c["source"].startswith("%%writefile")) < 1:
            raise SystemExit(f"{kernel}: required line not in any written module: {line!r}")

    out = HERE / kernel
    out.mkdir(exist_ok=True)
    text = json.dumps(nb, indent=1, ensure_ascii=False)
    (out / f"{kernel}.ipynb").write_text(text, encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb",
            **json.loads((HERE / "kernel-metadata.base.json").read_text(encoding="utf-8"))}
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if d != SVC_TIERS] + variant["datasets"]
    meta["kernel_sources"] = meta["kernel_sources"] + variant.get("kernels", [])
    meta["model_sources"] = meta["model_sources"] + variant.get("models", [])
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}/{kernel}.ipynb: {len(cells)} cells, {len(variant['patches'])} patches, self-test cell {test}, "
          f"run cell {run}, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    for name in sys.argv[1:] or VARIANTS:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
        build_variant(name)
