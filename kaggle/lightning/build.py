"""Build our fork of lightningv08's "LoRA + Llama + Qwen3.8-27B" notebook plus our SVC change.

    python kaggle/lightning/build.py [v25|v28]      (default v25)

Two scored upstream versions are kept byte-for-byte; each builds its own private kernel:
  upstream_v25.ipynb  version 25, public 0.641: Qwen 5% + Kirill helper 10%   -> ntl-0641-svc-tiers
  upstream_v28.ipynb  version 28, public 0.642: Qwen 10% + Kirill helper 10%  -> ntl-0642-svc-tiers
V28 differs from V25 only in the Qwen share and the recipe guard that checks it. Do not refresh either from the
API: `kernels pull` returns whatever the latest version is.

The only functional change is the main n-gram SVC (run_fulltext_selected_ensemble.py, `svc_scores=...`).
It reads our quality-tiered refit of ntl627-svc-prefit, omerfarukmerey/ntl-svc627-quality-tiers: same
vocabulary and idf, with train rows at weight 1.0, pool_v4 rows that are also in fleongg's quality pool at 0.3
and the other pool rows at 0.1. Its log-probability enters the blend x1.64 instead of x1.0. The readout is
unscaled, so the x1.64 is applied once, in code. This is the lever from experiments/exp004-exp006 and
kaggle/stack, where it moved kirilltym's stack from 0.639 to 0.640.

Our changes are:
  1. REWRITES: a kernel pushed through the API mounts every dataset at /kaggle/input/datasets/<owner>/<slug>
     (probe kernel omerfarukmerey/ntl-mount-probe). Upstream hard-codes legacy short mounts for two datasets, and
     /kaggle/input is read-only, so the paths are rewritten. Each rewrite must hit its exact upstream count.
  2. PATCHES: exact substitutions inside upstream code cells. Each must match exactly once.
  3. INSERTS: a commit-time self-test (selftest_cell.py) and, just before it, a %%writefile cell with its child
     script (selftest_child.py), which runs the notebook's own cached_svc in a separate process so the notebook
     process never holds sklearn or the readout during the hidden rerun. On the 3-row public test upstream skips
     all inference, so without the self-test nothing we change would run before the hidden rerun. The self-test
     checks the recipe guard of its base, so each base swaps those strings (BASES).
The build writes kaggle/lightning/<kernel>/: the notebook and its kernel-metadata.json (kernel-metadata.base.json
plus the kernel's id, title and code file). Any mismatch fails the build loudly. Run prepush_check.py <kernel>
right before `kaggle kernels push -p kaggle/lightning/<kernel>`: the push attaches each source's current version,
and that script fails if any input moved past the versions upstream scored with.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASES = {  # base: (upstream notebook, its sha256, our kernel, (old, new) swaps in the self-test for its recipe guard)
    "v25": ("upstream_v25.ipynb", "08ffbe8f3c26601c3aec1995158c3959897bf3d7543b253d04af448c2bc03448",
            "ntl-0641-svc-tiers", []),
    "v28": ("upstream_v28.ipynb", "ac3ad0b8bb54d219cbbb5e1b90bac4d2bfb167b725ecf213f4396a73c22f4ac9",
            "ntl-0642-svc-tiers", [("QWEN05_KIRILL10", "QWEN10_KIRILL10"), ("'share != .05'", "'share != 0.10'"),
                                   ("Qwen5%", "Qwen10%")]),
}

REWRITES = [  # (old, new, occurrences in upstream), applied to every cell as plain substring replacements
    ("/kaggle/input/ntl-olmo2-native24-final-inference-assets",
     "/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets", 145),
    ("/kaggle/input/nlm-l24-adapters", "/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters", 2),
]
SVC_ROOT = """
# ---- ntl-0641-svc-tiers (kaggle/lightning/build.py): the main n-gram SVC reads our quality-tiered refit of
# ntl627-svc-prefit (same vocabulary/idf, unscaled logits); run_fulltext_selected_ensemble.py weights it x1.64.
_svc_tiers_roots = [root for root in ['/kaggle/input/datasets/omerfarukmerey/ntl-svc627-quality-tiers',
                                      '/kaggle/input/ntl-svc627-quality-tiers'] if (Path(root) / 'readout.npz').is_file()]
assert len(_svc_tiers_roots) == 1, ('Attach omerfarukmerey/ntl-svc627-quality-tiers exactly once', _svc_tiers_roots)
SVC_TIERS_ROOT = PATHS['svc_prefit'] = _svc_tiers_roots[0]
del _svc_tiers_roots
"""
PATCHES = [  # (old, new), applied after REWRITES
    ("/night24_readout_banks/COMPLETE.json'}\n\nRECIPE = {",
     "/night24_readout_banks/COMPLETE.json'}\n" + SVC_ROOT + "\nRECIPE = {"),
    (" {'source': 'v17.paths.svc_prefit',\n"
     "  'path': '/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit'},\n",
     " {'source': 'v17.paths.svc_prefit',\n"
     "  'path': SVC_TIERS_ROOT},\n"),
    ("    total=clipped_logp(svc_scores)\n", "    total=1.64*clipped_logp(svc_scores)\n"),
]
SELFTEST_CHILD = "/kaggle/temp/ntl_svc_tiers_selftest_child.py"  # outside RUNTIME, so RUNTIME stays upstream's
INSERTS = [  # (anchor text in an upstream code cell, "before" / "after", [(cell header, our cell file), ...])
    ("ULTRA_ROOT = Path('/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl')", "after",
     [("%%writefile " + SELFTEST_CHILD + "\n", "selftest_child.py"), ("", "selftest_cell.py")]),
]


def code_cell(source):  # same key order and str source as upstream's cells
    return {"cell_type": "code", "source": source, "metadata": {}, "outputs": [], "execution_count": None}


def count(cells, text):
    return sum(c["source"].count(text) for c in cells)


def find_cell(cells, anchor):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and anchor in c["source"]]
    if len(hits) != 1:
        raise SystemExit(f"anchor matched {len(hits)} cells: {anchor!r}")
    return hits[0]


def main(base="v25"):
    if base not in BASES:
        raise SystemExit(f"usage: build.py [{'|'.join(BASES)}]")
    upstream, upstream_sha256, kernel, swaps = BASES[base]
    raw = (HERE / upstream).read_bytes()
    if hashlib.sha256(raw).hexdigest() != upstream_sha256:
        raise SystemExit(f"{upstream} is not the verified {base} (sha256 {hashlib.sha256(raw).hexdigest()})")
    nb = json.loads(raw)
    cells = nb["cells"]
    if not all(isinstance(c["source"], str) for c in cells):
        raise SystemExit("expected str cell sources, as in upstream")

    for old, new, expected in REWRITES:
        found, already = count(cells, old), count(cells, new)
        if found != expected or already:
            raise SystemExit(f"rewrite {old!r}: {found} hits (expected {expected}), {already} already rewritten")
        for c in cells:
            c["source"] = c["source"].replace(old, new)
    code = [c for c in cells if c["cell_type"] == "code"]
    for old, new in PATCHES:
        if count(code, old) != 1:
            raise SystemExit(f"patch matched {count(code, old)} times: {old!r}")
        i = find_cell(cells, old)
        cells[i]["source"] = cells[i]["source"].replace(old, new)
    for anchor, where, parts in INSERTS:
        i = find_cell(cells, anchor) + (where == "after")
        for header, name in parts:
            source = (HERE / name).read_text(encoding="utf-8").rstrip("\n")
            if name == "selftest_cell.py":
                for old, new in swaps:
                    if source.count(old) != 1:
                        raise SystemExit(f"self-test swap matched {source.count(old)} times: {old!r}")
                    source = source.replace(old, new)
            cells.insert(i, code_cell(header + source))
            i += 1

    test = find_cell(cells, "def _ntl_svc_tiers_selftest():")
    child = find_cell(cells, "%%writefile " + SELFTEST_CHILD + "\n")
    writefiles = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and c["source"].startswith("%%writefile")]
    run = find_cell(cells, "if not preview:\n    import subprocess, traceback")
    if not (child == max(writefiles) == test - 1 and test < run):
        raise SystemExit(f"self-test cell {test} must directly follow its child script {child}, the last %%writefile "
                         f"({max(writefiles)}), and precede the run cell {run}")
    if cells[test]["source"].count(f"Path('{SELFTEST_CHILD}')") != 1:
        raise SystemExit(f"the self-test cell must run {SELFTEST_CHILD} exactly once")

    out = HERE / kernel
    out.mkdir(exist_ok=True)
    text = json.dumps(nb, indent=1, ensure_ascii=False)  # upstream's own layout
    (out / f"{kernel}.ipynb").write_text(text, encoding="utf-8")
    meta = {"id": f"omerfarukmerey/{kernel}", "title": kernel, "code_file": f"{kernel}.ipynb",
            **json.loads((HERE / "kernel-metadata.base.json").read_text(encoding="utf-8"))}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {kernel}/{kernel}.ipynb from {upstream}: {len(cells)} cells, sha256 "
          f"{hashlib.sha256(text.encode()).hexdigest()[:16]}")
    for old, new, expected in REWRITES:
        print(f"  rewrite x{expected}: {old} -> {new}")
    print(f"  {len(PATCHES)} patches: svc_prefit -> SVC_TIERS_ROOT (cell 4 + its input check), total=1.64*clipped_logp")
    print(f"  self-test child script cell {child} ({SELFTEST_CHILD}), self-test cell {test}, run cell {run}")


if __name__ == "__main__":
    main(*sys.argv[1:])
