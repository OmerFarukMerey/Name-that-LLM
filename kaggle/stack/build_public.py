"""Build the public version of ntl-stack into kaggle/public/, ready for `kaggle kernels push -p kaggle/public`.

    python kaggle/stack/build_public.py V4=0.639 V5=0.641

V4 and V5 are the public LB scores of ntl-stack v4 (kirilltym's version 1, unmodified) and v5 (ours). The notebook
is the v5 build with public_intro.md prepended as a markdown cell; the code cells are exactly what v5 ran.
Publish only if V5 beats the best public notebook in the Code tab, and never publish an unmodified fork.
"""

import json
import re
import sys
from pathlib import Path

import build

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "public"
V5_KNOBS = ["SVC_POOL_WEIGHT=0.3", "SVC_LOW_WEIGHT=0.1", "SVC_WEIGHT_SCALE=1.64"]
LB_TABLE = """## Leaderboard

Public LB of the hidden rerun, both from my own copies, submitted the same day:

| notebook | public LB |
|---|---:|
| kirilltym's version 1, unmodified | {V4} |
| **this notebook** (one change: the SVC's pool weights) | **{V5}** |

An earlier, accidental test points the same way. My first three runs used a restructured copy of the stack whose fusion
stages silently fell back to the part-two anchor (part two includes this SVC at weight 1.0). On that anchor alone the SVC
change went **0.620** (upstream weights) → **0.621** (all public rows 0.3, ×1.44) → **0.622** (this weighting), the
same order as the CV table above. Each step is small, but none went the wrong way."""


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main(args):
    scores = dict(a.split("=", 1) for a in args)
    if set(scores) != {"V4", "V5"}:
        raise SystemExit("usage: build_public.py V4=<score> V5=<score>")
    build.main(V5_KNOBS)
    nb = json.loads((HERE / "ntl-stack.ipynb").read_text())
    intro = (HERE / "public_intro.md").read_text()
    intro = intro.replace("{LB_TABLE}", LB_TABLE.format(**scores)).replace("{SCORE}", scores["V5"])
    if re.search(r"\{[A-Z0-9_]+\}", intro):
        raise SystemExit("unfilled placeholder in public_intro.md")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)})

    title = intro.splitlines()[0].lstrip("# ").strip()
    if len(title) > 50:
        raise SystemExit(f"Kaggle titles are at most 50 characters: {title!r}")
    meta = json.loads((HERE / "kernel-metadata.json").read_text())
    meta.update(id=f"omerfarukmerey/{slugify(title)}", title=title, code_file=f"{slugify(title)}.ipynb",
                is_private=False)
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.ipynb"):
        old.unlink()
    (OUT / meta["code_file"]).write_text(json.dumps(nb, indent=1))
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("built", OUT / meta["code_file"], "as", meta["id"])


if __name__ == "__main__":
    main(sys.argv[1:])
