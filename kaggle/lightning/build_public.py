"""Build the public version of one scored ntl-06xx-svc-tiers kernel, ready for `kaggle kernels push`.

    python kaggle/lightning/build_public.py BASE=v28 V25=<score of #6> V28=<score of #7>

V25 and V28 are the public LB scores of our two forks (INDEX #6 on upstream version 25, #7 on version 28). BASE picks
the one to publish. The output keeps that kernel's id, so the new version shows the score its version 1 already earned;
the code cells are exactly what was scored, with public_intro.md prepended as a markdown cell. Publish only if that
score beats the best public notebook in the Code tab, and make omerfarukmerey/ntl-svc627-quality-tiers public first,
otherwise nobody else can run the notebook.
"""

import json
import re
import sys
from pathlib import Path

import build

HERE = Path(__file__).resolve().parent
OUT = HERE / "public"
UPSTREAM = {"v25": ("0.641", "25", "5%"), "v28": ("0.642", "28", "10%")}  # base: (its public LB, version, Qwen share)
LB_TABLE = """## Leaderboard

Public LB of the hidden rerun. Each row is one base, run unmodified and with this change, all from my own submissions
except the two lightningv08 baselines, which are his listed scores:

| base | unmodified | + this change |
|---|---:|---:|
| [kirilltym's 0.639 stack](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) (no Qwen-27B) | 0.639 | 0.640 |
| lightningv08 version 25 (Qwen 5%){MARK25} | 0.641 | {V25} |
| lightningv08 version 28 (Qwen 10%){MARK28} | 0.642 | {V28} |

The SVC is the same member, with the same weighting, in all three ensembles. A 0.001 step is within the noise of the
public split, so read these rows as a check that the change does no harm; the CV above is the evidence for it."""


def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main(args):
    opts = dict(a.split("=", 1) for a in args)
    if set(opts) != {"BASE", "V25", "V28"} or opts["BASE"] not in UPSTREAM:
        raise SystemExit("usage: build_public.py BASE=v25|v28 V25=<score> V28=<score>")
    base = opts["BASE"]
    upstream, version, share = UPSTREAM[base]
    score = opts[base.upper()]
    kernel = build.BASES[base][2]
    build.main(base)
    nb = json.loads((HERE / kernel / f"{kernel}.ipynb").read_text(encoding="utf-8"))

    cells = {b: (f"**{opts[b.upper()]}**", ", **this notebook**") if b == base else (opts[b.upper()], "")
             for b in UPSTREAM}
    table = LB_TABLE.format(V25=cells["v25"][0], MARK25=cells["v25"][1], V28=cells["v28"][0], MARK28=cells["v28"][1])
    intro = (HERE / "public_intro.md").read_text(encoding="utf-8").replace("{LB_TABLE}", table)
    for key, value in {"SCORE": score, "UPSTREAM": upstream, "UPSTREAM_VERSION": version, "QWEN_SHARE": share}.items():
        intro = intro.replace("{" + key + "}", value)
    if re.search(r"\{[A-Z0-9_]+\}", intro):
        raise SystemExit("unfilled placeholder in public_intro.md")
    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro})

    title = intro.splitlines()[0].lstrip("# ").strip()
    if len(title) > 50:
        raise SystemExit(f"Kaggle titles are at most 50 characters: {title!r}")
    meta = json.loads((HERE / kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update(title=title, is_private=False)
    out = OUT / kernel
    out.mkdir(parents=True, exist_ok=True)
    (out / meta["code_file"]).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"built {out / meta['code_file']}: {meta['id']} titled {title!r} (slug {slugify(title)})")


if __name__ == "__main__":
    main(sys.argv[1:])
