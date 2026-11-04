"""exp033: which extra passages judith007's public 0.664 read. V31's load_extra_passages (prompts answered by >= 12
labels, 30-300 words, one passage per prompt and label) emulated on the five passage files for the batch whitelists of
2026-10-28 (our pin), 10-29 (two public reruns of our a20 at 0.662), 10-30, 10-31 (judith007's push) and today. The
first count must be the 10,325 rows our commit logs print.

    python -I experiments/exp033_extra_rows.py <dl_dir> data/raw/train.csv

<dl_dir> holds the current files, fetched with kaggle/tools/ds_download.py: gen/ntl_generations.csv
(abdullah1afify/ntl-generations), lucas/gen_passages.csv (lucasboesen/ntl-free-proxy-generations), prv/passages_clean.csv
(prvsiyan/prvsiyans-extended-name-that-llm-dataset), seyit/ntl_extra_passages_claude.csv
(seyitkaangunes/ntl-extra-api-passages) and dop/ntl_proxy_generations.csv (dophuchung/ntl-proxy-generations). The
sources only append tagged batches, so each date's file is the current one filtered to that date's batches.
"""
import re, sys
from pathlib import Path
import pandas as pd
D, train = Path(sys.argv[1]), pd.read_csv(sys.argv[2])
LABELS = sorted(train["model"].unique())
files = {"ntl_generations.csv": (D / "gen/ntl_generations.csv", "label", "finish_reason"),
         "gen_passages.csv": (D / "lucas/gen_passages.csv", "model", None),
         "passages_clean.csv": (D / "prv/passages_clean.csv", "model", None),
         "ntl_extra_passages_claude.csv": (D / "seyit/ntl_extra_passages_claude.csv", "model", None),
         "ntl_proxy_generations.csv": (D / "dop/ntl_proxy_generations.csv", "label", None)}
order = ["ntl_extra_passages_claude.csv", "gen_passages.csv", "ntl_generations.csv", "ntl_proxy_generations.csv", "passages_clean.csv"]
raw = {k: pd.read_csv(p) for k, (p, _, _) in files.items()}
g = raw["ntl_generations.csv"]; print("ntl_generations batches:", g["batch"].astype(str).value_counts().sort_index().to_dict())
print("gen_passages batches:", raw["gen_passages.csv"]["batch"].astype(str).value_counts().sort_index().to_dict())
print("passages_clean dataset_version_added:", raw["passages_clean.csv"]["dataset_version_added"].value_counts().sort_index().to_dict())
G28 = {'b001', 'b001g', 'b003', 'b003a', 'd1028', 'pilot', 'pilot2', 'pilot3', 'pilotpro'}
L28 = {f'b{i}' for i in (1, 2, *range(5, 16))} | {f'v2b{i:02d}' for i in range(1, 19)}
SCEN = {"pinned 10-28": (G28, L28, {1, 2, 4}),
        "10-29 (d1029, v2b19-20, prv 5)": (G28 | {"d1029"}, L28 | {"v2b19", "v2b20"}, {1, 2, 4, 5}),
        "10-30 (+ d1030)": (G28 | {"d1029", "d1030"}, L28 | {"v2b19", "v2b20"}, {1, 2, 4, 5}),
        "judith007 10-31 (+ d1031)": (G28 | {"d1029", "d1030", "d1031"}, L28 | {"v2b19", "v2b20"}, {1, 2, 4, 5}),
        "now 11-02 (all)": (None, None, None)}


def load(gb, lb, pv):
    parts = []
    for name in order:
        d = raw[name].copy()
        _, label_col, finish_col = files[name]
        if name == "ntl_generations.csv" and gb is not None: d = d[d["batch"].astype(str).isin(gb)]
        if name == "gen_passages.csv" and lb is not None: d = d[d["batch"].astype(str).isin(lb)]
        if name == "passages_clean.csv" and pv is not None: d = d[pd.to_numeric(d["dataset_version_added"], errors="coerce").isin(pv)]
        if finish_col and finish_col in d: d = d[d[finish_col].eq("stop")]
        d = d.rename(columns={label_col: "model"})[["prompt_id", "model", "text"]].copy()
        d["prompt_id"] = ("pv_" if name == "passages_clean.csv" else "") + d["prompt_id"].astype(str)
        parts.append(d)
    x = pd.concat(parts, ignore_index=True)
    x = x[x["model"].isin(LABELS)].copy()
    x["text"] = x["text"].astype(str).map(lambda t: re.sub(r"\s+", " ", re.sub(r"^\*\*(.*)\*\*$", r"\1", t.strip())).strip())
    x = x[x["text"].str.split().str.len().between(30, 300)].drop_duplicates(subset=["model", "text"])
    x = x.drop_duplicates(subset=["prompt_id", "model"])
    return x[x.groupby("prompt_id")["model"].transform("nunique") >= 12].reset_index(drop=True)


base = None
for k, v in SCEN.items():
    x = load(*v)
    c = x["model"].value_counts().reindex(LABELS, fill_value=0)
    print(f"{k:34s} {len(x):6d} rows {x['prompt_id'].nunique():5d} prompts", "" if base is None else f"(+{len(x) - len(base)} rows vs pinned; per label min +{(c - bc).min()} max +{(c - bc).max()})")
    if base is None: base, bc = x, c
