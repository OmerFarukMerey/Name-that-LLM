# Public notebooks at 0.635+

Every notebook in the competition's Code tab with a public score of 0.635 or higher on 2026-10-25 (15:10 UTC), read at
the version that earned the score. Notebooks just below the cut were 0.632 (amanatar) and 0.631 (evgendvorkin, flexonafft,
avazaku). The source of each was pulled and diffed against its parent. Our variants of the 0.642 notebook are built by
[`kaggle/lightning/build_variants.py`](../kaggle/lightning/build_variants.py).

| Notebook | Author | Scored version | Public LB | Contribution |
|---|---|---|---:|---|
| [0.642 \| LoRA + Llama + Qwen3.8-27B](https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b) | Vladislav Lavrentev (lightningv08) | V28 | 0.642 | The base we fork. It rebuilds fleongg's 0.636 reference with cached readouts, adds Kirill's 10% spectral helper after ATLAS, and mixes in the author's own full-text Qwen3.8-27B NF4 classifier at 10% on every row before one Sinkhorn. |
| [Name That LLM \| Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) | Roman Tamrazov | V22 | 0.640 | First to use the Qwen3.8-27B checkpoint's unused prompt-centred head (an agreement-gated dual head), with streaming, deadline-guarded Qwen inference. His adaptive SVC topic correction is copied into V28's code, but V28 never calls it. |
| [0.640 \| kirilltym 0.639 + GLM/Kimi SVC pool patch](https://www.kaggle.com/code/jnliang/0-640-kirilltym-0-639-glm-kimi-svc-pool-patch) | JN Liang | V2 | 0.640 | A paired test showing that adding Daniel Andres Miranda's 1,516 GLM-5.3 / GLM-5.3-flash / Kimi-K3 passages to the n-gram SVC pool moved kirilltym's 0.639 to 0.640. |
| [Kaggle_name_the_LLM_v2](https://www.kaggle.com/code/maverickss26/kaggle-name-the-llm-v2) | maverick_ss_26 | V4 | 0.639 | A readable, single-pass restructuring of the 0.639 synthetic4404 stack that reproduces its predictions. It removes dead family-gate compute and exposes a within-Claude gate and mix knobs. |
| [TestNTL Part Two Complement](https://www.kaggle.com/code/mtoshidesu/testntl-part-two-complement) | m-toshi desu | V3 | 0.639 | Republished the 0.639 stack with memory-safe engineering: chunked LoRA merge, OOM batch halving, float32 topic distances and log-domain Sinkhorn. It confirms that 0.639 reproduces. |
| [0.639 \| Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) | Kirill Tymchishin | V1 | 0.639 | The Night24 spectral readout helper: 8 heads on the public LoRA24 features, fit on 2,699 official plus 4,404 synthetic passages and mixed at 10% after ATLAS. It lifted fleongg's 0.636 to 0.639, and V28 reuses it unchanged. |
| [ntl-public-0-639-fork](https://www.kaggle.com/code/arlearonreidma/ntl-public-0-639-fork) | Arle Aron Reidma | V1 | 0.639 | An independent, byte-identical rerun that reproduced kirilltym's 0.639. |
| [NTL v26 spectral 0639](https://www.kaggle.com/code/huangkx666/ntl-v26-spectral-0639) | Leo (huangkx666) | V1 | 0.639 | A credited verbatim copy of cosmorin's reproduction of the 0.639. It records the bank hash check and the 0.637 official-only control. |
| [LLM Decoder Qwen Ver.](https://www.kaggle.com/code/lynnsakurai/llm-decoder-qwen-ver) | Arlene (lynnsakurai) | V9 | 0.636 | Re-ran fleongg's Part Two Complement on the V28 Docker image and added an ATLAS time-budget guard and a LoRA24 loader fallback. |
| [NTL Part Two Complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement) | fle3n (fleongg) | V1 | 0.636 | The reference ensemble inside V28: Nishant Dahal's two-stage LoRA part two, plus a fixed 15% complementary frozen-feature stack and 10% ATLAS on low-margin rows, then one Sinkhorn. |
| [NTL v25 complement 0636](https://www.kaggle.com/code/huangkx666/ntl-v25-complement-0636) | Leo (huangkx666) | V1 | 0.636 | A credited verbatim replication that independently reproduced fleongg's 0.636. |
| [0.636 \| Reproducible LLM Fingerprinting](https://www.kaggle.com/code/avazaku/0-636-reproducible-llm-fingerprinting) | avazaku | V3 | 0.636 | Reproduced fleongg's stack with a fail-fast input and wheel preflight. It also published a useful negative result: a leave-one-out SVC peer correction without balancing scores 0.635 on the full stack (V2). |
| [ntl-public-0-636-fork](https://www.kaggle.com/code/arlearonreidma/ntl-public-0-636-fork) | Arle Aron Reidma | V4 | 0.636 | Re-ran lightningv08's half-row Qwen3.8-27B recipe (V20) with mount-path rewrites, reproducing 0.636 on a different Docker image. |

These notebooks also build on work by Nishant Dahal (two-stage LoRA part two), prvsiyan (Gaussian shared-prompt heads), Lucas Boesen (LoRA24 and token-attention heads), Travis Nguyen (in-style Qwen2.5 LoRA), Zbigniew Kirchner (Claude-head LoRA), pjmathematician (UltraBERT), Joseph Adamski (ATLAS), Dhruv Tibarewal (L24 adapters and wheels), Daniel Andres Miranda (GLM/Kimi passages, CC BY 4.0), seyitkaangunes (extra API passages and prompt bank) and mrpicklerick (decoder reference).

## Update, 2026-10-25 23:45 UTC

New entries at 0.642+ since the survey above (top public score now 0.651):

| Notebook | Author | Scored version | Public LB | Contribution |
|---|---|---|---:|---|
| [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) | Dhruv Tibarewal | V1 | **0.651** | A separate pipeline, not a V28 fork. It averages five fold adapters and a full-data adapter of two-stage LoRA24 heads on Qwen3-14B-Base, Qwen3-8B-Base and Llama-3.1-8B (plus Nishant Dahal's Qwen3-14B and Qwen2.5-14B checkpoints and a DeBERTa-v3-large). It adds shared-prompt Gaussian heads, centred twins, a 4-view n-gram SVC on train plus pool, fleongg's complementary stack at 15%, and ATLAS at 10% on rows with top-2 margin below 0.3, then Sinkhorn and typography guards. Llama-3.1 is a gated Kaggle model, so forking it needs Meta's licence accepted on the account. |
| [0.643 \| Same Stack, Sharper Blend — Enjoy](https://www.kaggle.com/code/areebbb/0-643-same-stack-sharper-blend-enjoy) | Areeb Ali | V1 | 0.643 | V28 with Qwen3.8-27B at 15% and Kirill's spectral heads at 15% (both were 10%). The re-blend cell first reproduces V28's own blend from its saved ingredients and falls back to it on any mismatch. It is the first scored two-share change of V28: +0.001. |
| [ntl-public-stack](https://www.kaggle.com/code/shakshamkarkiii/ntl-public-stack) | Shaksham Karkiii | V1 | 0.643 | The same 15% / 15% re-blend as Areeb Ali's notebook, with the credits removed. It confirms that 0.643 reproduces. |
| [0.642 \| Teach More, Keep the Signature](https://www.kaggle.com/code/kirilltym/0-642-teach-more-keep-the-signature) | Kirill Tymchishin | latest | 0.642 | Kirill's update that moves his helper onto V28 and adds a rare-class SVC (rare1516). It ties V28. |

## Update, 2026-10-27 08:15 UTC: forks of nlm-v27

The top public score is still 0.651. Every entry below changes nlm-v27; a displayed score can belong to an older
version, so each scored version was pulled and diffed against nlm-v27 V1.

| Notebook | Author | Scored version | Public LB | Change against nlm-v27 |
|---|---|---|---:|---|
| [0.651 Verified Reproduction](https://www.kaggle.com/code/avazaku/name-that-llm-0-651-verified-reproduction) | avazaku | V3 of the reproduction | **0.651** | none: the six code cells byte for byte, Llama-3.1-8B read from his public [Verified Llama Asset Cache](https://www.kaggle.com/code/avazaku/name-that-llm-verified-llama-asset-cache) (NousResearch copy). Its commit ran 4 h against nlm-v27's 1.5 h (a slow first read of the Llama files) |
| [Kaggle_name_the_LLM_v3](https://www.kaggle.com/code/maverickss26/kaggle-name-the-llm-v3) | maverickss26 | V8 | **0.651** | restructured, no weight changed; skips the zero-weight TF-IDF fit and the extra rows only it read |
| [Enhanced 0.651 recipe](https://www.kaggle.com/code/kozykappa/name-that-llm-enhanced-0-651-recipe) | kozykappa | V1 | **0.651** | an SVC topic correction in groups of 8-18 passages (alpha 0.62) |
| [Name That LLM Solution](https://www.kaggle.com/code/mohammedafreedpasha/name-that-llm-solution) | mohammedafreedpasha | V24 | **0.651** | none (a pip flag); its re-weighted V25 is unscored |
| [0.649 Q35 Restoration Audit](https://www.kaggle.com/code/avazaku/name-that-llm-0-649-q35-restoration-audit) | avazaku | (dated receipt) | 0.649 | restores the stack's Qwen3.5-9B token-attention member (dropped upstream): -0.002 |
| [NTL C9 public data adaptation](https://www.kaggle.com/code/jnliang/ntl-c9-public-data-adaptation-20261026) | jnliang | latest | 0.647 | a READER-adapted ModernBERT at weight 1.5; its notes also record Dhruv's own nlm-enc-v5 ModernBERT at 1.5 as **0.632** |
| [Name That LLM \| Who am I?](https://www.kaggle.com/code/romantamrazov/name-that-llm-who-am-i) | Roman Tamrazov | V24 | 0.645 | no Llama (Qwen2.5-7B in its place) plus a soft-Sinkhorn reassignment; the latest version (partial ATLAS) is unscored |

Dhruv Tibarewal's own [Name That LLM Fully Reproducible](https://www.kaggle.com/code/dhruvtibarewal/name-that-llm-fully-reproducible)
(nlm-v27 + his ModernBERT member, documented) shows no public score. Its notes put the hidden rerun's run-to-run noise
at about +-0.005 (identical notebooks scored 0.600 and 0.605). Our exp013 (ModernBERT + small fixes) scored 0.645.

## What the survey found

- Everything at 0.636-0.642 descends from one stack: Nishant Dahal's two-stage LoRA part two, fleongg's complement (0.636),
  Kirill's spectral helper (0.639), and lightningv08's full-text Qwen3.8-27B classifier (0.641-0.642). Most of the other
  entries are faithful reproductions, which shows that the scores reproduce across Docker images.
- lightningv08's own version history is the cleanest evidence in the field: Qwen share 5% -> 10% gave +0.001 twice, and
  adding Kirill's helper at 10% gave +0.005. No one has scored any other helper share.
- The two newest 0.640s are a change of the n-gram SVC pool (JN Liang, +0.001 on kirilltym) and a dual-head 27B
  (Roman Tamrazov, 0.640 against its 0.641 base, bundled with selective Qwen). Both are within noise.
