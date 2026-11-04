#!/bin/zsh
# usage: midnight_submit_versions.sh "YYYY-MM-DD HH:MM:SS" kernel:version [kernel:version ...]
# wait (wall clock) until the given UTC time, submit each already committed kaggle/nlm/<kernel> at its version (only if
# the kernel's latest run is COMPLETE), then wait until all of them are scored
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
V31="mercurydust's 0.660 (V31 v2) + exp018 speed-up, Llama from alexchilton's public copy (sha256-identical), extra passages pinned to their 10-28 versions"
AX="ATLAS-8B 20% with our retrained adapters (official rows + 500 open-pool passages per label)"
typeset -A MSG
SK3="$V31 + ATLAS-8B 20%, Sinkhorn 0.30 (#34's 0.664 code)"
V31P30="${V31/pinned to their 10-28 versions/pinned to their 10-30 batches (judith007's rows)}"
KX="exp034: $V31P30 + ATLAS-8B 20%, Sinkhorn 0.30 + open-pool SVC w12"
MSG=(ntl-v31ac-a20-sk30-osvc12-xm2-p30 "$KX + external rows 2x drawn by match_weight (areebbb's K, on our code)"
     ntl-v31ac-a20-sk30-osvc12-xm1-cx-p30 "$KX + external rows 1x by match_weight (areebbb's scored 0.666 = J) + stack extra passages + open pool's 12+-model prompts"
     ntl-v31ac-a20-sk30-osvc12-xm1-cx-sib08-p30 "$KX + external rows 1x by match_weight (J) + stack open-pool prompts + V40A sibling-style residual at 0.08"
     ntl-v31ac-a20-sk30-osvc12-xm2-cx-sib08-p30 "$KX + external rows 2x by match_weight (K) + stack open-pool prompts + V40A sibling-style residual at 0.08"
     ntl-v31ac-a20-sk30-osvc12-xm1-cx-g37-p30 "exp042: $KX + external rows 1x by match_weight (J) + stack open-pool prompts + gemini-3.7-flash +0.3 after balancing"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-p30 "$KX + external rows 1x by match_weight (J) + Sovereign Gold Apex's V40A + V34.1 pairwise steps (the two public 0.666 lines)"
     ntl-v31ac-a20-sk30-osvc12-xm2-sov-p30 "$KX + external rows 2x by match_weight (K) + Sovereign Gold Apex's V40A + V34.1 pairwise steps"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-p30 "$KX + J + Sovereign's steps (= zbigniewkirchner's public 0.667) + stack open-pool prompts"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-g37-p30 "exp042: $KX + J + Sovereign's steps (= public 0.667) + gemini-3.7-flash +0.3 after balancing"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-g37-p30 "exp042: $KX + J + Sovereign's steps (= public 0.667) + stack open-pool prompts + gemini-3.7-flash +0.3"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-v35-pp13-p30 "$KX + J + Sovereign's steps (= public 0.667) + mohammedafreedpasha's V35 pairwise verifier (refs 8, top-4, weight 0.12), open pool pinned to Kaggle version 13"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-v35-pp13-p30 "$KX + J + Sovereign's steps + stack open-pool prompts (= our public 0.668) + mohammedafreedpasha's V35 pairwise verifier, open pool pinned to Kaggle version 13"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-kx-pp13-p30 "exp039: $KX + J + Sovereign's steps + stack open-pool prompts (= our public 0.668) + kilupy's 18-model corpus at TF-IDF 0.25, open pool pinned to Kaggle version 13"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-mw-pp13-p30 "exp046: $KX + J + Sovereign's steps + stack open-pool prompts (= our public 0.668), the stack's open-pool rows weighted by match_weight, open pool pinned to Kaggle version 13"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-kc-pp13-p30 "exp039: $KX + J + Sovereign's steps + stack open-pool prompts (= our public 0.668) + kilupy's Claude passages in the stack's Claude gate, open pool pinned to Kaggle version 13"
     ntl-v31ac-a20-sk30-osvc12-xm1-sov-cx-v14-p30 "$KX + J + Sovereign's steps + stack open-pool prompts (= our public 0.668) on open pool Kaggle version 14 (what a new copy reads)"
     ntl-v31ac-a20-sk30-osvc12-xm3-p30 "$KX + external rows 3x drawn by match_weight (K: 2x)"
     ntl-v31ac-a20-sk30-osvc12-xm4-p30 "$KX + external rows 4x drawn by match_weight (K: 2x)"
     ntl-v31ac-a20-sk60-osvc12-xm2-typo12-p30 "${KX/Sinkhorn 0.30/Sinkhorn 0.60} + external rows 2x by match_weight + typography x1.2 (K + qasminumber v5's knobs)"
     ntl-v31ac-a20-sk30-osvc12-xm2-ew10-p30 "$KX + external rows 2x by match_weight at --extra_w 1.0 (K: 0.5)"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf50-p30 "$KX + external rows 2x by match_weight + TF-IDF weight 0.5 (K: 0.25)"
     ntl-v31ac-a20-sk30-osvc12-xm4-tf50-p30 "$KX + external rows 4x by match_weight + TF-IDF weight 0.5"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf75-p30 "$KX + external rows 2x by match_weight + TF-IDF weight 0.75 (exp034 replica's best)"
     ntl-v31ac-a20-sk30-osvc12-xm4-tf75-p30 "$KX + external rows 4x by match_weight + TF-IDF weight 0.75"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf100-p30 "$KX + external rows 2x by match_weight + TF-IDF weight 1.0"
     ntl-v31ac-a20-sk30-osvc12-xm4-tf100-p30 "$KX + external rows 4x by match_weight + TF-IDF weight 1.0"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf75-cx-p30 "$KX + external rows 2x by match_weight + TF-IDF 0.75 + stack extra passages + open pool's 12+-model prompts"
     ntl-v31ac-a20-sk30-osvc12-xm4-tf75-cx-p30 "$KX + external rows 4x by match_weight + TF-IDF 0.75 + stack extra passages + open pool's 12+-model prompts"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf150-kx-p30 "exp039: $KX + external rows 2x by match_weight + kilupy's 18-model corpus (CC BY 4.0) + TF-IDF 1.5"
     ntl-v31ac-a20-sk30-osvc12-xm2-tf50-cx-p30 "$KX + external rows 2x by match_weight + TF-IDF 0.5 + stack extra passages + open pool's 12+-model prompts"
     ntl-v31ac-a20-sk30-osvc12-xm2-cx-p30 "$KX + external rows 2x by match_weight + stack extra passages + open pool's 12+-model prompts (exp035)"
     ntl-v31ac-a20-sk30-osvc12-xm4-cx-p30 "$KX + external rows 4x by match_weight + stack extra passages + open pool's 12+-model prompts (exp035)"
     ntl-v31ac-a20-sk30-osvc12 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 12 (zbigniewkirchner's 0.664 change)"
     ntl-v31ac-a20-sk30-osvc15 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 1.5"
     ntl-v31ac-a20-sk30-osvc20 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 20"
     ntl-v31ac-a20-sk30-ax4 "exp029: $SK3 + our retrained ATLAS-8B (official + 500 open-pool passages per label), mixed rows x ATLAS^4"
     ntl-v31ac-a20-sk30-typo12 "exp031: $SK3 + typography evidence x1.2 (judith007's 0.664 change)"
     ntl-v31ac-a20-sk30-osvc12-ax4-typo12 "exp029-031: $SK3 + open-pool SVC w12 + retrained ATLAS x ATLAS^4 + typography x1.2"
     ntl-v31ac-a20-sk30-osvc12-ax4 "exp029-030: $SK3 + open-pool SVC w12 + retrained ATLAS x ATLAS^4"
     ntl-v31ac-a20-sk20 "exp028: $V31 + ATLAS-8B 20%, Sinkhorn 0.20 (0.30 scored 0.664)"
     ntl-v31ac-a20-sk30-p30 "exp033: $V31P30 + ATLAS-8B 20%, Sinkhorn 0.30 (#34's 0.664 code on the 10,743 rows judith007's 0.664 read instead of 10,325)"
     ntl-v31ac-a20-p30 "exp033: $V31P30 + ATLAS-8B 20% (Sinkhorn 0.45)"
     ntl-v31ac-a20-sk60-p30 "exp033: $V31P30 + ATLAS-8B 20%, Sinkhorn 0.60 (judith007's 0.664 without x1.2; ours on the 10-28 rows 0.658)"
     ntl-v31ac-a20-sk60-typo12 "exp033: $V31 + ATLAS-8B 20%, Sinkhorn 0.60 + typography evidence x1.2 (judith007's 0.664 on the 10-28 rows)"
     ntl-v31ac-a20-typo12 "exp033: $V31 + ATLAS-8B 20% + typography evidence x1.2 (Sinkhorn 0.45)"
     ntl-v31ac-a20-sk10 "exp028: $V31 + ATLAS-8B 20%, Sinkhorn 0.10 (0.60 0.658, 0.45 0.663, 0.30 0.664)"
     ntl-v31ac-a20-sk30-typo10 "exp031: $SK3 + typography evidence x1.0 (judith007's x1.2, one step further)"
     ntl-v31ac-a20-sk20-typo12 "exp028/031: $V31 + ATLAS-8B 20%, Sinkhorn 0.20 + typography evidence x1.2 (judith007's change)"
     ntl-v31ac-a20-atlasx-p2 "exp029: $V31 + $AX, mixed rows x ATLAS^2"
     ntl-v31ac-a20-atlasx-p4 "exp029: $V31 + $AX, mixed rows x ATLAS^4"
     ntl-v31ac-a20-atlasx-p6 "exp029: $V31 + $AX, mixed rows x ATLAS^6"
     ntl-v31ac-a20-atlasx "exp029: $V31 + $AX (linear mix only)"
     ntl-v31ac-a20 "control: $V31 + ATLAS-8B 20% (= #24's code)")
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); shift
for kv in "$@"; do [ -z "${MSG[${kv%%:*}]}" ] && { echo "no message for ${kv%%:*}, stopping"; exit 1; }; done
now=$(date -u +%s); echo "waiting $((target-now)) s"
while [ $(date -u +%s) -lt $target ]; do sleep 30; done   # wall clock: a single long sleep stalls while the Mac sleeps
n=0
for kv in "$@"; do
  k=${kv%%:*}; v=${kv##*:}
  for i in 1 2 3 4 5 6 7 8 9 10; do   # a status that names no state (network down) is retried
    st=$($K kernels status omerfarukmerey/$k 2>&1 | grep -v Warning)
    echo "$st" | grep -q "KernelWorkerStatus" && break
    sleep 60
  done
  if ! echo "$st" | grep -q COMPLETE; then echo "not submitted: $k v$v: $(echo "$st" | tail -1 | cut -c1-160)"; continue; fi
  for i in 1 2 3; do
    r=$($K competitions submit name-that-llm -k omerfarukmerey/$k -v $v -f submission.parquet -m "${MSG[$k]}" 2>&1 | grep -v Warning | tail -2)
    echo "$(date -u +%H:%M:%S) submit $k v$v: $r"
    echo "$r" | grep -qi "remaining\|success" && { n=$((n+1)); break; }
    sleep 60
  done
done
[ $n = 0 ] && { echo "nothing submitted"; exit 1; }
while true; do   # an empty answer (failed call) is not a result: keep polling
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$((n+1))p")
  if [ -n "$out" ] && ! echo "$out" | grep -q PENDING; then echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,2,4-; break; fi
  sleep 300
done
