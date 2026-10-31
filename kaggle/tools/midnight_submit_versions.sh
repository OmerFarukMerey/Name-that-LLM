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
MSG=(ntl-v31ac-a20-sk30-osvc12 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 12 (zbigniewkirchner's 0.664 change)"
     ntl-v31ac-a20-sk30-osvc15 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 1.5"
     ntl-v31ac-a20-sk30-osvc20 "exp030: $SK3 + SVC on crtproduction's open pool, svc_w 20"
     ntl-v31ac-a20-sk30-ax4 "exp029: $SK3 + our retrained ATLAS-8B (official + 500 open-pool passages per label), mixed rows x ATLAS^4"
     ntl-v31ac-a20-sk30-typo12 "exp031: $SK3 + typography evidence x1.2 (judith007's 0.664 change)"
     ntl-v31ac-a20-sk30-osvc12-ax4-typo12 "exp029-031: $SK3 + open-pool SVC w12 + retrained ATLAS x ATLAS^4 + typography x1.2"
     ntl-v31ac-a20-sk30-osvc12-ax4 "exp029-030: $SK3 + open-pool SVC w12 + retrained ATLAS x ATLAS^4"
     ntl-v31ac-a20-sk20 "exp028: $V31 + ATLAS-8B 20%, Sinkhorn 0.20 (0.30 scored 0.664)"
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
