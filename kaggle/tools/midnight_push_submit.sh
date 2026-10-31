#!/bin/zsh
# usage: midnight_push_submit.sh "YYYY-MM-DD HH:MM:SS" kernel [kernel ...]
# wait until the given UTC time and until the weekly GPU quota has room, push each kaggle/nlm/<kernel> (retrying while
# Kaggle's 2-session GPU limit is reached), wait for the fast commits, submit each one that COMPLETEd at the version its
# push reported, then wait until all of them are scored
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
V31="mercurydust's 0.660 (V31 v2) + exp018 speed-up, Llama from alexchilton's public copy (sha256-identical), extra passages pinned to their 10-28 versions"
typeset -A MSG VER
MSG=(ntl-v31ac-a25 "exp028: $V31 + ATLAS-8B 25%"
     ntl-v31ac-a20-comp10 "exp028: $V31 + ATLAS-8B 20%, complementary stack share 0.10 (was 0.15)"
     ntl-v31ac-a20-comp22 "exp028: $V31 + ATLAS-8B 20%, complementary stack share 0.22 (was 0.15)"
     ntl-v31ac-a20-sk30 "exp028: $V31 + ATLAS-8B 20%, Sinkhorn strength 0.30 (was 0.45)"
     ntl-v31ac-a20-sk60 "exp028: $V31 + ATLAS-8B 20%, Sinkhorn strength 0.60 (was 0.45)"
     ntl-v31ac-a20 "exp028 control: $V31 + ATLAS-8B 20% (= #24's code)"
     ntl-v31ac-a30-m45 "exp028: $V31 + ATLAS-8B 30% on rows with top-2 margin < 0.45 (was 0.3)")
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); shift
now=$(date -u +%s); echo "waiting $((target-now)) s"
while [ $(date -u +%s) -lt $target ]; do sleep 30; done   # wall clock: a single long sleep stalls while the Mac sleeps
for k in "$@"; do [ -z "${MSG[$k]}" ] && { echo "no message for $k, stopping"; exit 1; }; done
for i in $(seq 1 30); do   # the weekly quota refreshes at 00:00 UTC; push only with more than 2 h left
  q=$($K quota 2>&1 | grep '^GPU')
  echo "$(date -u +%H:%M:%S) quota: $q"
  rem=$(echo "$q" | awk '{print $3}' | tr -d h)
  [ -n "$rem" ] && [ "$(echo "$rem > 2" | bc)" = 1 ] && break
  [ $i = 30 ] && { echo "no GPU quota after an hour, stopping"; exit 1; }
  sleep 120
done
pushed=()
for k in "$@"; do
  for i in $(seq 1 120); do
    r=$($K kernels push -p kaggle/nlm/$k 2>&1 | grep -v Warning)
    echo "$(date -u +%H:%M:%S) push $k: $r"
    if echo "$r" | grep -q "successfully pushed"; then
      VER[$k]=$(echo "$r" | sed -n 's/.*Kernel version \([0-9]*\) successfully pushed.*/\1/p'); pushed+=($k); break
    fi
    echo "$r" | grep -q "Maximum batch GPU session" || { echo "push of $k failed, skipped"; break; }
    sleep 60
  done
done
[ ${#pushed} = 0 ] && { echo "nothing pushed, stopping"; exit 1; }
$S/tools/wait_kernels.sh $pushed
n=0
for k in $pushed; do
  st=$($K kernels status omerfarukmerey/$k 2>&1 | grep -v Warning)
  if ! echo "$st" | grep -q COMPLETE || [ -z "${VER[$k]}" ]; then echo "not submitted: $k v${VER[$k]}: $st"; continue; fi
  echo "$(date -u +%H:%M:%S) submit $k v${VER[$k]}: $($K competitions submit name-that-llm -k omerfarukmerey/$k -v ${VER[$k]} -f submission.parquet -m "${MSG[$k]}" 2>&1 | grep -v Warning | tail -2)"
  n=$((n+1))
done
[ $n = 0 ] && { echo "nothing submitted"; exit 1; }
while true; do   # an empty answer (failed call) is not a result: keep polling
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$((n+1))p")
  if [ -n "$out" ] && ! echo "$out" | grep -q PENDING; then echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,2,4-; break; fi
  sleep 300
done
