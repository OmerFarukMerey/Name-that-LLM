#!/bin/zsh
# usage: refresh_push_submit.sh "YYYY-MM-DD HH:MM:SS" kernel "message"
# Wait (wall clock) until the given UTC time; then, only while the weekly GPU quota has more than 2 h of FREE time left
# (gpu_quota.py; pay-to-scale is on, so never push without that margin), push kaggle/nlm/<kernel> as a normal commit,
# wait for it, submit the version it created if it COMPLETEd, and wait for the score.
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kglrun.py)
k=$2; msg=$3
[ -d kaggle/nlm/$k ] && [ -n "$msg" ] || { echo "usage: $0 'YYYY-MM-DD HH:MM:SS' kernel message"; exit 1; }
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s)
echo "$(date -u +%m-%d\ %H:%M:%S) waiting $(( target - $(date -u +%s) )) s for $1 UTC"
while [ $(date -u +%s) -lt $target ]; do sleep 30; done   # wall clock: a single long sleep stalls while the Mac sleeps
free_s() {   # free GPU seconds left this week, empty on a failed call
  $K /opt/anaconda3/bin/python -I kaggle/tools/gpu_quota.py 2>/dev/null | grep '^gpu_quota' | /opt/anaconda3/bin/python -I -c "
import ast, sys
q = ast.literal_eval(sys.stdin.read().split(' ', 1)[1])
used = float(q['timeUsed'].split('.')[0].rstrip('s')); total = float(q['totalTimeAllowed'].rstrip('s'))
print(int(total - used))"
}
for i in $(seq 1 36); do   # up to 6 h of waiting for the refresh to show
  f=$(free_s)
  echo "$(date -u +%H:%M:%S) free GPU seconds: ${f:-?}"
  [ -n "$f" ] && [ "$f" -gt 7200 ] && break
  [ $i = 36 ] && { echo "no free GPU quota after 6 h, nothing pushed"; exit 1; }
  for j in {1..10}; do sleep 60; done
done
r=$($K kaggle kernels push -p kaggle/nlm/$k 2>&1 | grep -v -i "warning\|token")
echo "$(date -u +%H:%M:%S) push $k: $r"
v=$(echo "$r" | sed -n 's/.*Kernel version \([0-9]*\) successfully pushed.*/\1/p')
[ -n "$v" ] || { echo "push failed, nothing submitted"; exit 1; }
while true; do
  for j in {1..5}; do sleep 60; done
  st=$($K kaggle kernels status omerfarukmerey/$k 2>/dev/null | grep -i status | tail -1)
  echo "$(date -u +%H:%M:%S) $st"
  echo "$st" | grep -q -i -E 'complete|error|cancel' && break
done
echo "$st" | grep -q COMPLETE || { echo "commit did not complete, nothing submitted"; exit 1; }
echo "$(date -u +%H:%M:%S) free GPU seconds after the commit: $(free_s)"
r=$($K kaggle competitions submit name-that-llm -k omerfarukmerey/$k -v $v -f submission.parquet -m "$msg" 2>&1 | grep -v -i "warning\|token" | tail -2)
echo "$(date -u +%H:%M:%S) submit $k v$v: $r"
echo "$r" | grep -q -i "error" && exit 1
while true; do   # an empty answer (failed call) is not a result: keep polling
  for j in {1..10}; do sleep 60; done
  out=$($K kaggle competitions submissions name-that-llm --csv 2>/dev/null | grep -v -i "warning\|token" | sed -n '2p')
  if [ -n "$out" ] && ! echo "$out" | grep -q -i pending; then echo "$(date -u +%m-%d\ %H:%M) SCORED: $out"; break; fi
done
