#!/bin/zsh
# usage: push_after.sh <log-file> <egrep-pattern> dir/kernel [dir/kernel ...]
# wait (wall clock) until the pattern appears in the log file, then push each kaggle/<dir>/<kernel> (retrying while
# Kaggle's 2-session GPU limit is reached) and wait until none of them is QUEUED/RUNNING
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
log=$1; pat=$2; shift 2
until grep -qE "$pat" "$log"; do sleep 30; done
echo "$(date -u +%H:%M:%S) found: $(grep -E "$pat" "$log" | tail -1 | cut -c1-120)"
names=()
for kd in "$@"; do
  k=${kd##*/}
  for i in $(seq 1 600); do
    r=$($K kernels push -p kaggle/$kd 2>&1 | grep -v Warning)
    echo "$(date -u +%H:%M:%S) push $k: $r"
    if echo "$r" | grep -q "successfully pushed"; then names+=($k); break; fi
    echo "$r" | grep -q "Maximum batch GPU session" || { echo "push of $k failed, skipped"; break; }
    sleep 60
  done
done
[ ${#names} = 0 ] && { echo "nothing pushed"; exit 1; }
$S/tools/wait_kernels.sh $names
