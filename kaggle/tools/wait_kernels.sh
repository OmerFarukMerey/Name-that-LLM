#!/bin/zsh
# usage: wait_kernels.sh kernel [kernel ...]   poll omerfarukmerey/<kernel> until none is QUEUED/RUNNING, then print statuses
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
while true; do
  st=""; pending=0
  for k in "$@"; do
    s=$(/opt/anaconda3/bin/python $S/tools/kgl.py kernels status omerfarukmerey/$k 2>&1 | grep -v Warning | sed 's/.*status //')
    st="$st\n$k $s"
    echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL" || pending=1   # running, queued, or no answer (network): keep waiting
  done
  [ $pending -eq 0 ] && { echo "$(date -u +%H:%M) final:$st"; break; }
  sleep 90
done
