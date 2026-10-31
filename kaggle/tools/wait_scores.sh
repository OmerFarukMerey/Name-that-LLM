#!/bin/zsh
# usage: wait_scores.sh N   poll until none of the latest N submissions is PENDING, then print them
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
while true; do
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$(($1+1))p")
  echo "$out" | grep -q PENDING || { echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,4-; break; }
  sleep 300
done
