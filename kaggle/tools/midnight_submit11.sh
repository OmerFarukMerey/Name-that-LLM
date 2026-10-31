#!/bin/zsh
# usage: midnight_submit11.sh "YYYY-MM-DD HH:MM:SS" kernel:version [kernel:version ...]
# wait until the given UTC time, submit each omerfarukmerey/<kernel> at <version>, then wait until all are scored
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
V31="mercurydust's 0.660 (V31 v2) + exp018 speed-up, Llama from alexchilton's public copy (sha256-identical), extra passages pinned to their 10-28 versions"
typeset -A MSG
MSG=(ntl-v31ac-a20-gsq10 "exp027: $V31 + ATLAS-8B 20% + lightningv08's Qwen3.8-27B GSQ3 readout at 10% after ATLAS"
     ntl-v31ac-a30-gsq10 "exp027: $V31 + ATLAS-8B 30% + lightningv08's Qwen3.8-27B GSQ3 readout at 10% after ATLAS"
     ntl-v31ac-a20-gsq20 "exp027: $V31 + ATLAS-8B 20% + lightningv08's Qwen3.8-27B GSQ3 readout at 20% after ATLAS"
     ntl-v31ac-a20 "exp027 control: $V31 + ATLAS-8B 20% (= #24 with the new Llama source)"
     ntl-v31ac-a40 "exp027: $V31 + ATLAS-8B 40%"
     ntl-v31ac-a30-m45 "exp027: $V31 + ATLAS-8B 30% on rows with top-2 margin < 0.45 (was 0.3)")
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); shift
now=$(date -u +%s); echo "sleeping $((target-now)) s"; [ $now -lt $target ] && sleep $((target-now))
for kv in "$@"; do
  k=${kv%%:*}; v=${kv##*:}
  [ -z "${MSG[$k]}" ] && { echo "no message for $k, skipped"; continue; }
  echo "$(date -u +%H:%M:%S) submit $k v$v: $($K competitions submit name-that-llm -k omerfarukmerey/$k -v $v -f submission.parquet -m "${MSG[$k]}" 2>&1 | grep -v Warning | tail -2)"
done
n=$#
while true; do
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$((n+1))p")
  echo "$out" | grep -q PENDING || { echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,4-; break; }
  sleep 300
done
