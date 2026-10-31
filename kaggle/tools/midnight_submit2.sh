#!/bin/zsh
# usage: midnight_submit2.sh "YYYY-MM-DD HH:MM:SS" kernel:version [kernel:version ...]
# wait until the given UTC time, submit each omerfarukmerey/<kernel> at <version>, then wait until all are scored
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
typeset -A MSG
MSG=(ntl-v28-ctrl "exp010 control v2: lightningv08 V28 (0.642) unmodified, on upstream's 2026-10-26 inputs"
     ntl-v28-mistral-5 "ntl-v28-mistral-5: V28 + our full-fit Mistral-7B LoRA member at 5% of the prior (exp010)"
     ntl-v28-qwen-15 "exp008: V28 + Qwen3.8-27B share 15% (was 10%), V28's inputs"
     ntl-v28-qwen-20 "exp008: V28 + Qwen3.8-27B share 20% (was 10%), upstream's 2026-10-26 inputs")
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "$1" +%s); shift
now=$(date -u +%s); echo "sleeping $((target-now)) s"; [ $now -lt $target ] && sleep $((target-now))
for kv in "$@"; do
  k=${kv%%:*}; v=${kv##*:}
  echo "$(date -u +%H:%M:%S) submit $k v$v: $($K competitions submit name-that-llm -k omerfarukmerey/$k -v $v -f submission.parquet -m "${MSG[$k]}" 2>&1 | grep -v Warning | tail -2)"
done
n=$#
while true; do
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$((n+1))p")
  echo "$out" | grep -q PENDING || { echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,4-; break; }
  sleep 300
done
