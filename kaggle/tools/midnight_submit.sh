#!/bin/zsh
# usage: midnight_submit.sh kernel [kernel ...]   wait for 00:00:30 UTC, submit each omerfarukmerey/<kernel> v1, then wait for scores
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
typeset -A MSG
MSG=(ntl-v28-ctrl "exp008 control: lightningv08 V28 (0.642) unmodified except mount paths"
     ntl-v28-helper15 "exp008: V28 + Kirill spectral helper 15% (was 10%)"
     ntl-v28-qwen-15 "exp008: V28 + Qwen3.8-27B share 15% (was 10%)"
     ntl-v28-helper-125 "exp008: V28 + Kirill spectral helper 12.5% (was 10%)"
     ntl-v28-dual "exp008: V28 + dual-head Qwen3.8-27B (Roman Tamrazov), Qwen 10%"
     ntl-v28-rare-helper15 "exp008: V28 + Kirill rare-class SVC (rare1516) + helper 15%")
target=$(date -u -j -f "%Y-%m-%d %H:%M:%S" "2026-10-26 00:00:30" +%s)
now=$(date -u +%s); [ $now -lt $target ] && sleep $((target-now))
for k in "$@"; do
  echo "$(date -u +%H:%M:%S) submit $k: $($K competitions submit name-that-llm -k omerfarukmerey/$k -v 1 -f submission.parquet -m "${MSG[$k]}" 2>&1 | grep -v Warning | tail -2)"
done
$K competitions submissions name-that-llm --csv 2>/dev/null | head -8
n=$#
while true; do
  out=$($K competitions submissions name-that-llm --csv 2>/dev/null | grep -v Warning | sed -n "2,$((n+1))p")
  echo "$out" | grep -q PENDING || { echo "$(date -u +%H:%M) SCORED:"; echo "$out" | cut -d, -f1,4-; break; }
  sleep 300
done
