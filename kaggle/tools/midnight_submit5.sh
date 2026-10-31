#!/bin/zsh
# usage: midnight_submit5.sh "YYYY-MM-DD HH:MM:SS" kernel:version [kernel:version ...]
# wait until the given UTC time, submit each omerfarukmerey/<kernel> at <version>, then wait until all are scored
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
K=(/opt/anaconda3/bin/python $S/tools/kgl.py)
typeset -A MSG
MSG=(ntl-nlm27-mn8-noatlas "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w1.0, ATLAS off"
     ntl-nlm27-mn8w2-noatlas "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w2.0, ATLAS off"
     ntl-nlm27-mn8w3-noatlas "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w3.0, ATLAS off"
     ntl-nlm27-mn8-deadline "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w1.0, ATLAS kept up to its deadline"
     ntl-nlm27-mn8w2-deadline "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w2.0, ATLAS kept up to its deadline"
     ntl-nlm27-mn8w3-deadline "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w3.0, ATLAS kept up to its deadline"
     ntl-nlm27-mn8q25-noatlas "exp016: nlm-v27 0.651 recipe (avazaku's Llama cache) + our Ministral-3-8B two-stage LoRA24 member mn8v1 w1.0 + Dhruv's Qwen2.5-14B q25_14v5 (full adapter) w1.0, ATLAS off"
     ntl-nlm27-mn8w2-dp "exp018: nlm-v27 0.651 recipe + our Ministral-3-8B member mn8v1 w2.0, ATLAS kept up to its deadline, 8B members on a copy per GPU + GPU pooling (~1 h faster)"
     ntl-nlm27-mn8w3-dp "exp018: nlm-v27 0.651 recipe + our Ministral-3-8B member mn8v1 w3.0, ATLAS kept up to its deadline, 8B members on a copy per GPU + GPU pooling (~1 h faster)")
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
