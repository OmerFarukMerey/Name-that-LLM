#!/bin/zsh
# usage: midnight_submit8.sh "YYYY-MM-DD HH:MM:SS" kernel:version [kernel:version ...]
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
     ntl-nlm27-mn8w3-dp "exp018: nlm-v27 0.651 recipe + our Ministral-3-8B member mn8v1 w3.0, ATLAS kept up to its deadline, 8B members on a copy per GPU + GPU pooling (~1 h faster)"
     ntl-nlm27-mn8w2-dp-a2 "exp019: mn8w2-dp + a second ATLAS bundle (harshitakumari256 Qwen3-4B) at 10% after ATLAS-8B on the same low-margin rows, up to the deadline"
     ntl-nlm27-mn8w2-dp-a20 "exp019: mn8w2-dp with ATLAS-8B mixed at 20% instead of 10%"
     ntl-nlm27-dp "exp019 control: nlm-v27 0.651 recipe with the exp018 copy-per-GPU l24_predict, ATLAS up to its deadline, no Ministral"
     ntl-nlm27-dp-a2 "exp019: ntl-nlm27-dp + a second ATLAS bundle (harshitakumari256 Qwen3-4B) at 10% after ATLAS-8B"
     ntl-nlm27-dp-a20 "exp019: ntl-nlm27-dp with ATLAS-8B mixed at 20% instead of 10%"
     ntl-v38-dp "exp020 control: Roman Tamrazov's 0.654 (nlm-v27 + gated typography likelihood in place of the post step, verbatim) with exp018's copy-per-GPU l24_predict"
     ntl-v38-dp-a2 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + second ATLAS bundle (harshitakumari256 Qwen3-4B) at 10%"
     ntl-v38-dp-a20 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + ATLAS-8B at 20%"
     ntl-v38-dp-a2-a20 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + second ATLAS 10% + ATLAS-8B at 20%"
     ntl-v38-mn8w2-dp "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + our Ministral-3-8B member mn8v1 w2.0"
     ntl-v38-mn8w2-dp-a2 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Ministral mn8v1 w2.0 + second ATLAS 10%"
     ntl-v38-mn8w3-dp "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + our Ministral-3-8B member mn8v1 w3.0"
     ntl-v38-mn8w2-dp-a20 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Ministral mn8v1 w2.0 + ATLAS-8B at 20%"
     ntl-v38-mn8w3-dp-a20 "exp020: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Ministral mn8v1 w3.0 + ATLAS-8B at 20%"
     ntl-v38-dp-a30 "exp020/021: the 0.654 recipe (Roman's typography step) + exp018 speed-up + ATLAS-8B at 30% (was 10%)"
     ntl-v38-dp-a40 "exp020/021: the 0.654 recipe (Roman's typography step) + exp018 speed-up + ATLAS-8B at 40% (was 10%)"
     ntl-v38-dp-a30-m45 "exp020/021: the 0.654 recipe (Roman's typography step) + exp018 speed-up + ATLAS-8B at 30% on rows with top-2 margin < 0.45 (was 10%, < 0.3)"
     ntl-v38-dp-sk50 "exp024: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Sinkhorn 0.5 (was 1.0; PeterDreamHan's Soft Sinkhorn 0.5, public 0.653)"
     ntl-v38-dp-sk50-a20 "exp024: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Sinkhorn 0.5 (was 1.0) + ATLAS-8B at 20% (was 10%)"
     ntl-v38-dp-sk50-a30 "exp024: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Sinkhorn 0.5 (was 1.0) + ATLAS-8B at 30% (was 10%)"
     ntl-v38-dp-sk25-a20 "exp024: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Sinkhorn 0.25 (was 1.0) + ATLAS-8B at 20% (was 10%)"
     ntl-v38-dp-sk25 "exp024: the 0.654 recipe (Roman's typography step) + exp018 speed-up + Sinkhorn 0.25 (was 1.0)")
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
