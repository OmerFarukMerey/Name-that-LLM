#!/bin/zsh
# usage: push_when_free.sh kernel [kernel ...]   push each kaggle/nlm/<kernel>, retrying every 60 s while Kaggle's
# 2-session GPU limit is reached or the network is down, then wait until the pushed ones are no longer QUEUED/RUNNING
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
cd /Users/ofm/Desktop/Kaggle/Name-that-LLM
names=()
for k in "$@"; do
  for i in $(seq 1 240); do
    r=$(/opt/anaconda3/bin/python $S/tools/kgl.py kernels push -p kaggle/nlm/$k 2>&1 | grep -v Warning)
    echo "$(date -u +%H:%M:%S) $k: $(echo "$r" | tail -1 | cut -c1-160)"
    echo "$r" | grep -q "successfully pushed" && { names+=($k); break; }
    # retry while both GPU sessions are taken or the network is down; any other error skips this kernel
    echo "$r" | grep -qE "Maximum batch GPU session|ConnectionError|NameResolution|Max retries|timed out" \
      || { echo "push of $k failed, skipped"; break; }
    sleep 60
  done
done
[ ${#names} = 0 ] && { echo "nothing pushed"; exit 1; }
$S/tools/wait_kernels.sh $names
