#!/bin/zsh
# usage: watch_top.sh "HH:MM" [refs|owners|first]   every 30 min until that UTC time (same day), compare the Code tab's
# score-sorted top 3 (ref @ last run time; with "refs" the refs only, so re-saves of the same notebooks do not count;
# with "owners" their authors only, so renames do not count either; with "first" the first entry's author only, for when
# the entries below it are ties whose order can shuffle) with the first reading; on a change print the new top 8 and exit
S=/private/tmp/claude-501/-Users-ofm-Desktop-Kaggle-Name-that-LLM/0ad83a29-2ffe-4376-89b7-1dacba45d40b/scratchpad
MODE=$2
top() { /opt/anaconda3/bin/python $S/tools/top_kernels.py 8 2>/dev/null | grep -v -i 'warning\|access token' ; }
key() { echo "$1" | head -$([ "$MODE" = first ] && echo 1 || echo 3) | awk -v m="$MODE" '{split($6, o, "/"); print (m == "owners" || m == "first" ? o[1] : m == "refs" ? $6 : $6 "@" $2 "T" $3)}' | tr '\n' ' '; }
first=$(top); base=$(key "$first"); echo "$(date -u +%H:%M) baseline: $base"
end=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u +%Y-%m-%d) $1" +%s)
while [ $(date -u +%s) -lt $end ]; do
  sleep 1800
  cur=$(top); k=$(key "$cur")
  [ -n "$k" ] && [ "$k" != "$base" ] && { echo "$(date -u +%H:%M) TOP-3 CHANGED: $k"; echo "$cur"; exit 0; }
done
echo "$(date -u +%H:%M) watch ended (no change)"
