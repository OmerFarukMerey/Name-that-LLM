"""print a Kaggle kernel log (JSON list of {time, data}) without progress bars / warnings: time + line"""
import json, re, sys
raw = open(sys.argv[1]).read()
ev = json.loads(raw)
noise = re.compile(r'Debugger|frozen|PYDEVD|UNEXPECTED|LOAD REPORT|Key +\||^-+$|Warning|warn|it/s\]|s/it\]|Loading (checkpoint|weights)|\d+%\|')
t_from = float(sys.argv[2]) if len(sys.argv) > 2 else 0
for e in ev:
    t = e.get('time', 0)
    if t < t_from: continue
    for l in e.get('data', '').splitlines():
        if l.strip() and not noise.search(l):
            print(f'{t:9.0f} {l[:190]}')
