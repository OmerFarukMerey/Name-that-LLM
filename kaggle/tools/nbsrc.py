import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    print(f"#### CELL {i} [{c['cell_type']}]")
    print("".join(c["source"]))
