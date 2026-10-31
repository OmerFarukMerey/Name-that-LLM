"""exp009_member_pilot: phase 1 of the training track. Does a LoRA classifier on a backbone family the public
0.64x notebooks don't use add anything to an ensemble?

The pilots train on Kaggle (kaggle/member/build_member.py; T4 x2, one model per outer fold of lightningv08's
OOF bank, official rows only). This script reads a pilot's oof.parquet (from `kaggle kernels output`) and scores it
with ntl.oofbank.yardstick: the member alone, and what it adds to the bank's blend, with a paired prompt bootstrap.

    PYTHONPATH=src python experiments/exp009_member_pilot.py <dir with oof.parquet> [...]

Gate for phase 2 (full training): member alone >= 0.45 and the share gain's 95% interval above 0.
"""

import json
import sys
from pathlib import Path

from ntl.oofbank import member_from_oof, yardstick
from ntl.paths import OUTPUT_DIR

OUT = OUTPUT_DIR / Path(__file__).stem


def main(dirs):
    OUT.mkdir(parents=True, exist_ok=True)
    for d in map(Path, dirs):
        print(f"==== {d.name}")
        metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        for record in metrics.get("fold_records", []):
            print(f"fold {record['fold']}: train {record['train_minutes']:.0f} min, valid acc {record['valid_accuracy']:.3f}")
        report = yardstick(member_from_oof(d / "oof.parquet"))
        gate = report["member"] >= 0.45 and report["share_gain"][1] > 0
        print(f"gate (alone >= 0.45 and share gain interval > 0): {'PASS' if gate else 'FAIL'}")
        (OUT / f"{d.name}.json").write_text(json.dumps(dict(report, gate=gate, hours=metrics.get("hours")), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
