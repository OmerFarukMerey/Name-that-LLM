"""Stage the private Kaggle dataset that the offline inference notebook attaches.

Copies src/ntl and each named experiment's outputs/<name>/model/ into
build/kaggle-dataset/, ready for upload:

    python scripts/build_kaggle_dataset.py exp001_tfidf_lr
    kaggle datasets create  -p build/kaggle-dataset --dir-mode zip        # first time
    kaggle datasets version -p build/kaggle-dataset --dir-mode zip -m msg # afterwards
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "kaggle-dataset"
METADATA = ROOT / "kaggle" / "dataset" / "dataset-metadata.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="*", help="names under outputs/ to ship")
    args = parser.parse_args()

    shutil.rmtree(STAGE, ignore_errors=True)
    STAGE.mkdir(parents=True)
    shutil.copy(METADATA, STAGE)
    shutil.copytree(
        ROOT / "src" / "ntl", STAGE / "ntl", ignore=shutil.ignore_patterns("__pycache__")
    )
    for name in args.experiments:
        model_dir = ROOT / "outputs" / name / "model"
        if not model_dir.is_dir():
            raise SystemExit(f"{model_dir} not found; run experiments/{name}.py first")
        shutil.copytree(model_dir, STAGE / "models" / name)

    files = [p for p in STAGE.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"staged {len(files)} files ({size_mb:.1f} MB) in {STAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
