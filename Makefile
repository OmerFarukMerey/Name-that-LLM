PY ?= python
COMP := name-that-llm
KERNEL := omerfarukmerey/name-that-llm-inference
STAGE := build/kaggle-dataset

.PHONY: data test dataset dataset-create dataset-version kernel submit

data:  # download and unzip the competition files into data/raw
	kaggle competitions download -c $(COMP) -p data/raw
	unzip -o -q data/raw/$(COMP).zip -d data/raw && rm data/raw/$(COMP).zip

test:
	$(PY) -m pytest -q

dataset:  # stage src/ntl + model files:  make dataset EXP="exp001_x exp002_y"
	$(PY) scripts/build_kaggle_dataset.py $(EXP)

dataset-create: dataset  # first upload only; private by default
	kaggle datasets create -p $(STAGE) --dir-mode zip

dataset-version: dataset  # make dataset-version EXP=exp001_x MSG="..."
	kaggle datasets version -p $(STAGE) --dir-mode zip -m "$(MSG)"

kernel:  # push the inference notebook; Kaggle runs it on the 3-row test set
	kaggle kernels push -p kaggle/inference

submit:  # make submit V=<kernel version> MSG="..."
	kaggle competitions submit $(COMP) -k $(KERNEL) -v $(V) -f submission.parquet -m "$(MSG)"
