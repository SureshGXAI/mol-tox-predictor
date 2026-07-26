.PHONY: install test figures baseline train explain mlflow-ui clean

DATA ?= tox21.csv

install:
	pip install -r requirements.txt

test:
	MLFLOW_DISABLE=1 pytest tests/ -q

# Regenerate every figure in reports/figures/ (EDA, comparisons, GBM eval, explainability)
figures:
	python scripts/make_figures.py --data $(DATA)

# Strong classical baseline (no torch needed)
baseline:
	python -m baselines.gbm_baseline --data $(DATA)

# Train the D-MPNN with MLflow tracking (GPU recommended)
train:
	python train.py --data $(DATA) --split scaffold --epochs 60 --ensemble 3

# Explain one molecule (needs trained checkpoints + optionally a running Ollama)
explain:
	python scripts/explain_molecule.py --name caffeine

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns

clean:
	rm -rf __pycache__ */__pycache__ mlruns saved_models reports/explanations
