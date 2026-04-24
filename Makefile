.PHONY: feast-apply feast-materialize train serve

feast-apply:
	feast -c feature_store apply

feast-materialize:
	feast -c feature_store materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

train:
	python src/training/train.py

serve:
	python src/training/serve.py