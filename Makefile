.PHONY: feast-apply feast-materialize train serve feast-check

feast-apply:
	feast -c feature_store apply

feast-materialize:
	feast -c feature_store materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

train:
	python -m src.training.train

serve:
	python -m src.training.serve

feast-check:
	python -c "from feast import FeatureStore; s = FeatureStore(repo_path='feature_store'); print(s.get_online_features(features=['wine_features:alcohol'], entity_rows=[{'wine_id': 0}]).to_df())"