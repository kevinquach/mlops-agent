from datetime import datetime, timezone

import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X.columns = X.columns.str.replace("/", "_").str.replace(" ", "_")

X["wine_id"] = range(len(X))
X["target"] = y
X["event_timestamp"] = datetime.now(tz=timezone.utc)

X.to_parquet("feature_store/data/wine_features.parquet", index=False)
print(f"Created feature data: {X.shape[0]} rows, {X.shape[1]} columns")
print(f"Columns: {X.columns.tolist()}")