from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine

data_dir = Path(__file__).parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X.columns = X.columns.str.replace("/", "_").str.replace(" ", "_")

X["wine_id"] = range(len(X))
X["target"] = y
X["event_timestamp"] = datetime.now(tz=timezone.utc)

output_path = data_dir / "wine_features.parquet"
X.to_parquet(output_path, index=False)
print(f"Created feature data: {X.shape[0]} rows, {X.shape[1]} columns, at {output_path}")
print(f"Columns: {X.columns.tolist()}")