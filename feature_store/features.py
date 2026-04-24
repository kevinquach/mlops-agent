from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

wine = Entity(
    name="wine_id",
    description="Unique identifier for each wine sample"
)

wine_source = FileSource(
    path="feature_store/data/wine_features.parquet",
    timestamp_field="event_timestamp"
)

wine_feature_view = FeatureView(
    name="wine_features",
    entities=[wine],
    ttl=timedelta(days=365),
    schema=[
        Field(name="alcohol", dtype=Float32),
        Field(name="malic_acid", dtype=Float32),
        Field(name="ash", dtype=Float32),
        Field(name="alcalinity_of_ash", dtype=Float32),
        Field(name="magnesium", dtype=Float32),
        Field(name="total_phenols", dtype=Float32),
        Field(name="flavanoids", dtype=Float32),
        Field(name="nonflavanoid_phenols", dtype=Float32),
        Field(name="proanthocyanins", dtype=Float32),
        Field(name="color_intensity", dtype=Float32),
        Field(name="hue", dtype=Float32),
        Field(name="od280_od315_of_diluted_wines", dtype=Float32),
        Field(name="proline", dtype=Float32),
        Field(name="target", dtype=Int64)
    ],
    source=wine_source,
)


