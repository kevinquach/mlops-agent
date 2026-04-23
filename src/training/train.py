import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# point to your local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# create or connect to an experiment
mlflow.set_experiment("wine-classifier")

def train(n_estimators=100, max_depth=5):
    # load data
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    X.columns = X.columns.str.replace("/", "_").str.replace(" ", "_")
    y = data.target

    X = X + np.random.normal(0, 1.5, X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # log the model with signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            registered_model_name="wine-classifier"
        )

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"\nRun logged to MLflow: http://localhost:5000")

        return accuracy

if __name__ == "__main__":
    # train three versions with different parameters
    # so you can see experiment comparison in MLflow
    print("--- Run 1: default params ---")
    train(n_estimators=100, max_depth=5)

    print("\n--- Run 2: more trees ---")
    train(n_estimators=200, max_depth=5)

    print("\n--- Run 3: deeper trees ---")
    train(n_estimators=100, max_depth=10)