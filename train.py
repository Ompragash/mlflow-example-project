import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_random_forest(n_estimators):
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Random Forest model.')
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in the forest.')
    args = parser.parse_args()

    train_random_forest(n_estimators=args.n_estimators)
