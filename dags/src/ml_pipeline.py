import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

def _read_csv_safe(path: str) -> pd.DataFrame:
    """
    Read CSV with UTF-8 fallback handling for CI/Linux environments.
    """
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")

def load_data():
    # df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    csv_path = os.path.join(os.path.dirname(__file__), "../data/file.csv")
    df = _read_csv_safe(csv_path)
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing & returns base64-encoded pickled clustered data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


# def build_save_model(data_b64: str, filename: str):
# def build_save_model(data_b64: str, filename: str, k_min=1, k_max=50):
def build_save_model(data_b64: str, filename: str, k_min: int = 1, k_max: int = 50):
    """
    Builds a KMeans model on the preprocessed data and saves it. Returns the SSE list (JSON-serializable).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k. Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    # df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    test_csv_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
    df = _read_csv_safe(test_csv_path)
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred


if __name__ == "__main__":
    data = load_data()
    pre = data_preprocessing(data)
    sse = build_save_model(pre, "kmeans.pkl", k_min=1, k_max=4)  # keep small for demo
    pred = load_model_elbow("kmeans.pkl", sse)
    print("Prediction:", pred)