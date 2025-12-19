import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64
import json
from datetime import datetime, timezone

import boto3


def _read_csv_safe(path: str) -> pd.DataFrame:
    """Read CSV with UTF-8 fallback handling for CI/Linux environments."""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _s3_client():
    # Uses task role credentials on ECS automatically (recommended)
    # Uses local AWS creds if run locally
    return boto3.client("s3")


def _get_s3_config():
    """
    Reads S3 config from environment variables.
    Required in ECS:
      - S3_BUCKET
      - S3_PREFIX
      - RUN_ID
    """
    bucket = os.environ.get("S3_BUCKET")
    prefix = os.environ.get("S3_PREFIX", "mlops-cloud").strip("/")
    run_id = os.environ.get("RUN_ID")

    if not bucket or not run_id:
        raise ValueError(
            "Missing required env vars. Set S3_BUCKET and RUN_ID (and optionally S3_PREFIX)."
        )
    return bucket, prefix, run_id


def _upload_file_to_s3(local_path: str, bucket: str, key: str):
    s3 = _s3_client()
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def _upload_json_to_s3(obj: dict, bucket: str, key: str):
    s3 = _s3_client()
    body = json.dumps(obj, indent=2).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "../data/file.csv")
    df = _read_csv_safe(csv_path)
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing & returns base64-encoded pickled data.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str, k_min: int = 1, k_max: int = 50):
    """
    Fits KMeans for k in [k_min, k_max) to compute SSE, then saves the LAST fitted model locally.
    Returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    kmeans = None

    for k in range(k_min, k_max):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    if kmeans is None:
        raise ValueError("No model was trained. Check k_min/k_max values.")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads saved model and runs elbow method for logging.
    Returns the first prediction (int) for test.csv.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    with open(output_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Elbow logging (works but may return None for small sse)
    x = list(range(1, len(sse) + 1))
    kl = KneeLocator(x, sse, curve="convex", direction="decreasing")
    elbow = kl.elbow
    print(f"Optimal no. of clusters: {elbow}")

    test_csv_path = os.path.join(os.path.dirname(__file__), "../data/test.csv")
    df = _read_csv_safe(test_csv_path)

    # Avoid sklearn warning: fit on numpy, predict on numpy
    pred = loaded_model.predict(df.values)[0]

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred


def persist_outputs_to_s3(model_filename: str, metrics: dict):
    """
    Uploads model file + metrics.json to S3 using env vars.
    """
    bucket, prefix, run_id = _get_s3_config()

    # Model: s3://bucket/prefix/artifacts/<RUN_ID>/kmeans.pkl
    local_model_path = os.path.join(os.path.dirname(__file__), "../model", model_filename)
    model_key = f"{prefix}/artifacts/{run_id}/{model_filename}"
    model_uri = _upload_file_to_s3(local_model_path, bucket, model_key)

    # Metrics: s3://bucket/prefix/metrics/<RUN_ID>.json
    metrics_key = f"{prefix}/metrics/{run_id}.json"
    metrics_uri = _upload_json_to_s3(metrics, bucket, metrics_key)

    print("Uploaded model to:", model_uri)
    print("Uploaded metrics to:", metrics_uri)

    return model_uri, metrics_uri


if __name__ == "__main__":
    # Pipeline run
    started = datetime.now(timezone.utc).isoformat()

    data = load_data()
    pre = data_preprocessing(data)

    k_min, k_max = 1, 4  # demo small
    model_filename = "kmeans.pkl"
    sse = build_save_model(pre, model_filename, k_min=k_min, k_max=k_max)
    pred = load_model_elbow(model_filename, sse)
    print("Prediction:", pred)

    # Build metrics payload
    metrics = {
        "run_id": os.environ.get("RUN_ID", "local-run"),
        "timestamp_utc": started,
        "k_min": k_min,
        "k_max_exclusive": k_max,
        "sse_points": len(sse),
        "prediction_first_row": pred,
        "note": "Elbow may be None for very small k ranges.",
    }

    # Upload to S3 (requires env vars + IAM permissions)
    model_uri, metrics_uri = persist_outputs_to_s3(model_filename, metrics)