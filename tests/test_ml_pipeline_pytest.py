import os
import base64
import pickle
import numpy as np
import pytest

# Import your module
from dags.src import ml_pipeline


def test_load_data_returns_base64_string():
    b64 = ml_pipeline.load_data()
    assert isinstance(b64, str)
    # Should decode cleanly
    raw = base64.b64decode(b64)
    df = pickle.loads(raw)
    assert df is not None
    assert len(df) > 0

def test_data_preprocessing_returns_base64_and_numpy_array():
    b64 = ml_pipeline.load_data()
    clustered_b64 = ml_pipeline.data_preprocessing(b64)

    assert isinstance(clustered_b64, str)

    arr = pickle.loads(base64.b64decode(clustered_b64))
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape[1] == 3  # BALANCE, PURCHASES, CREDIT_LIMIT
    assert arr.shape[0] > 0


@pytest.mark.slow

def test_build_save_model_exists():
    assert callable(ml_pipeline.build_save_model)

# def test_build_save_model_saves_model_file(tmp_path, monkeypatch):
#     """
#     This test is marked slow because build_save_model trains KMeans 49 times.
#     We'll reduce the workload via monkeypatch so CI stays fast.
#     """

#     # 1) Create tiny fake data (already "preprocessed")
#     X = np.random.rand(20, 3)
#     data_b64 = base64.b64encode(pickle.dumps(X)).decode("ascii")

#     # 2) Redirect the output model directory into tmp_path
#     # build_save_model uses: os.path.dirname(os.path.dirname(__file__)) + "/model"
#     # We'll monkeypatch __file__ location by monkeypatching os.path.dirname calls is messy.
#     # Easiest: monkeypatch os.path.join/os.makedirs/open? Not great.
#     #
#     # Better: just ensure it writes into the repo's model/ folder and clean up after,
#     # OR refactor build_save_model to accept output_dir. (Recommended.)
#     #
#     # For now: run with a filename and then assert file exists in expected path.

#     filename = "test_model.pkl"

#     # 3) Monkeypatch range(1, 50) -> range(1, 4) to make it fast
#     original_range = range

#     def fast_range(a, b):
#         # only affects this function call path
#         if a == 1 and b == 50:
#             return original_range(1, 4)
#         return original_range(a, b)

#     monkeypatch.setattr(ml_pipeline, "range", fast_range)

#     # sse = ml_pipeline.build_save_model(data_b64, filename)
#     sse = build_save_model(data_b64, "model.pkl", k_min=1, k_max=4)
#     assert isinstance(sse, list)
#     assert len(sse) == 3  # k=1..3 because of fast_range

#     # expected path: <repo_root>/dags/model/<filename> ? (depends on your path logic)
#     # Your build_save_model uses: dirname(dirname(__file__)) -> dags/  then /model
#     model_path = os.path.join(os.path.dirname(os.path.dirname(ml_pipeline.__file__)), "model", filename)
#     assert os.path.exists(model_path)

#     # cleanup
#     os.remove(model_path)