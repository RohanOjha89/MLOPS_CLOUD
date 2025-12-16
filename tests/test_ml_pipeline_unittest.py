import unittest
import base64
import pickle
import numpy as np

from dags.src import ml_pipeline


class TestMLPipeline(unittest.TestCase):

    def test_load_data_decodes_to_dataframe(self):
        b64 = ml_pipeline.load_data()
        raw = base64.b64decode(b64)
        df = pickle.loads(raw)
        self.assertTrue(len(df) > 0)

    def test_data_preprocessing_output_shape(self):
        b64 = ml_pipeline.load_data()
        clustered_b64 = ml_pipeline.data_preprocessing(b64)
        arr = pickle.loads(base64.b64decode(clustered_b64))
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
