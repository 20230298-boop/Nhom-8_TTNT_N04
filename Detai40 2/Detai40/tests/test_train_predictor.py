"""
Unit Test kiểm thử quy trình huấn luyện mô hình train.py và suy luận predictor.py.
"""
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_student_data, split_features_target
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH
from src.predictor import StudentGraduationPredictor

class TestTrainAndPredictor(unittest.TestCase):
    """
    Tập hợp các bài test cho mô hình huấn luyện và predictor.
    """
    @classmethod
    def setUpClass(cls):
        """
        Nạp dữ liệu mẫu để kiểm thử.
        """
        cls.df = load_student_data()
        cls.X, cls.y = split_features_target(cls.df)

    def test_artifacts_exist(self):
        """
        Kiểm tra sự tồn tại của file mô hình và preprocessor sau khi huấn luyện.
        """
        self.assertTrue(Path(BEST_MODEL_PATH).exists(), f"Chua tim thấy file model tại {BEST_MODEL_PATH}")
        self.assertTrue(Path(PREPROCESSOR_PATH).exists(), f"Chua tim thấy file preprocessor tại {PREPROCESSOR_PATH}")

    def test_predictor_batch(self):
        """
        Kiểm tra suy luận dự đoán cho một lô 10 sinh viên mẫu.
        """
        predictor = StudentGraduationPredictor()
        sample_df = self.X.head(10)
        results = predictor.predict_batch(sample_df)
        
        self.assertEqual(len(results), 10)
        self.assertIn("Predicted_Label", results.columns)
        self.assertIn("Predicted_Status", results.columns)
        self.assertIn("Prob_Dropout", results.columns)
        self.assertIn("Prob_Graduate", results.columns)

    def test_predictor_single(self):
        """
        Kiểm tra suy luận dự đoán cho 1 sinh viên đơn lẻ.
        """
        predictor = StudentGraduationPredictor()
        single_student = self.X.iloc[0].to_dict()
        res = predictor.predict_single(single_student)
        
        self.assertIn("predicted_label", res)
        self.assertIn(res["predicted_label"], ["Dropout", "Enrolled", "Graduate"])
        self.assertGreaterEqual(res["prob_graduate"], 0.0)
        self.assertLessEqual(res["prob_graduate"], 1.0)

if __name__ == "__main__":
    unittest.main()
