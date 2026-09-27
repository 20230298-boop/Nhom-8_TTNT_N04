"""
Unit Test kiểm thử bộ tiền xử lý StudentDataPreprocessor và FeatureEngineer.
"""
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_student_data, split_features_target
from src.preprocessor import StudentDataPreprocessor, FeatureEngineer

class TestStudentDataPreprocessor(unittest.TestCase):
    """
    Tập hợp các bài test cho tiền xử lý dữ liệu sinh viên.
    """
    @classmethod
    def setUpClass(cls):
        """
        Nạp dữ liệu mẫu từ dataset thực tế.
        """
        cls.df = load_student_data()
        cls.X, cls.y = split_features_target(cls.df)

    def test_feature_engineer(self):
        """
        Kiểm tra khởi tạo các thuộc tính Feature Engineering mới.
        """
        fe = FeatureEngineer()
        X_fe = fe.transform(self.X)
        
        self.assertIn("PassRatio_Sem1", X_fe.columns)
        self.assertIn("PassRatio_Sem2", X_fe.columns)
        self.assertIn("Grade_Change", X_fe.columns)
        self.assertIn("Total_Approved", X_fe.columns)
        self.assertFalse(X_fe["PassRatio_Sem1"].isnull().any())

    def test_preprocessor_fit_transform(self):
        """
        Kiểm tra quá trình fit_transform của StudentDataPreprocessor.
        """
        preprocessor = StudentDataPreprocessor()
        preprocessor.fit(self.X)
        X_transformed = preprocessor.transform(self.X)
        
        # Kiểm tra không có giá trị NaN sau khi biến đổi
        self.assertFalse(np.isnan(X_transformed).any())
        self.assertEqual(X_transformed.shape[0], len(self.X))

    def test_encode_target(self):
        """
        Kiểm tra mã hóa nhãn target dạng chuỗi sang dạng số int.
        """
        preprocessor = StudentDataPreprocessor()
        y_encoded = preprocessor.encode_target(self.y)
        
        self.assertEqual(len(y_encoded), len(self.y))
        self.assertTrue(set(np.unique(y_encoded)).issubset({0, 1, 2}))

    def test_smote_resampling(self):
        """
        Kiểm tra khả năng cân bằng dữ liệu của SMOTE trong fit_transform_resample.
        """
        preprocessor = StudentDataPreprocessor()
        X_res, y_res = preprocessor.fit_transform_resample(self.X, self.y)
        
        # Đảm bảo số lượng mẫu giữa các lớp bằng nhau sau SMOTE
        unique, counts = np.unique(y_res, return_counts=True)
        self.assertEqual(len(set(counts)), 1)
        self.assertGreater(len(y_res), len(self.y))

if __name__ == "__main__":
    unittest.main()
