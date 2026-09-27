"""
Mô-đun tiền xử lý dữ liệu và kỹ thuật đặc trưng (Feature Engineering) cho sinh viên.
Sử dụng ColumnTransformer, StandardScaler, OneHotEncoder và SMOTE để xử lý dữ liệu.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Union, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RANDOM_STATE, TARGET_MAPPING, TARGET_COLUMN

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Class tạo các thuộc tính mới (Feature Engineering) từ dữ liệu sinh viên.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo các biến đặc trưng học tập bổ sung.
        """
        X_out = X.copy()
        
        # 1. Tỷ lệ qua môn Học kỳ 1 và Học kỳ 2
        if "Curricular units 1st sem (approved)" in X_out.columns and "Curricular units 1st sem (evaluations)" in X_out.columns:
            enrolled_1st = X_out["Curricular units 1st sem (evaluations)"].replace(0, np.nan)
            X_out["PassRatio_Sem1"] = (X_out["Curricular units 1st sem (approved)"] / enrolled_1st).fillna(0.0)
            
        if "Curricular units 2nd sem (approved)" in X_out.columns and "Curricular units 2nd sem (evaluations)" in X_out.columns:
            enrolled_2nd = X_out["Curricular units 2nd sem (evaluations)"].replace(0, np.nan)
            X_out["PassRatio_Sem2"] = (X_out["Curricular units 2nd sem (approved)"] / enrolled_2nd).fillna(0.0)

        # 2. Mức độ thay đổi điểm số giữa Kỳ 2 và Kỳ 1
        if "Curricular units 2nd sem (grade)" in X_out.columns and "Curricular units 1st sem (grade)" in X_out.columns:
            X_out["Grade_Change"] = X_out["Curricular units 2nd sem (grade)"] - X_out["Curricular units 1st sem (grade)"]

        # 3. Tỷ lệ tổng môn đăng ký vượt qua
        if "Curricular units 1st sem (approved)" in X_out.columns and "Curricular units 2nd sem (approved)" in X_out.columns:
            X_out["Total_Approved"] = X_out["Curricular units 1st sem (approved)"] + X_out["Curricular units 2nd sem (approved)"]

        return X_out

class StudentDataPreprocessor:
    """
    Lớp quản lý toàn bộ Pipeline tiền xử lý dữ liệu và cân bằng dữ liệu SMOTE.
    """
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.feature_engineer = FeatureEngineer()
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder = LabelEncoder()
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.feature_names: List[str] = []

    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Phân loại các cột dạng số và dạng phân loại.
        """
        cat_cols = []
        num_cols = []
        
        # Các thuộc tính định danh / phân loại trong UCI dataset
        categorical_candidates = [
            "Marital status", "Application mode", "Application order", "Course",
            "Daytime/evening attendance", "Previous qualification", "Nacionality",
            "Mother's qualification", "Father's qualification", "Mother's occupation",
            "Father's occupation", "Displaced", "Educational special needs",
            "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder", "International"
        ]
        
        for col in df.columns:
            if col == TARGET_COLUMN:
                continue
            if col in categorical_candidates or df[col].dtype == 'object':
                cat_cols.append(col)
            else:
                num_cols.append(col)
                
        return num_cols, cat_cols

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Khớp (fit) pipeline tiền xử lý với dữ liệu huấn luyện.
        """
        # Bước 1: Feature Engineering
        X_fe = self.feature_engineer.transform(X)
        
        # Bước 2: Phân loại cột
        self.numerical_cols, self.categorical_cols = self._identify_column_types(X_fe)
        
        # Bước 3: ĐỊnh nghĩa Pipeline
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer([
            ('num', num_pipeline, self.numerical_cols),
            ('cat', cat_pipeline, self.categorical_cols)
        ])
        
        self.column_transformer.fit(X_fe)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Biến đổi dữ liệu đặc trưng bằng pipeline đã khớp.
        """
        if self.column_transformer is None:
            raise ValueError("Preprocessor chưa được khớp (fit). Vui lòng gọi fit() trước.")
            
        X_fe = self.feature_engineer.transform(X)
        return self.column_transformer.transform(X_fe)

    def encode_target(self, y: pd.Series) -> np.ndarray:
        """
        Mã hóa biến mục tiêu dạng chuỗi ('Dropout', 'Enrolled', 'Graduate') thành dạng số.
        """
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            # Sử dụng mapping chuẩn trong config nếu có
            return y.map(TARGET_MAPPING).fillna(0).astype(int).values
        return y.values

    def fit_transform_resample(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Khớp preprocessor, biến đổi dữ liệu train và áp dụng SMOTE cân bằng lớp.

        Tham số:
            X_train (pd.DataFrame): Tập đặc trưng train.
            y_train (pd.Series): Tập biến mục tiêu train.

        Trả về:
            Tuple[np.ndarray, np.ndarray]: (X_resampled, y_resampled)
        """
        self.fit(X_train, y_train)
        X_transformed = self.transform(X_train)
        y_encoded = self.encode_target(y_train)
        
        # Áp dụng SMOTE cân bằng dữ liệu
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_transformed, y_encoded)
        
        return X_resampled, y_resampled
