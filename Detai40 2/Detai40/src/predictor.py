"""
Mô-đun dự đoán kết quả học tập và khả năng tốt nghiệp của sinh viên.
Nạp mô hình đã huấn luyện (best_student_model.pkl) và preprocessor để thực hiện suy luận (Inference).
"""
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Union, List, Tuple

from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, TARGET_NAMES
from src.preprocessor import StudentDataPreprocessor

# Danh sách đảo ngược mapping
INVERSE_TARGET_MAPPING = {
    0: "Dropout",
    1: "Enrolled",
    2: "Graduate"
}

VIETNAMESE_LABEL_MAPPING = {
    "Dropout": "Thoi hoc / Nguy co cao",
    "Enrolled": "Dang theo hoc / Nguy co trung binh",
    "Graduate": "Tot nghiep / An toan"
}

class StudentGraduationPredictor:
    """
    Lớp cung cấp giao diện nạp mô hình và thực hiện dự đoán cho mẫu sinh viên mới.
    """
    def __init__(self, model_path: Union[str, Path] = BEST_MODEL_PATH, preprocessor_path: Union[str, Path] = PREPROCESSOR_PATH):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)
        self.model = None
        self.preprocessor = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Nạp mô hình và preprocessor từ đĩa cứng.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Khong tim thay file mo hinh tai: {self.model_path.resolve()}")
        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Khong tim thay file preprocessor tai: {self.preprocessor_path.resolve()}")

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

    def predict_batch(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Dự đoán kết quả cho danh sách nhiều sinh viên (dạng DataFrame).

        Tham số:
            df_raw (pd.DataFrame): DataFrame chứa dữ liệu đặc trưng của các sinh viên.

        Trả về:
            pd.DataFrame: DataFrame bổ sung các cột 'Predicted_Label', 'Predicted_Status', và xác suất rủi ro.
        """
        # Lọc chỉ lấy các cột thuộc tính nằm trong pipeline tiền xử lý (loại bỏ Mã SV, Họ tên)
        feature_cols = self.preprocessor.numerical_cols + self.preprocessor.categorical_cols
        valid_feature_cols = [c for c in feature_cols if c in df_raw.columns]
        
        df_input = df_raw[valid_feature_cols] if valid_feature_cols else df_raw
        
        # Tiền xử lý dữ liệu đầu vào
        X_trans = self.preprocessor.transform(df_input)
        
        # Dự đoán nhãn và xác suất
        preds = self.model.predict(X_trans)
        probs = self.model.predict_proba(X_trans)
        
        results_df = df_raw.copy()
        results_df["Predicted_Code"] = preds
        results_df["Predicted_Label"] = [INVERSE_TARGET_MAPPING.get(p, "Unknown") for p in preds]
        results_df["Predicted_Status"] = [VIETNAMESE_LABEL_MAPPING.get(results_df["Predicted_Label"].iloc[i], "") for i in range(len(preds))]
        
        # Xác suất tương ứng
        results_df["Prob_Dropout"] = probs[:, 0]
        results_df["Prob_Enrolled"] = probs[:, 1]
        results_df["Prob_Graduate"] = probs[:, 2]
        
        return results_df

    def predict_single(self, student_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dự đoán cho một sinh viên đơn lẻ từ từ điểm thông tin.

        Tham số:
            student_dict (Dict[str, Any]): Từ điển thuộc tính sinh viên.

        Trả về:
            Dict[str, Any]: Kết quả dự đoán kèm xác suất rủi ro.
        """
        df_single = pd.DataFrame([student_dict])
        res_df = self.predict_batch(df_single)
        
        row = res_df.iloc[0]
        return {
            "predicted_code": int(row["Predicted_Code"]),
            "predicted_label": row["Predicted_Label"],
            "predicted_status": row["Predicted_Status"],
            "prob_dropout": float(row["Prob_Dropout"]),
            "prob_enrolled": float(row["Prob_Enrolled"]),
            "prob_graduate": float(row["Prob_Graduate"])
        }
