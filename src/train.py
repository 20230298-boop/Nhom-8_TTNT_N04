"""
Mô-đun huấn luyện, tinh chỉnh tham số và đánh giá các mô hình AI cho Đề tài 40.
Các thuật toán thử nghiệm: Decision Tree, Random Forest, Gradient Boosting.
Ghi nhận nhật ký huấn luyện (training_history.json) và tầm quan trọng thuộc tính.
"""
import os
import sys
import json
import joblib
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Union, Optional

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from src.config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, BEST_MODEL_PATH,
    PREPROCESSOR_PATH, MODEL_DIR, TARGET_NAMES, RAW_DATA_PATH
)
from src.data_loader import load_student_data, split_features_target
from src.preprocessor import StudentDataPreprocessor

HISTORY_FILE_PATH = MODEL_DIR / "training_history.json"
IMPORTANCE_FILE_PATH = MODEL_DIR / "feature_importances.json"

def get_feature_names(preprocessor: StudentDataPreprocessor) -> list:
    """
    Trích xuất tên các cột đặc trưng sau khi qua ColumnTransformer.
    """
    feature_names = []
    if preprocessor.column_transformer is not None:
        # 1. Các cột số
        feature_names.extend(preprocessor.numerical_cols)
        
        # 2. Các cột mã hóa OneHotEncoder
        try:
            cat_transformer = preprocessor.column_transformer.named_transformers_['cat']
            encoder = cat_transformer.named_steps['encoder']
            cat_encoded_cols = encoder.get_feature_names_out(preprocessor.categorical_cols)
            feature_names.extend(list(cat_encoded_cols))
        except Exception:
            feature_names.extend(preprocessor.categorical_cols)
            
    return feature_names

def record_training_history(dataset_name: str, sample_count: int, best_model_name: str, accuracy: float, f1_score_val: float):
    """
    Ghi lại lịch sử huấn luyện vào file JSON.
    """
    history = []
    if HISTORY_FILE_PATH.exists():
        try:
            with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
            
    new_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "sample_count": sample_count,
        "best_model_name": best_model_name,
        "accuracy": round(accuracy * 100, 2),
        "f1_score": round(f1_score_val, 4)
    }
    history.append(new_entry)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def record_feature_importances(model, feature_names: list):
    """
    Trích xuất và lưu Top các thuộc tính quan trọng nhất.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            fi_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False)
            
            top_fi = fi_df.head(15).to_dict(orient="records")
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(IMPORTANCE_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(top_fi, f, ensure_ascii=False, indent=2)

def train_and_evaluate_models(data_path: Union[str, Path] = RAW_DATA_PATH) -> Dict[str, Any]:
    """
    Thực hiện quy trình huấn luyện, tối ưu siêu tham số và đánh giá các mô hình.

    Tham số:
        data_path (Union[str, Path]): Đường dẫn tới file dữ liệu huấn luyện CSV.

    Trả về:
        Dict[str, Any]: Từ điển chứa kết quả đánh giá các mô hình và mô hình tốt nhất.
    """
    data_path_obj = Path(data_path)
    print("=" * 60)
    print(f"BAT DAU HUAN LUYEN MO HINH AI TREN TAP DU LIEU: {data_path_obj.name}")
    print("=" * 60)

    # 1. Nạp dữ liệu
    df = load_student_data(data_path_obj)
    X, y = split_features_target(df)
    
    # 2. Chia tập Train/Test (80/20) phân lớp đồng đều (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Tap Train: {len(X_train)} mau | Tap Test: {len(X_test)} mau")

    # 3. Tiền xử lý và Cân bằng dữ liệu SMOTE trên tập Train
    preprocessor = StudentDataPreprocessor(random_state=RANDOM_STATE)
    X_train_res, y_train_res = preprocessor.fit_transform_resample(X_train, y_train)
    
    # Tiền xử lý tập Test (không dùng SMOTE để tránh rò rỉ dữ liệu)
    X_test_trans = preprocessor.transform(X_test)
    y_test_encoded = preprocessor.encode_target(y_test)

    print(f"Kich thuoc tap Train sau khi SMOTE: {X_train_res.shape[0]} mau")

    # 4. Định nghĩa danh sách các mô hình và không gian siêu tham số
    models_config = {
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "max_depth": [10],
                "min_samples_split": [2]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            "params": {
                "n_estimators": [100],
                "max_depth": [15],
                "min_samples_split": [2]
            }
        }
    }

    results = {}
    best_f1 = -1.0
    best_accuracy = 0.0
    best_model_name = ""
    best_estimator = None

    # 5. Huấn luyện và tìm kiếm siêu tham số tối ưu bằng GridSearchCV
    for model_name, config in models_config.items():
        print(f"\n--- Dang thuc hien huan luyen cho mo hinh: {model_name} ---")
        sys.stdout.flush()
        
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=1
        )
        grid_search.fit(X_train_res, y_train_res)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_trans)
        
        acc = accuracy_score(y_test_encoded, y_pred)
        f1_w = f1_score(y_test_encoded, y_pred, average='weighted')
        cm = confusion_matrix(y_test_encoded, y_pred)
        
        results[model_name] = {
            "best_params": grid_search.best_params_,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "confusion_matrix": cm,
            "model": best_model
        }
        
        print(f"Tham so toi uu: {grid_search.best_params_}")
        print(f"Do chinh xac (Accuracy): {acc:.4f}")
        print(f"F1-Score (Weighted): {f1_w:.4f}")
        sys.stdout.flush()
        
        if f1_w > best_f1:
            best_f1 = f1_w
            best_accuracy = acc
            best_model_name = model_name
            best_estimator = best_model

    print("\n" + "=" * 60)
    print(f"MO HINH XUAT SAC NHAT: {best_model_name} voi F1-Score: {best_f1:.4f}")
    print("=" * 60)

    # 6. Lưu mô hình tốt nhất, preprocessor và nhật ký
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_estimator, BEST_MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # Ghi nhận lịch sử và tầm quan trọng thuộc tính
    record_training_history(data_path_obj.name, len(df), best_model_name, best_accuracy, best_f1)
    feature_names = get_feature_names(preprocessor)
    record_feature_importances(best_estimator, feature_names)

    print(f"Da luu mo hinh toi uu tai: {BEST_MODEL_PATH}")
    print(f"Da luu nhat ky huan luyen tai: {HISTORY_FILE_PATH}")

    return {
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "results": results
    }

if __name__ == "__main__":
    train_and_evaluate_models()
