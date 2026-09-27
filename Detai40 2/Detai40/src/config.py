"""
File cấu hình chứa các hằng số chung cho toàn bộ dự án Đề tài 40.
"""
from pathlib import Path

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).resolve().parent.parent

# Đường dẫn dữ liệu
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "student_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_student_data.csv"

# Đường dẫn mô hình
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_student_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Cấu hình tham số Machine Learning
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Tên cột biến mục tiêu
TARGET_COLUMN = "Target"

# Ánh xạ nhãn biến mục tiêu
TARGET_MAPPING = {
    "Dropout": 0,    # Thôi học / Bỏ học
    "Enrolled": 1,   # Đang học
    "Graduate": 2    # Tốt nghiệp
}

TARGET_NAMES = ["Thôi học (Dropout)", "Đang học (Enrolled)", "Tốt nghiệp (Graduate)"]
