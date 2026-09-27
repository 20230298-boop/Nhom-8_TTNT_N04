"""
Mô-đun nạp và kiểm tra tính hợp lệ của dữ liệu sinh viên.
"""
import sys
import pandas as pd
from pathlib import Path
from typing import Tuple, Union

# Thêm thư mục gốc vào sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, TARGET_COLUMN

def load_student_data(file_path: Union[str, Path] = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Nạp dữ liệu sinh viên từ file CSV và kiểm tra tính hợp lệ.

    Tham số:
        file_path (Union[str, Path]): Đường dẫn tới file dữ liệu CSV.

    Trả về:
        pd.DataFrame: DataFrame chứa dữ liệu sinh viên.

    Ngoại lệ:
        FileNotFoundError: Nếu file dữ liệu không tồn tại tại đường dẫn chỉ định.
        ValueError: Nếu DataFrame nạp vào bị rỗng hoặc thiếu cột mục tiêu.
    """
    path_obj = Path(file_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {path_obj.resolve()}")
    
    df = pd.read_csv(path_obj)
    
    if df.empty:
        raise ValueError(f"File dữ liệu '{path_obj.name}' bị rỗng!")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cột mục tiêu '{TARGET_COLUMN}' không có trong tập dữ liệu.")
    
    return df

def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tách DataFrame thành tập thuộc tính (X) và biến mục tiêu (y).

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu tổng hợp.
        target_col (str): Tên cột biến mục tiêu.

    Trả về:
        Tuple[pd.DataFrame, pd.Series]: (X, y) tương ứng với đặc trưng và nhãn.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

if __name__ == "__main__":
    try:
        data = load_student_data()
        X_df, y_df = split_features_target(data)
        print(f"Nap du lieu thanh cong! Tong so mau: {len(data)}, So dac trung: {X_df.shape[1]}")
    except Exception as err:
        print(f"[LOI] Loi nap du lieu: {err}")
