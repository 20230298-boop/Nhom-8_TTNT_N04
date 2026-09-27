"""
Script tải bộ dữ liệu UCI ID 697 (Predict Students' Dropout and Academic Success)
và lưu bản sao vào thư mục data/student_data.csv.
"""
import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_and_save_dataset(data_dir: str = "data", filename: str = "student_data.csv") -> str:
    """
    Tải bộ dữ liệu từ UCI Repo (ID: 697) và lưu thành file CSV.

    Tham số:
        data_dir (str): Thư mục lưu dữ liệu. Mặc định là "data".
        filename (str): Tên file CSV đầu ra. Mặc định là "student_data.csv".

    Trả về:
        str: Đường dẫn tuyệt đối tới file CSV đã lưu.
    """
    print("Dang tai bo du lieu UCI ID: 697 (Predict Students' Dropout and Academic Success)...")
    
    # Tải dataset từ ucimlrepo
    dataset = fetch_ucirepo(id=697)
    
    # Tách đặc trưng (features) và biến mục tiêu (targets)
    X = dataset.data.features
    y = dataset.data.targets
    
    # Kết hợp X và y thành một DataFrame hoàn chỉnh
    df = pd.concat([X, y], axis=1)
    
    # Tạo thư mục dữ liệu nếu chưa tồn tại
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, filename)
    
    # Lưu ra file CSV
    df.to_csv(output_path, index=False)
    print(f"Tai du lieu thanh cong! Da luu tai: {output_path}")
    print(f"Kich thuoc du lieu: {df.shape[0]} dong, {df.shape[1]} cot.")
    
    return output_path

if __name__ == "__main__":
    download_and_save_dataset()
