"""
Script tạo bộ dữ liệu sinh viên mở rộng (10,000 bản ghi mô phỏng)
và kết hợp với dữ liệu UCI gốc tạo tập dữ liệu gộp 14,424 sinh viên.
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Thêm thư mục gốc vào sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import RAW_DATA_PATH, DATA_DIR, RANDOM_STATE, TARGET_COLUMN

def generate_synthetic_data(base_df: pd.DataFrame, n_samples: int = 10000) -> pd.DataFrame:
    """
    Sinh n_samples bản ghi dữ liệu sinh viên mô phỏng dựa trên phân bố thống kê của base_df.

    Tham số:
        base_df (pd.DataFrame): DataFrame dữ liệu gốc.
        n_samples (int): Số lượng bản ghi cần tạo mới. Mặc định 10,000.

    Trả về:
        pd.DataFrame: Tập dữ liệu mô phỏng mới.
    """
    np.random.seed(RANDOM_STATE)
    synthetic_data = {}
    
    for col in base_df.columns:
        if col == TARGET_COLUMN:
            # Sinh target theo tỷ lệ thực tế
            target_probs = base_df[col].value_counts(normalize=True)
            synthetic_data[col] = np.random.choice(
                target_probs.index, size=n_samples, p=target_probs.values
            )
        else:
            if base_df[col].dtype == 'object':
                probs = base_df[col].value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(probs.index, size=n_samples, p=probs.values)
            else:
                # Đối với cột số: thêm nhiễu Gaussian nhẹ vào phân bố thực tế
                values = base_df[col].dropna().values
                mean = np.mean(values)
                std = np.std(values)
                
                if std == 0:
                    synthetic_data[col] = np.full(n_samples, mean)
                else:
                    min_val = np.min(values)
                    max_val = np.max(values)
                    sampled = np.random.normal(mean, std, n_samples)
                    
                    # Giới hạn trong khoảng [min, max] thực tế
                    sampled = np.clip(sampled, min_val, max_val)
                    
                    # Nếu cột gốc là số nguyên, làm tròn thành integer
                    if np.issubdtype(base_df[col].dtype, np.integer):
                        sampled = np.round(sampled).astype(int)
                        
                    synthetic_data[col] = sampled

    synthetic_df = pd.DataFrame(synthetic_data)
    return synthetic_df

def main():
    print("=" * 60)
    print("BAT DAU SINH DU LIEU MO RONG CUA SINH VIEN")
    print("=" * 60)
    
    if not RAW_DATA_PATH.exists():
        print(f"[LOI] Khong tim thấy file du lieu goc tai: {RAW_DATA_PATH}")
        return

    base_df = pd.read_csv(RAW_DATA_PATH)
    print(f"Du lieu goc UCI: {len(base_df)} mau")
    
    # 1. Sinh 10,000 mẫu mô phỏng
    synthetic_df = generate_synthetic_data(base_df, n_samples=10000)
    synthetic_path = DATA_DIR / "synthetic_vietnamese_students_10k.csv"
    synthetic_df.to_csv(synthetic_path, index=False)
    print(f"Da sinh và luu tap du lieu mo phong (10,000 mau) tai: {synthetic_path}")
    
    # 2. Gộp dữ liệu gốc + mô phỏng thành 14,424 mẫu
    combined_df = pd.concat([base_df, synthetic_df], ignore_index=True)
    combined_path = DATA_DIR / "combined_training_data.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"Da tao tap du lieu gop tong hop ({len(combined_df)} mau) tai: {combined_path}")
    print("=" * 60)
    print("HOAN THANH TAO DU LIEU MO RONG!")

if __name__ == "__main__":
    main()
