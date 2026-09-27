"""
Script Phân tích Khám phá Dữ liệu (EDA) cho Đề tài 40.
Tạo các biểu đồ trực quan hóa và báo cáo thống kê dữ liệu sinh viên.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Thêm thư mục gốc dự án vào PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH, TARGET_COLUMN, BASE_DIR

# Thư mục lưu biểu đồ
EDA_OUTPUT_DIR = BASE_DIR / "data" / "eda_plots"

def perform_eda(csv_path: Path = RAW_DATA_PATH) -> None:
    """
    Thực hiện EDA và xuất các biểu đồ trực quan hóa.

    Tham số:
        csv_path (Path): Đường dẫn tới file dữ liệu sinh viên.
    """
    if not csv_path.exists():
        print(f"[LOI] Khong tim thay file du lieu tai: {csv_path}")
        return
    
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    print("=" * 60)
    print("TONG QUAN TAP DU LIEU SINH VIEN (UCI ID: 697)")
    print("=" * 60)
    print(f"- Tong so sinh vien (ban ghi): {df.shape[0]}")
    print(f"- Tong so thuoc tinh (cot): {df.shape[1]}")
    print(f"- So luong gia tri thieu (Null/NaN): {df.isnull().sum().sum()}")
    print(f"- So dong trung lap: {df.duplicated().sum()}")
    
    print("\nPhan bo bien muc tieu (Target):")
    target_counts = df[TARGET_COLUMN].value_counts()
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    for category in target_counts.index:
        print(f"  + {category:12s}: {target_counts[category]:4d} sinh vien ({target_pct[category]:.2f}%)")
    
    # Set style biểu đồ
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["font.family"] = "sans-serif"
    
    # 1. Biểu đồ phân bố Target
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, order=target_counts.index, ax=ax, palette="viridis", legend=False)
    ax.set_title("Phan bo Trang thai Sinh vien (Bien Target)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trang thai", fontsize=12)
    ax.set_ylabel("So luong sinh vien", fontsize=12)
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)} ({height/len(df)*100:.1f}%)",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=11, xytext=(0, 3),
                    textcoords="offset points")
        
    plt.tight_layout()
    target_fig_path = EDA_OUTPUT_DIR / "target_distribution.png"
    plt.savefig(target_fig_path, dpi=300)
    plt.close()
    print(f"Da luu bieu do phan bo Target tai: {target_fig_path}")

    # 2. Phân tích tương quan giữa Điểm Học kỳ 1, Học kỳ 2 với Kết quả Tốt nghiệp
    key_features = [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Admission grade",
        "Age at enrollment"
    ]
    
    existing_key_features = [f for f in key_features if f in df.columns]
    if existing_key_features:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for idx, col in enumerate(existing_key_features):
            sns.boxplot(data=df, x=TARGET_COLUMN, y=col, hue=TARGET_COLUMN, ax=axes[idx], palette="Set2", legend=False)
            axes[idx].set_title(f"Phan bo {col}", fontsize=11, fontweight="bold")
            axes[idx].set_xlabel("")
        plt.tight_layout()
        box_fig_path = EDA_OUTPUT_DIR / "key_features_boxplot.png"
        plt.savefig(box_fig_path, dpi=300)
        plt.close()
        print(f"Da luu bieu do Boxplot cac dac trung hoc tap tai: {box_fig_path}")

    print("\nHoan thanh phan tich EDA du lieu!")

if __name__ == "__main__":
    perform_eda()
