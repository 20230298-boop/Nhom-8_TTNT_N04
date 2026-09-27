"""
Ung dung Web Streamlit Phan loai Ket qua Hoc tap va Du doan Kha nang Tot nghiep cua Sinh vien.
Phien ban Viet hoa 100%, Hien thi Ma sinh vien & Ho ten, Quan ly Du lieu va Retrain Mo hinh AI.
De tai 40 - Mon hoc: Tri tue nhan tao (AI).
"""
import sys
from pathlib import Path

# Them thu muc goc vao sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.config import RAW_DATA_PATH, DATA_DIR, MODEL_DIR, TARGET_COLUMN
from src.data_loader import load_student_data
from src.predictor import StudentGraduationPredictor
from src.train import train_and_evaluate_models, HISTORY_FILE_PATH, IMPORTANCE_FILE_PATH

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Phân loại & Dự đoán Tốt nghiệp Sinh viên",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Danh sách tên tiếng Việt mẫu sinh động
SAMPLE_VIETNAMESE_NAMES = [
    "Nguyễn Văn An", "Trần Thị Bình", "Lê Hoàng Cường", "Phạm Minh Đức", "Đỗ Thu Hà",
    "Hoàng Văn Hùng", "Vũ Thị Khoa", "Bùi Tấn Lộc", "Đặng Thùy Linh", "Ngô Bảo Nam",
    "Nguyễn Thị Mai", "Phan Văn Phong", "Dương Thị Quyên", "Lương Thành Sơn", "Trịnh Như Thảo",
    "Vũ Hoàng Nam", "Nguyễn Đức Anh", "Trần Khánh Linh", "Lê Thanh Tùng", "Hoàng Kim Oanh"
]

# Từ điển Việt hóa thuộc tính dữ liệu
FEATURE_VIETNAMESE_MAP = {
    "Marital status": "Tình trạng hôn nhân",
    "Application mode": "Phương thức xét tuyển",
    "Application order": "Thứ tự nguyện vọng",
    "Course": "Ngành học",
    "Daytime/evening attendance": "Hình thức đào tạo (Ngày/Tối)",
    "Previous qualification": "Trình độ học vấn trước đó",
    "Nacionality": "Quốc tịch",
    "Mother's qualification": "Trình độ Mẹ",
    "Father's qualification": "Trình độ Bố",
    "Mother's occupation": "Nghề nghiệp Mẹ",
    "Father's occupation": "Nghề nghiệp Bố",
    "Displaced": "Sinh viên ở trọ / Xa nhà",
    "Educational special needs": "Nhu cầu giáo dục đặc biệt",
    "Debtor": "Có nợ tài chính khác",
    "Tuition fees up to date": "Đã đóng đủ học phí",
    "Gender": "Giới tính",
    "Scholarship holder": "Có học bổng",
    "Age at enrollment": "Tuổi khi nhập học",
    "International": "Sinh viên quốc tế",
    "Curricular units 1st sem (credited)": "Số môn miễn giảm HK1",
    "Curricular units 1st sem (enrolled)": "Số môn đăng ký HK1",
    "Curricular units 1st sem (evaluations)": "Số môn thi HK1",
    "Curricular units 1st sem (approved)": "Số môn đỗ HK1",
    "Curricular units 1st sem (grade)": "Điểm trung bình HK1 (Thang 20)",
    "Curricular units 1st sem (without evaluations)": "Số môn không thi HK1",
    "Curricular units 2nd sem (credited)": "Số môn miễn giảm HK2",
    "Curricular units 2nd sem (enrolled)": "Số môn đăng ký HK2",
    "Curricular units 2nd sem (evaluations)": "Số môn thi HK2",
    "Curricular units 2nd sem (approved)": "Số môn đỗ HK2",
    "Curricular units 2nd sem (grade)": "Điểm trung bình HK2 (Thang 20)",
    "Curricular units 2nd sem (without evaluations)": "Số môn không thi HK2",
    "Unemployment rate": "Tỷ lệ thất nghiệp vùng",
    "Inflation rate": "Tỷ lệ lạm phát",
    "GDP": "Tăng trưởng GDP",
    "Admission grade": "Điểm xét tuyển đầu vào",
    "Previous qualification (grade)": "Điểm tốt nghiệp cấp trước",
    "PassRatio_Sem1": "Tỷ lệ đỗ môn HK1",
    "PassRatio_Sem2": "Tỷ lệ đỗ môn HK2",
    "Grade_Change": "Chênh lệch điểm HK2-HK1",
    "Total_Approved": "Tổng số môn đỗ tích lũy"
}

TARGET_VIETNAMESE_MAP = {
    "Graduate": "Tốt nghiệp",
    "Dropout": "Thôi học",
    "Enrolled": "Đang theo học"
}

STATUS_RISK_MAP = {
    "Graduate": "An toàn / Tốt nghiệp đúng hạn",
    "Enrolled": "Nguy cơ trung bình / Nợ môn",
    "Dropout": "Cảnh báo học vụ / Nguy cơ bỏ học"
}

# Tiêm CSS tùy chỉnh giao diện
CUSTOM_CSS = """
<style>
    /* CSS cho thanh Tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #f8f9fa;
        padding: 8px 12px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        color: #495057;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a8a !important;
        color: #ffffff !important;
        box-shadow: 0 4px 6px rgba(30, 58, 138, 0.2);
    }
    /* Metric Card Custom */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-title {
        font-size: 14px;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 24px;
        color: #0f172a;
        font-weight: 700;
    }
    /* Banner Header */
    .banner-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 24px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.3);
    }
    .banner-title {
        font-size: 26px;
        font-weight: 800;
        margin: 0;
    }
    .banner-subtitle {
        font-size: 15px;
        opacity: 0.9;
        margin-top: 6px;
    }
</style>
"""

@st.cache_resource
def get_predictor():
    return StudentGraduationPredictor()

@st.cache_data
def get_raw_data():
    return load_student_data(RAW_DATA_PATH)

def translate_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    rename_dict = {}
    for col in df.columns:
        if col in FEATURE_VIETNAMESE_MAP:
            rename_dict[col] = FEATURE_VIETNAMESE_MAP[col]
        else:
            col_clean = col.strip().lower()
            for k, v in FEATURE_VIETNAMESE_MAP.items():
                if k.lower() == col_clean:
                    rename_dict[col] = v
                    break
    return df.rename(columns=rename_dict)

def ensure_student_identity(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    id_cols = [c for c in df_copy.columns if any(k in c.lower() for k in ["mã sv", "mã sinh viên", "student id", "student_id", "masv", "stt"])]
    if id_cols:
        df_copy.rename(columns={id_cols[0]: "Mã sinh viên"}, inplace=True)
    else:
        df_copy.insert(0, "Mã sinh viên", [f"SV2024{i+1:04d}" for i in range(len(df_copy))])

    name_cols = [c for c in df_copy.columns if any(k in c.lower() for k in ["họ và tên", "họ tên", "fullname", "full_name", "student_name", "tên sinh viên"])]
    if name_cols:
        df_copy.rename(columns={name_cols[0]: "Họ và tên"}, inplace=True)
    else:
        names = [SAMPLE_VIETNAMESE_NAMES[i % len(SAMPLE_VIETNAMESE_NAMES)] for i in range(len(df_copy))]
        df_copy.insert(1, "Họ và tên", names)

    return df_copy

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Banner Tiêu đề chính
    st.markdown("""
        <div class="banner-header">
            <div class="banner-title">HỆ THỐNG PHÂN LOẠI KẾT QUẢ HỌC TẬP VÀ DỰ ĐOÁN KHẢ NĂNG TỐT NGHIỆP SINH VIÊN</div>
            <div class="banner-subtitle">Đề tài 40 - Ứng dụng Trí tuệ nhân tạo (AI) trong Quản lý & Cảnh báo học vụ Sinh viên</div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar thông tin hệ thống
    st.sidebar.header("CẤU HÌNH HỆ THỐNG AI")
    st.sidebar.markdown("""
    * **Mô hình**: Random Forest Classifier
    * **Độ chính xác (Accuracy)**: 75.48%
    * **Độ đo F1 (F1-Score)**: 0.7571
    * **Dữ liệu**: UCI Machine Learning (ID: 697)
    * **Kỹ thuật xử lý**: StandardScaler + OneHotEncoder + SMOTE
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Môn học: Trí tuệ nhân tạo (AI)")

    try:
        predictor = get_predictor()
        df_raw = get_raw_data()
    except Exception as err:
        st.error(f"[LOI] Không thể nạp mô hình hoặc dữ liệu: {err}")
        return

    # Khởi tạo 4 Tabs giao diện
    tab1, tab2, tab3, tab4 = st.tabs([
        "TỔNG QUAN & PHÂN TÍCH EDA",
        "DỰ ĐOÁN ĐƠN SINH VIÊN",
        "DỰ ĐOÁN THEO FILE (BATCH)",
        "NHẬT KÝ HUẤN LUYỆN & GIẢI THÍCH AI"
    ])

    # =========================================================================
    # TAB 1: TỔNG QUAN & EDA INTERACTIVE
    # =========================================================================
    with tab1:
        st.subheader("1. Chỉ số tổng quan dữ liệu sinh viên")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">TỔNG SỐ SINH VIÊN</div>
                <div class="metric-value">{len(df_raw):,}</div>
            </div>
        """, unsafe_allow_html=True)
        
        c2.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">SỐ THUỘC TÍNH PHÂN TÍCH</div>
                <div class="metric-value">{df_raw.shape[1] - 1}</div>
            </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">DỮ LIỆU KHUYẾT (NULL)</div>
                <div class="metric-value">{df_raw.isnull().sum().sum()}</div>
            </div>
        """, unsafe_allow_html=True)

        grad_pct = (df_raw[TARGET_COLUMN] == 'Graduate').mean() * 100
        c4.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">TỶ LỆ TỐT NGHIỆP</div>
                <div class="metric-value">{grad_pct:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("2. Biểu đồ phân bố Trạng thái Học tập của Sinh viên")

        target_counts_raw = df_raw[TARGET_COLUMN].value_counts()
        target_vn_counts = pd.Series({
            TARGET_VIETNAMESE_MAP.get(k, k): v for k, v in target_counts_raw.items()
        })

        col_left, col_right = st.columns(2)
        with col_left:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            sns.barplot(
                x=target_vn_counts.index,
                y=target_vn_counts.values,
                hue=target_vn_counts.index,
                ax=ax_bar,
                palette="viridis",
                legend=False
            )
            ax_bar.set_title("Số lượng Sinh viên theo Trạng thái", fontsize=13, fontweight="bold", pad=12)
            ax_bar.set_xlabel("Trạng thái học tập", fontsize=11)
            ax_bar.set_ylabel("Số lượng sinh viên", fontsize=11)
            
            for p in ax_bar.patches:
                h = p.get_height()
                ax_bar.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2., h),
                                ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')
            st.pyplot(fig_bar)
            plt.close()

        with col_right:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
            ax_pie.pie(
                target_vn_counts.values,
                labels=target_vn_counts.index,
                autopct='%1.1f%%',
                colors=['#22c55e', '#ef4444', '#f59e0b'],
                startangle=140,
                textprops={'fontsize': 11, 'weight': 'bold'}
            )
            ax_pie.set_title("Tỷ lệ % Phân bố Trạng thái", fontsize=13, fontweight="bold", pad=12)
            st.pyplot(fig_pie)
            plt.close()

        st.markdown("---")
        st.subheader("3. Biểu đồ tương quan các thuộc tính học tập")

        feature_options_en = [
            "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 2nd sem (grade)",
            "Admission grade",
            "Age at enrollment"
        ]
        
        feature_options_map = {f: FEATURE_VIETNAMESE_MAP.get(f, f) for f in feature_options_en}
        
        selected_feature_en = st.selectbox(
            "Chọn thuộc tính học tập cần trực quan hóa:",
            options=feature_options_en,
            format_func=lambda x: feature_options_map[x]
        )
        
        selected_feature_vn = feature_options_map[selected_feature_en]

        df_plot = df_raw.copy()
        df_plot["Trạng thái"] = df_plot[TARGET_COLUMN].map(TARGET_VIETNAMESE_MAP)

        fig_box, ax_box = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(
            data=df_plot,
            x="Trạng thái",
            y=selected_feature_en,
            hue="Trạng thái",
            ax=ax_box,
            palette="Set2",
            legend=False
        )
        ax_box.set_title(f"Phân bố: {selected_feature_vn} theo Trạng thái học tập", fontsize=13, fontweight="bold", pad=12)
        ax_box.set_xlabel("Trạng thái sinh viên", fontsize=11)
        ax_box.set_ylabel(selected_feature_vn, fontsize=11)
        st.pyplot(fig_box)
        plt.close()

    # =========================================================================
    # TAB 2: DỰ ĐOÁN ĐƠN SINH VIÊN
    # =========================================================================
    with tab2:
        st.subheader("Nhập thông tin cá nhân và điểm số của sinh viên")
        st.caption("Điền Mã SV, Họ tên cùng các chỉ số học tập để hệ thống đánh giá khả năng tốt nghiệp:")

        with st.form("form_single_student"):
            col_id, col_name = st.columns(2)
            with col_id:
                student_id = st.text_input("Mã số sinh viên", value="SV20240001")
            with col_name:
                student_name = st.text_input("Họ và tên sinh viên", value="Nguyễn Văn An")

            st.markdown("---")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("### Kết quả Học kỳ 1")
                approved_1st = st.number_input("Số môn đỗ HK1", min_value=0, max_value=30, value=5)
                eval_1st = st.number_input("Số môn đăng ký/thi HK1", min_value=0, max_value=30, value=6)
                grade_1st = st.number_input("Điểm trung bình HK1 (Thang điểm 20)", min_value=0.0, max_value=20.0, value=13.5, step=0.1)

            with col_b:
                st.markdown("### Kết quả Học kỳ 2")
                approved_2nd = st.number_input("Số môn đỗ HK2", min_value=0, max_value=30, value=5)
                eval_2nd = st.number_input("Số môn đăng ký/thi HK2", min_value=0, max_value=30, value=6)
                grade_2nd = st.number_input("Điểm trung bình HK2 (Thang điểm 20)", min_value=0.0, max_value=20.0, value=14.0, step=0.1)

            with col_c:
                st.markdown("### Thông tin Cá nhân & Tài chính")
                age = st.number_input("Tuổi khi nhập học", min_value=15, max_value=70, value=20)
                admission_grade = st.number_input("Điểm thi đầu vào (Thang 200)", min_value=0.0, max_value=200.0, value=125.0, step=0.5)
                tuition_up_to_date = st.selectbox("Đã đóng đủ học phí?", [1, 0], format_func=lambda x: "Đã nộp đủ (Đúng hạn)" if x == 1 else "Nợ học phí (Chưa nộp)")
                scholarship = st.selectbox("Có học bổng?", [0, 1], format_func=lambda x: "Không có" if x == 0 else "Có học bổng")
                debtor = st.selectbox("Có khoản nợ tài chính khác?", [0, 1], format_func=lambda x: "Không có nợ" if x == 0 else "Có nợ")

            submit_button = st.form_submit_button("DỰ ĐOÁN KẾT QUẢ SINH VIÊN")

        if submit_button:
            sample_student = df_raw.drop(columns=[TARGET_COLUMN]).iloc[0].to_dict()
            
            sample_student["Curricular units 1st sem (approved)"] = approved_1st
            sample_student["Curricular units 1st sem (evaluations)"] = eval_1st
            sample_student["Curricular units 1st sem (grade)"] = grade_1st
            sample_student["Curricular units 2nd sem (approved)"] = approved_2nd
            sample_student["Curricular units 2nd sem (evaluations)"] = eval_2nd
            sample_student["Curricular units 2nd sem (grade)"] = grade_2nd
            sample_student["Age at enrollment"] = age
            sample_student["Admission grade"] = admission_grade
            sample_student["Tuition fees up to date"] = tuition_up_to_date
            sample_student["Scholarship holder"] = scholarship
            sample_student["Debtor"] = debtor

            res = predictor.predict_single(sample_student)
            
            st.markdown("---")
            st.subheader(f"KẾT QUẢ ĐÁNH GIÁ: {student_name.upper()} (Mã SV: {student_id})")
            
            raw_label = res["predicted_label"]
            vn_label = TARGET_VIETNAMESE_MAP.get(raw_label, raw_label)
            risk_status = STATUS_RISK_MAP.get(raw_label, "")
            
            c_r1, c_r2, c_r3 = st.columns(3)
            
            if raw_label == "Graduate":
                c_r1.success(f"Kết quả dự đoán: {vn_label} ({raw_label})")
                c_r2.success(f"Mức độ an toàn: {risk_status}")
            elif raw_label == "Enrolled":
                c_r1.warning(f"Kết quả dự đoán: {vn_label} ({raw_label})")
                c_r2.warning(f"Mức độ an toàn: {risk_status}")
            else:
                c_r1.error(f"Kết quả dự đoán: {vn_label} ({raw_label})")
                c_r2.error(f"Mức độ an toàn: {risk_status}")

            c_r3.metric("Xác suất Tốt nghiệp dự kiến", f"{res['prob_graduate'] * 100:.1f}%")

            st.markdown("### Chi tiết xác suất phân loại:")
            st.progress(res['prob_graduate'], text=f"Tốt nghiệp (Graduate): {res['prob_graduate']*100:.1f}%")
            st.progress(res['prob_enrolled'], text=f"Đang học (Enrolled): {res['prob_enrolled']*100:.1f}%")
            st.progress(res['prob_dropout'], text=f"Thôi học / Bỏ học (Dropout): {res['prob_dropout']*100:.1f}%")

    # =========================================================================
    # TAB 3: DỰ ĐOÁN THEO FILE BATCH
    # =========================================================================
    with tab3:
        st.subheader("Dự đoán hàng loạt cho danh sách sinh viên")
        st.write("Tải lên file danh sách sinh viên (.csv hoặc .xlsx). Hệ thống tự động nhận diện/tạo **Mã sinh viên** và **Họ tên**:")

        uploaded_file = st.file_uploader("Tải file danh sách sinh viên", type=["csv", "xlsx"])
        use_sample = st.button("Sử dụng dữ liệu thử nghiệm mẫu (10 sinh viên)")

        df_loaded = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_loaded = pd.read_csv(uploaded_file)
                else:
                    df_loaded = pd.read_excel(uploaded_file)
            except Exception as e:
                st.error(f"[LOI] Không thể đọc file: {e}")
        elif use_sample:
            df_loaded = df_raw.drop(columns=[TARGET_COLUMN]).head(10)

        if df_loaded is not None:
            df_to_predict = ensure_student_identity(df_loaded)

            st.markdown(f"**Danh sách dữ liệu sinh viên xem trước ({len(df_to_predict)} sinh viên):**")
            df_preview_ordered = df_to_predict[["Mã sinh viên", "Họ và tên"] + [c for c in df_to_predict.columns if c not in ["Mã sinh viên", "Họ và tên"]]]
            df_preview_vn = translate_dataframe_headers(df_preview_ordered)
            st.dataframe(df_preview_vn.head(10))

            if st.button("KHỞI CHẠY DỰ ĐOÁN HÀNG LOẠT"):
                with st.spinner("Đang xử lý dữ liệu và thực thi mô hình AI..."):
                    results_df = predictor.predict_batch(df_to_predict)
                
                st.success(f"Hoàn thành dự đoán thành công cho {len(results_df)} sinh viên!")
                
                results_df["Trạng thái Dự đoán"] = results_df["Predicted_Label"].map(TARGET_VIETNAMESE_MAP)
                results_df["Đánh giá Rủi ro"] = results_df["Predicted_Label"].map(STATUS_RISK_MAP)
                results_df["Xác suất Tốt nghiệp (%)"] = (results_df["Prob_Graduate"] * 100).round(1)
                results_df["Xác suất Bỏ học (%)"] = (results_df["Prob_Dropout"] * 100).round(1)

                st.markdown("---")
                st.subheader("1. Thống kê và Biểu đồ Kết quả Dự đoán Lô")

                batch_counts = results_df["Trạng thái Dự đoán"].value_counts()
                b_grad = batch_counts.get("Tốt nghiệp", 0)
                b_enrolled = batch_counts.get("Đang theo học", 0)
                b_dropout = batch_counts.get("Thôi học", 0)

                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Tổng sinh viên", len(results_df))
                bm2.metric("Dự đoán Tốt nghiệp", b_grad)
                bm3.metric("Dự đoán Đang học", b_enrolled)
                bm4.metric("Dự đoán Thôi học (Nguy cơ)", b_dropout)

                bc1, bc2 = st.columns(2)
                with bc1:
                    fig_batch_bar, ax_bbar = plt.subplots(figsize=(6, 3.8))
                    sns.barplot(
                        x=batch_counts.index,
                        y=batch_counts.values,
                        hue=batch_counts.index,
                        ax=ax_bbar,
                        palette="viridis",
                        legend=False
                    )
                    ax_bbar.set_title("Phân bố Trạng thái Dự đoán (Tập File tải lên)", fontsize=12, fontweight="bold")
                    ax_bbar.set_ylabel("Số sinh viên", fontsize=10)
                    for p in ax_bbar.patches:
                        h = p.get_height()
                        ax_bbar.annotate(f"{int(h)}", (p.get_x() + p.get_width()/2., h),
                                         ha='center', va='bottom', fontsize=10, xytext=(0, 2), textcoords='offset points')
                    st.pyplot(fig_batch_bar)
                    plt.close()

                with bc2:
                    fig_batch_pie, ax_bpie = plt.subplots(figsize=(6, 3.8))
                    ax_bpie.pie(
                        batch_counts.values,
                        labels=batch_counts.index,
                        autopct='%1.1f%%',
                        colors=['#22c55e', '#ef4444', '#f59e0b'],
                        startangle=140,
                        textprops={'fontsize': 10, 'weight': 'bold'}
                    )
                    ax_bpie.set_title("Tỷ lệ % Phân bố Kết quả", fontsize=12, fontweight="bold")
                    st.pyplot(fig_batch_pie)
                    plt.close()

                st.markdown("---")
                st.subheader("2. Danh sách Cảnh báo Học vụ & Kết quả Chi tiết theo Sinh viên")

                high_risk_batch = results_df[results_df["Prob_Dropout"] > 0.5]
                if not high_risk_batch.empty:
                    st.error(f"CẢNH BÁO HỌC VỤ: Phát hiện {len(high_risk_batch)} sinh viên có xác suất Thôi học (Dropout) > 50%!")
                    high_risk_show = translate_dataframe_headers(high_risk_batch[["Mã sinh viên", "Họ và tên", "Trạng thái Dự đoán", "Đánh giá Rủi ro", "Xác suất Bỏ học (%)", "Xác suất Tốt nghiệp (%)"]])
                    st.dataframe(high_risk_show)

                st.markdown("**Bảng kết quả toàn bộ danh sách sinh viên (Đã Việt hóa 100%):**")
                priority_cols = ["Mã sinh viên", "Họ và tên", "Trạng thái Dự đoán", "Đánh giá Rủi ro", "Xác suất Tốt nghiệp (%)", "Xác suất Bỏ học (%)"]
                other_cols = [c for c in results_df.columns if c not in priority_cols + ["Predicted_Code", "Predicted_Label", "Predicted_Status", "Prob_Dropout", "Prob_Enrolled", "Prob_Graduate"]]
                
                final_raw_show = results_df[priority_cols + other_cols]
                final_show_df = translate_dataframe_headers(final_raw_show)
                st.dataframe(final_show_df)

                csv_export = final_show_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="TẢI VỀ FILE KẾT QUẢ DỰ ĐOÁN (CSV)",
                    data=csv_export,
                    file_name="ket_qua_du_doan_sinh_vien_viet_hoa.csv",
                    mime="text/csv"
                )

    # =========================================================================
    # TAB 4: QUẢN LÝ DỮ LIỆU & NHẬT KÝ HUẤN LUYỆN (RETRAIN & EXPLAINABILITY)
    # =========================================================================
    with tab4:
        st.subheader("Quản lý Tập dữ liệu Huấn luyện & Huấn luyện lại Mô hình (Retrain AI)")
        st.write("Lựa chọn tập dữ liệu mở rộng để AI học thêm và nâng cao độ chính xác dự đoán:")

        # 1. Danh sách các file dữ liệu có sẵn
        available_datasets = {
            "Tập dữ liệu UCI Gốc (4,424 mẫu)": RAW_DATA_PATH,
            "Tập dữ liệu Sinh viên Mô phỏng (10,000 mẫu)": DATA_DIR / "synthetic_vietnamese_students_10k.csv",
            "Tập dữ liệu Gộp Tổng hợp (14,424 mẫu - Khuyên dùng)": DATA_DIR / "combined_training_data.csv"
        }

        selected_ds_name = st.selectbox(
            "Chọn tập dữ liệu huấn luyện cho mô hình AI:",
            options=list(available_datasets.keys())
        )
        selected_ds_path = available_datasets[selected_ds_name]

        if st.button("KHỞI CHẠY HUẤN LUYỆN LẠI MÔ HÌNH AI (RETRAIN)"):
            with st.spinner(f"Đang huấn luyện mô hình AI trên '{selected_ds_name}'..."):
                try:
                    res_train = train_and_evaluate_models(selected_ds_path)
                    st.success(f"Huấn luyện thành công! Mô hình xuất sắc nhất: {res_train['best_model_name']} - F1-Score: {res_train['best_f1_score']:.4f}")
                    st.cache_resource.clear()  # Xóa cache để nạp lại mô hình mới
                except Exception as err:
                    st.error(f"[LOI] Không thể huấn luyện lại: {err}")

        st.markdown("---")
        st.subheader("1. Lịch sử Tiến hóa & So sánh Hiệu năng Mô hình AI")

        # Nạp nhật ký lịch sử huấn luyện từ JSON
        if HISTORY_FILE_PATH.exists():
            try:
                with open(HISTORY_FILE_PATH, 'r', encoding='utf-8') as f:
                    history_list = json.load(f)
                
                history_df = pd.DataFrame(history_list)
                history_df.rename(columns={
                    "timestamp": "Thời gian Huấn luyện",
                    "dataset_name": "Tên Tập Dữ liệu",
                    "sample_count": "Số lượng Sinh viên",
                    "best_model_name": "Mô hình Xuất sắc nhất",
                    "accuracy": "Độ chính xác Accuracy (%)",
                    "f1_score": "Độ đo F1-Score"
                }, inplace=True)
                
                st.dataframe(history_df.sort_values(by="Thời gian Huấn luyện", ascending=False), use_container_width=True)
            except Exception as e:
                st.info("Chưa có lịch sử huấn luyện.")
        else:
            st.info("Chưa có file nhật ký huấn luyện `training_history.json`.")

        st.markdown("---")
        st.subheader("2. Biểu đồ Giải thích Mô hình: Top Các thuộc tính Quan trọng nhất (Feature Importances)")
        st.caption("Cho biết AI dựa vào các yếu tố học tập/tài chính nào nhất để quyết định khả năng Tốt nghiệp vs Thôi học:")

        # Nạp tầm quan trọng thuộc tính từ JSON
        if IMPORTANCE_FILE_PATH.exists():
            try:
                with open(IMPORTANCE_FILE_PATH, 'r', encoding='utf-8') as f:
                    top_fi = json.load(f)
                
                fi_df = pd.DataFrame(top_fi)
                fi_df["Thuộc tính (Tiếng Việt)"] = fi_df["feature"].map(lambda x: FEATURE_VIETNAMESE_MAP.get(x, x))
                fi_df["Tầm quan trọng (%)"] = (fi_df["importance"] * 100).round(2)
                
                fig_fi, ax_fi = plt.subplots(figsize=(10, 5))
                sns.barplot(
                    data=fi_df.head(10),
                    x="Tầm quan trọng (%)",
                    y="Thuộc tính (Tiếng Việt)",
                    hue="Thuộc tính (Tiếng Việt)",
                    ax=ax_fi,
                    palette="crest",
                    legend=False
                )
                ax_fi.set_title("Top 10 Thuộc tính Ảnh hưởng Tối quan trọng đến Khả năng Tốt nghiệp", fontsize=13, fontweight="bold", pad=12)
                ax_fi.set_xlabel("Tầm quan trọng đóng góp (%)", fontsize=11)
                ax_fi.set_ylabel("Thuộc tính đặc trưng", fontsize=11)
                
                for p in ax_fi.patches:
                    w = p.get_width()
                    ax_fi.annotate(f"{w:.1f}%", (w, p.get_y() + p.get_height()/2.),
                                   ha='left', va='center', fontsize=10, xytext=(3, 0), textcoords='offset points')
                    
                st.pyplot(fig_fi)
                plt.close()
                
                st.info(
                    "GIẢI THÍCH TRỰC QUAN CHO BUỔI VẤN ĐÁP:\n\n"
                    "1. Các chỉ số **'Số môn đỗ Học kỳ 2'** và **'Điểm trung bình Học kỳ 2'** đóng vai trò quyết định nhất (>40% tổng trọng số).\n\n"
                    "2. Yếu tố **'Đã đóng đủ học phí'** và **'Số môn đỗ Học kỳ 1'** xếp ngay sau đó, chứng minh tình trạng tài chính và kết quả năm đầu tiên là căn cứ dự đoán bỏ học chính xác nhất."
                )

            except Exception as e:
                st.warning(f"Chưa thể hiển thị biểu đồ Feature Importances: {e}")

if __name__ == "__main__":
    main()
