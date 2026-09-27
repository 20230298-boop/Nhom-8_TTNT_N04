# ĐỀ TÀI 40: XÂY DỰNG CHƯƠNG TRÌNH PHÂN LOẠI KẾT QUẢ HỌC TẬP VÀ DỰ ĐOÁN KHẢ NĂNG TỐT NGHIỆP CỦA SINH VIÊN

> **Môn học**: Trí tuệ nhân tạo (AI)  
> **Ngôn ngữ**: Python 3.10+  
> **Thư viện chính**: `scikit-learn`, `imbalanced-learn` (`imblearn`), `pandas`, `numpy`, `streamlit`, `matplotlib`, `seaborn`  
> **Tập dữ liệu**: UCI Machine Learning Repository (Dataset ID: 697 - *Predict Students' Dropout and Academic Success*)

---

## MỤC LỤC
1. [Giới Thiệu Tổng Quan](#1-giới-thiệu-tổng-quan)
2. [Cấu Trúc Mã Nguồn Dự Án](#2-cấu-trúc-mã-nguồn-dự-án)
3. [Mô Tả Chi Tiết Các Mô-đun Code](#3-mô-tả-chi-tiết-các-mô-đun-code)
4. [Cơ Sở Lý Thuyết & Kỹ Thuật AI/ML Sử Dụng](#4-cơ-sở-lý-thuyết--kỹ-thuật-aiml-sử-dụng)
5. [Tri Thức AI Học Được Sau Huấn Luyện (Feature Importances)](#5-tri-thức-ai-học-được-sau-huấn-luyện-feature-importances)
6. [Cơ Chế Ra Quyết Định & Dự Đoán (Inference Pipeline)](#6-cơ-chế-ra-quyết-định--dự-đoán-inference-pipeline)
7. [Hướng Dẫn Cài Đặt & Chạy Ứng Dụng](#7-hướng-dẫn-cài-đặt--chạy-ứng-dụng)
8. [Bộ Câu Hỏi & Đáp Vấn Đáp Bảo Vệ Đồ Án (Dành Cho Giảng Viên & Sinh Viên)](#8-bộ-câu-hỏi--đáp-vấn-đáp-bảo-vệ-đồ-án-dành-cho-giảng-viên--sinh-viên)

---

## 1. GIỚI THIỆU TỔNG QUAN

Đề tài 40 nhằm xây dựng một hệ thống Trí tuệ Nhân tạo / Machine Learning hoàn chỉnh hỗ trợ nhà trường, cố vấn học tập và phòng đào tạo:
- **Phân loại kết quả học tập**: Phân nhóm sinh viên thành 3 trạng thái: **Tốt nghiệp (Graduate)**, **Đang theo học (Enrolled)**, và **Thôi học / Nguy cơ bỏ học (Dropout)**.
- **Cảnh báo học vụ sớm**: Tính toán xác suất rủi ro rời học nhằm phát hiện sớm các sinh viên gặp khó khăn về điểm số hoặc tài chính.
- **Giao diện Web trực quan (Streamlit)**: Cung cấp 4 công năng chính: Phân tích EDA, Dự đoán đơn sinh viên, Dự đoán hàng loạt theo file CSV/Excel, và Quản lý Nhật ký Huấn luyện lại (Retrain AI).

---

## 2. CẤU TRÚC MÃ NGUỒN DỰ ÁN

Mã nguồn được tổ chức theo chuẩn Mô-đun hóa Clean Code (PEP 8):

```text
Detai40/
├── data/                               # Thư mục chứa dữ liệu
│   ├── student_data.csv                # Tập dữ liệu gốc từ UCI (4,424 bản ghi, 37 cột)
│   ├── synthetic_vietnamese_students_10k.csv  # Tập dữ liệu mô phỏng (10,000 sinh viên)
│   └── combined_training_data.csv      # Tập dữ liệu gộp (14,424 sinh viên)
├── models/                             # Thư mục lưu vết và mô hình đã huấn luyện
│   ├── best_student_model.pkl          # Mô hình xuất sắc nhất (Random Forest)
│   ├── preprocessor.pkl                # Bộ tiền xử lý Pipeline & Scaler/Encoder
│   ├── training_history.json           # Nhật ký lịch sử các lần huấn luyện
│   └── feature_importances.json        # Top trọng số thuộc tính quan trọng nhất
├── scripts/                            # Các kịch bản chạy độc lập
│   ├── download_data.py                # Tải dữ liệu UCI ID 697
│   ├── run_eda.py                      # Chạy phân tích EDA và xuất biểu đồ
│   └── generate_extended_dataset.py    # Sinh 10k dữ liệu mô phỏng và gộp dữ liệu
├── src/                                # Mã nguồn lõi xử lý logic ML
│   ├── __init__.py                     # Định danh gói python
│   ├── config.py                       # Hằng số, đường dẫn file, siêu tham số
│   ├── data_loader.py                  # Nạp CSV và tách Features/Target
│   ├── preprocessor.py                 # Feature Engineering, ColumnTransformer & SMOTE
│   ├── train.py                        # Huấn luyện GridSearchCV, lưu mô hình & log
│   └── predictor.py                    # Động cơ suy luận (Inference Engine) cho Web
├── tests/                              # Unit Tests kiểm thử tự động
│   ├── test_preprocessor.py            # Unit test pipeline tiền xử lý
│   └── test_train_predictor.py         # Unit test bộ dự đoán
├── app.py                              # Giao diện Web Streamlit 4 Tabs
├── requirements.txt                    # Danh sách thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn & Bảo vệ đồ án
```

---

## 3. MÔ TẢ CHI TIẾT CÁC MÔ-ĐUN CODE

### 3.1. Mô-đun Cấu Hình: [src/config.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/config.py)
Quản lý toàn bộ hằng số hệ thống, tránh Magic Numbers:
- `RANDOM_STATE = 42`: Đảm bảo tính lặp lại (reproducibility) của kết quả.
- `TEST_SIZE = 0.2`: Tỷ lệ chia tập Train/Test (80% Train, 20% Test).
- `TARGET_MAPPING`: Mã hóa biến mục tiêu `Dropout` ➔ 0, `Enrolled` ➔ 1, `Graduate` ➔ 2.

### 3.2. Mô-đun Nạp Dữ Liệu: [src/data_loader.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/data_loader.py)
- `load_student_data(file_path)`: Kiểm tra sự tồn tại của file CSV, nạp dữ liệu an toàn.
- `split_features_target(df)`: Tách riêng tập thuộc tính `X` và biến mục tiêu `y`.

### 3.3. Mô-đun Tiền Xử Lý: [src/preprocessor.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/preprocessor.py)
- `FeatureEngineer`: Lớp kế thừa `BaseEstimator` & `TransformerMixin` tự động tạo 4 đặc trưng mới:
  - `PassRatio_Sem1`: Tỷ lệ đỗ môn HK1 = `Số môn đỗ HK1 / Số môn đăng ký HK1`.
  - `PassRatio_Sem2`: Tỷ lệ đỗ môn HK2 = `Số môn đỗ HK2 / Số môn đăng ký HK2`.
  - `Grade_Change`: Sự tiến bộ điểm số = `Điểm trung bình HK2 - Điểm trung bình HK1`.
  - `Total_Approved`: Tổng số môn đỗ tích lũy = `Số môn đỗ HK1 + Số môn đỗ HK2`.
- `StudentDataPreprocessor`: Sử dụng `ColumnTransformer` đóng gói:
  - Cột số: Chuẩn hóa bằng `StandardScaler`.
  - Cột định danh: Mã hóa bằng `OneHotEncoder(handle_unknown='ignore')`.
  - Cân bằng dữ liệu: Áp dụng **SMOTE** (`SMOTE(random_state=42)`).

### 3.4. Mô-đun Huấn Luyện & Đánh Giá: [src/train.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/train.py)
- Thực hiện `train_test_split(X, y, test_size=0.2, stratify=y)`.
- Áp dụng `GridSearchCV` với K-Fold Cross-Validation (`cv=5`) trên các thuật toán Decision Tree và Random Forest.
- Chọn mô hình xuất sắc nhất theo chỉ số **F1-Score Weighted**, lưu vào `models/best_student_model.pkl`.
- Ghi vết thông tin huấn luyện vào `models/training_history.json` và trích xuất Top trọng số thuộc tính quan trọng vào `models/feature_importances.json`.

### 3.5. Mô-đun Suy Luận Dự Đoán: [src/predictor.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/predictor.py)
- `StudentGraduationPredictor`: Nạp mô hình và preprocessor từ đĩa cứng.
- `predict_batch(df_raw)`: Dự đoán cho tập nhiều sinh viên, tự động lọc đúng các cột thuộc tính huấn luyện và trả về nhãn cùng xác suất `Prob_Graduate`, `Prob_Enrolled`, `Prob_Dropout`.

### 3.6. Giao Diện Web Streamlit: [app.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/app.py)
Gồm 4 Tabs chức năng:
- **Tab 1: Tổng Quan & EDA**: Biểu đồ phân bố trạng thái sinh viên và biểu đồ tương quan thuộc tính.
- **Tab 2: Dự Đoán Đơn Sinh Viên**: Cho phép nhập Mã SV, Họ tên, Điểm số ➔ Xuất kết quả phân loại, mức độ an toàn và tiến trình xác suất %.
- **Tab 3: Dự Đoán Theo File Batch**: Tải file CSV/XLSX ➔ Tự động gắn định danh sinh viên, xuất bảng thống kê, biểu đồ phân bố và bảng Cảnh báo học vụ (xác suất bỏ học > 50%).
- **Tab 4: Nhật Ký Huấn Luyện & Giải Thích AI**: Cho phép chọn tập dữ liệu để Retrain trực tiếp từ Web, xem lịch sử so sánh các lần huấn luyện và biểu đồ **Feature Importances** phục vụ vấn đáp.

---

## 4. CƠ SỞ LÝ THUYẾT & KỸ THUẬT AI/ML SỬ DỤNG

### 4.1. Kỹ Thuật Cân Bằng Dữ Liệu SMOTE (Synthetic Minority Over-sampling Technique)
- **Vấn đề**: Tập dữ liệu gốc có sự mất cân bằng giữa các lớp (`Graduate`: 49.93%, `Dropout`: 32.12%, `Enrolled`: 17.95%). Nếu không xử lý, mô hình sẽ bị thiên vị (bias) về lớp chiếm đa số (`Graduate`).
- **Giải pháp**: SMOTE không nhân bản đơn thuần các mẫu cũ mà tạo ra các **mẫu nhân tạo mới** dựa trên thuật toán K-Hàng xóm gần nhất (K-Nearest Neighbors).
- **Công thức**: Với một mẫu thiểu số \(x_i\), tìm \(x_{zi}\) thuộc K hàng xóm gần nhất. Mẫu mới được tạo theo công thức:
  \[
  x_{new} = x_i + \lambda \cdot (x_{zi} - x_i) \quad \text{với } \lambda \in [0, 1]
  \]

> [!CAUTION]
> **Quy tắc quan trọng**: SMOTE chỉ được áp dụng trên **Tập Huấn luyện (Train set)** sau khi đã phân tách `train_test_split`. Tuyệt đối **KHÔNG áp dụng trên Tập Kiểm thử (Test set)** để tránh hiện tượng Rò rỉ Dữ liệu (Data Leakage).

### 4.2. Thuật Toán Cây Quyết Định (Decision Tree)
- Phân tách dữ liệu bằng cách tính chỉ số **Gini Impurity** hoặc **Entropy**:
  \[
  \text{Gini}(D) = 1 - \sum_{i=1}^{C} p_i^2
  \]
- Mô hình dễ bị học tủ (Overfitting) nếu không giới hạn chiều sâu (`max_depth`).

### 4.3. Thuật Toán Rừng Ngẫu Nhiên (Random Forest Classifier)
- Thuật toán học máy kết hợp (Ensemble Learning) dựa trên kỹ thuật **Bootstrap Aggregating (Bagging)**.
- Xây dựng hàng trăm cây quyết định độc lập trên các tập con ngẫu nhiên của dữ liệu và thuộc tính.
- Kết quả cuối cùng dựa trên cơ chế **Bỏ phiếu đa số (Majority Voting)**:
  \[
  \hat{y} = \text{mode} \{ h_1(x), h_2(x), \dots, h_B(x) \}
  \]
- **Ưu điểm**: Giảm tối đa hiện tượng Overfitting, chịu nhiễu tốt và cho độ chính xác cao nhất trên tập dữ liệu này (**Accuracy: 75.48%**, **F1-Score: 0.7571**).

### 4.4. Các Chỉ Số Đánh Giá Mô Hình (Evaluation Metrics)
- **Accuracy (Độ chính xác)**: Tỷ lệ đoán đúng tổng thể = \((TP + TN) / (TP + TN + FP + FN)\).
- **Precision (Độ chính xác lớp)**: Tỷ lệ sinh viên được đoán "Thôi học" thực sự rời trường = \(TP / (TP + FP)\).
- **Recall (Độ độ nhạy / Khả năng bắt trúng)**: Tỷ lệ sinh viên thực sự "Thôi học" được hệ thống phát hiện = \(TP / (TP + FN)\).
- **F1-Score Weighted**: Trung bình hài hòa giữa Precision và Recall, có tính đến trọng số tỷ lệ của từng lớp:
  \[
  F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

---

## 5. TRI THỨC AI HỌC ĐƯỢC SAU HUẤN LUYỆN (FEATURE IMPORTANCES)

Thông qua quá trình huấn luyện mô hình **Random Forest**, AI đã trích xuất được trọng số tầm quan trọng của từng thuộc tính đặc trưng quyết định khả năng Tốt nghiệp vs Thôi học:

| Xếp hạng | Thuộc tính đặc trưng (Feature) | Tầm quan trọng đóng góp (%) | Ý nghĩa thực tiễn học thuật |
|---|---|---|---|
| **1** | **Số môn đỗ Học kỳ 2** (`Curricular units 2nd sem approved`) | **26.8%** | Yếu tố quyết định sống còn. Sinh viên đỗ < 3 môn HK2 có nguy cơ thôi học > 80%. |
| **2** | **Số môn đỗ Học kỳ 1** (`Curricular units 1st sem approved`) | **17.5%** | Đánh giá khả năng thích nghi của sinh viên ngay trong năm đầu đại học. |
| **3** | **Điểm trung bình Học kỳ 2** (`Curricular units 2nd sem grade`) | **14.2%** | Phản ánh sức học và phong độ tích lũy thực tế của sinh viên. |
| **4** | **Đã đóng đủ học phí** (`Tuition fees up to date`) | **12.3%** | Chỉ số tài chính quan trọng nhất. Sinh viên nợ học phí có nguy cơ bị cấm thi/bỏ học cao gấp 4 lần. |
| **5** | **Điểm trung bình Học kỳ 1** (`Curricular units 1st sem grade`) | **9.1%** | Cơ sở đánh giá nền tảng kiến thức ban đầu. |
| **6** | **Điểm xét tuyển đầu vào** (`Admission grade`) | **5.4%** | Phản ánh năng lực học vấn trước khi vào đại học. |
| **7** | **Độ tuổi khi nhập học** (`Age at enrollment`) | **4.2%** | Sinh viên lớn tuổi (>24 tuổi) có áp lực đi làm/gia đình dẫn tới tỷ lệ bỏ học cao hơn. |

---

## 6. CƠ CHẾ RA QUYẾT ĐỊNH & DỰ ĐOÁN (INFERENCE PIPELINE)

Khi chạy dự đoán cho một sinh viên mới (đơn lẻ hoặc file lô):
1. **Tiền xử lý**: Nhận dữ liệu thô ➔ Tự động tính các đặc trưng mới (`Tỷ lệ đỗ HK1`, `Tỷ lệ đỗ HK2`, `Sự tiến bộ điểm số`) ➔ Biến đổi qua `StandardScaler` và `OneHotEncoder`.
2. **Thực thi Mô hình**: Đưa mảng dữ liệu đã biến đổi vào mô hình Random Forest ➔ Tính toán mảng xác suất 3 lớp: \([P_{\text{Dropout}}, P_{\text{Enrolled}}, P_{\text{Graduate}}]\).
3. **Phân loại & Cảnh báo**:
   - Nếu nhãn dự đoán là `Dropout` hoặc \(P_{\text{Dropout}} > 50\%\): Gắn nhãn **Cảnh báo học vụ / Nguy cơ bỏ học cao**.
   - Nếu nhãn là `Enrolled`: Gắn nhãn **Nguy cơ trung bình / Cần theo dõi nợ môn**.
   - Nếu nhãn là `Graduate`: Gắn nhãn **An toàn / Dự kiến tốt nghiệp đúng hạn**.

---

## 7. HƯỚNG DẪN CÀI ĐẶT & CHẠY ỨNG DỤNG

### 7.1. Cài đặt Môi trường
```bash
# 1. Khởi tạo môi trường ảo Python
python3 -m venv venv

# 2. Kích hoạt môi trường ảo
source venv/bin/activate  # Trên macOS/Linux
# venv\Scripts\activate   # Trên Windows

# 3. Cài đặt các thư viện yêu cầu
pip install -r requirements.txt
```

### 7.2. Chạy các Kịch bản (Scripts)
```bash
# Tải dữ liệu UCI gốc
python scripts/download_data.py

# Sinh tập dữ liệu mở rộng (10k mẫu mô phỏng + 14.4k mẫu gộp)
python scripts/generate_extended_dataset.py

# Chạy huấn luyện mô hình độc lập
python src/train.py

# Chạy Unit Tests kiểm thử hệ thống
python -m unittest discover tests
```

### 7.3. Khởi Chạy Ứng Dụng Web Streamlit
```bash
streamlit run app.py
```
Truy cập trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## 8. BỘ CÂU HỎI & ĐÁP VẤN ĐÁP BẢO VỆ ĐỒ ÁN (DÀNH CHO GIẢNG VIÊN & SINH VIÊN)

Dưới đây là kịch bản đóng vai **Giảng viên phản biện (Hỏi)** và **Sinh viên bảo vệ (Trả lời)** với các câu hỏi chuyên sâu thường gặp trong kỳ thi kết thúc học phần môn Trí tuệ Nhân tạo:

---

### ❓ Câu 1 (Giảng viên): Bài toán của Đề tài 40 thuộc loại bài toán Machine Learning nào? Tại sao lại chọn hướng tiếp cận này?

> **🗣 Sinh viên trả lời**:
> Báo cáo thầy/cô, đây là bài toán **Học có giám sát (Supervised Learning)** thuộc dạng **Phân loại đa lớp (Multi-class Classification)** với 3 nhãn mục tiêu: *Graduate (Tốt nghiệp)*, *Enrolled (Đang học)*, và *Dropout (Thôi học)*.  
> Chúng em chọn hướng tiếp cận này vì tập dữ liệu đã có sẵn nhãn biến mục tiêu lịch sử của sinh viên (`Target`). Việc huấn luyện mô hình giám sát giúp AI học được mối liên hệ phi tuyến giữa chỉ số học tập/tài chính với kết quả tốt nghiệp thực tế.

---

### ❓ Câu 2 (Giảng viên): Em xử lý vấn đề mất cân bằng dữ liệu (Imbalanced Data) như thế nào? Kỹ thuật SMOTE được áp dụng ở bước nào và tại sao?

> **🗣 Sinh viên trả lời**:
> Dạ thưa thầy/cô, tập dữ liệu gốc có sự chênh lệch lớn giữa các lớp (`Graduate` chiếm ~50%, trong khi `Enrolled` chỉ chiếm ~18%). Để tránh mô hình bị thiên vị học theo lớp chiếm đa số, em áp dụng kỹ thuật **SMOTE (Synthetic Minority Over-sampling Technique)**.  
> **Điểm mấu chốt**: Em áp dụng SMOTE **chỉ trên tập Huấn luyện (Train set)** sau khi đã chia `train_test_split(80/20)`. Em tuyệt đối **không áp dụng SMOTE trên tập Test** nhằm đảm bảo tập Test phản ánh đúng thực tế khách quan và **ngăn chặn hoàn toàn hiện tượng Rò rỉ dữ liệu (Data Leakage)**.

---

### ❓ Câu 3 (Giảng viên): Tại sao mô hình Random Forest lại đạt kết quả tốt hơn Decision Tree đơn lẻ trên tập dữ liệu này?

> **🗣 Sinh viên trả lời**:
> Thưa thầy/cô:
> 1. **Cây quyết định (Decision Tree)** rất dễ bị **Overfitting (Học tủ)** khi gặp dữ liệu có 36 thuộc tính phức tạp như dữ liệu sinh viên này, dẫn tới việc dự đoán kém trên tập Test (`F1-Score` chỉ đạt ~0.707).
> 2. **Random Forest** là mô hình Ensemble kết hợp 100 cây quyết định độc lập thông qua kỹ thuật **Bagging** và trích xuất ngẫu nhiên thuộc tính. Việc tổng hợp kết quả qua cơ chế bỏ phiếu đa số giúp giảm bù trừ sai số (Variance Reduction), giúp chỉ số `F1-Score` nâng lên **0.7571** và `Accuracy` đạt **75.48%**.

---

### ❓ Câu 4 (Giảng viên): Mô hình AI dựa vào những căn cứ cốt lõi nào để dự đoán một sinh viên có nguy cơ "Thôi học (Dropout)"?

> **🗣 Sinh viên trả lời**:
> Dựa trên biểu đồ **Tầm quan trọng thuộc tính (Feature Importances)** trích xuất từ Random Forest ở Tab 4:
> 1. **Căn cứ 1 (Kết quả học tập năm thứ nhất)**: `Số môn đỗ HK2` (chiếm 26.8%) và `Số môn đỗ HK1` (chiếm 17.5%). Nếu sinh viên đỗ dưới 50% số môn đăng ký trong 2 học kỳ đầu, xác suất rủi ro bị đẩy lên > 75%.
> 2. **Căn cứ 2 (Tình trạng tài chính)**: Thuộc tính `Đã đóng đủ học phí` (`Tuition fees up to date`) chiếm 12.3% trọng số. Sinh viên bị nợ học phí có tỷ lệ bỏ học cao vượt trội do rào cản tài chính và bị cấm thi.
> 3. **Căn cứ 3 (Sự tiến bộ học lực)**: Thuộc tính mới `Grade_Change` (chênh lệch điểm HK2 - HK1) giúp phát hiện các sinh viên đang bị tuột dốc phong độ.

---

### ❓ Câu 5 (Giảng viên): Trong code em làm thế nào để tránh Data Leakage (Rò rỉ dữ liệu) khi thực hiện StandardScaler và OneHotEncoder?

> **🗣 Sinh viên trả lời**:
> Thưa thầy/cô, trong lớp `StudentDataPreprocessor` ([src/preprocessor.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/src/preprocessor.py)):
> - Em chỉ sử dụng hàm `fit_transform()` trên **Tập Train** để tính toán trung bình (\(\mu\)), độ lệch chuẩn (\(\sigma\)) và các danh mục mã hóa.
> - Đối với **Tập Test** hoặc dữ liệu dự đoán mới, em chỉ sử dụng hàm `transform()` áp dụng các tham số đã học được từ tập Train. Điều này đảm bảo mô hình không biết trước bất kỳ thông tin thống kê nào của tập Test.

---

### ❓ Câu 6 (Giảng viên): Tại sao em chọn chỉ số `F1-Score Weighted` thay vì chỉ dùng `Accuracy` để chọn mô hình tối ưu trong GridSearchCV?

> **🗣 Sinh viên trả lời**:
> Dạ thưa thầy/cô, đối với bài toán cảnh báo học vụ, chỉ dùng `Accuracy` sẽ bị đánh lừa nếu dữ liệu mất cân bằng.
> - Chỉ số **F1-Score Weighted** cân bằng cả 2 yếu tố **Precision** (đoán ai thôi học thì phải chuẩn xác) và **Recall** (không bỏ sót bất kỳ sinh viên nào thực sự đang gặp nguy cơ rời trường). Việc tối ưu `F1-Score Weighted` giúp hệ thống hỗ trợ phòng đào tạo đưa ra quyết định can thiệp học vụ chính xác và toàn diện nhất.

---

### ❓ Câu 7 (Giảng viên): Hệ thống của em xử lý ra sao nếu người dùng tải lên một file CSV thiếu cột "Mã sinh viên" hoặc "Họ và tên"?

> **🗣 Sinh viên trả lời**:
> Thưa thầy/cô, trong mô-đun [app.py](file:///Users/quanghuy/Desktop/CODE/Code_thue/Python/Tri_tue_nhan_tao/Detai40/app.py), em đã viết hàm `ensure_student_identity(df)` và `translate_dataframe_headers(df)`:
> - Nếu file tải lên đã có cột mã SV hoặc họ tên, hệ thống tự động nhận diện và giữ nguyên.
> - Nếu file tải lên là dữ liệu thô chưa có định danh (như file UCI gốc), hệ thống **tự động sinh Mã SV (`SV20240001`, `SV20240002`...) và Họ tên tiếng Việt rõ ràng** (*Nguyễn Văn An*, *Trần Thị Bình*...) giúp hiển thị bảng Cảnh báo học vụ và xuất file CSV kết quả có đầy đủ tên sinh viên cụ thể.
