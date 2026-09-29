# Video Game Sales Dashboard - Streamlit

## Cách chạy
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cấu trúc
- `app.py` — Dashboard chính (4 tab)
- `ADY201m_enhanced_v2.xlsx` — Dataset 74 cột
- `models/best_model.joblib` — Mô hình XGBoost tuned đã train
- `models/*.csv` — Kết quả CV, tuning, test, feature importance, error analysis

## Tính năng
1. **Tab RQ1:** Genre/Platform/Region impact + Feature Importance + bộ lọc
2. **Tab RQ2:** Bảng so sánh 7 mô hình + CV + tuning log
3. **Tab RQ3:** Heatmap Genre×ConsoleGen + xu hướng theo năm + sai số theo genre
4. **Tab Dự đoán:** Nhập thông tin game → dự đoán doanh số + khoảng tin cậy + cảnh báo ngoài phạm vi train
