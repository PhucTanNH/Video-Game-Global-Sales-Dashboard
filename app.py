# ============================================================
# DASHBOARD STREAMLIT - Video Game Global Sales Prediction
# Chay: streamlit run app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings('ignore')

# ---------- Cau hinh trang ----------
st.set_page_config(page_title="Video Game Sales Dashboard", layout="wide", page_icon="🎮")

# ---------- Load data & model (cache de nhanh) ----------
@st.cache_data
def load_data():
    return pd.read_excel('ADY201m_enhanced_v2.xlsx')

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.joblib')

df = load_data()
model = load_model()

# Feature dung de train model (phai dung thu tu)
CAT = ['Genre','Platform','Publisher_Tier','Platform_Manufacturer','Platform_Lifecycle_Stage','ConsoleGen']
NUM = ['Year','Is_Sequel','Is_Multiplatform','Num_Platforms','Prev_Franchise_MaxSales','Prev_Franchise_TitleCount',
       'Pub_Prev3Y_AvgSales','Pub_Prev3Y_Count','Genre_Prev3Y_AvgSales','Genre_Prev3Y_Count',
       'Platform_Prev3Y_AvgSales','Platform_Prev3Y_Count','Years_Since_Platform_Launch','Genre_vs_Market_Avg',
       'Publisher_MarketShare_Prev3Y','Genre_Trend_Momentum','PubGenre_Specialization','GenrePlatform_Fit',
       'SameGenre_Releases_PrevYear']

# Pham vi train de canh bao ngoai vung
TRAIN = df[df['Year']<=2012]
train_ranges = {c: (TRAIN[c].min(), TRAIN[c].max()) for c in NUM if c in TRAIN.columns}

# ---------- Sidebar: Bo loc ----------
st.sidebar.title("🎮 Bộ lọc")
genre_filter = st.sidebar.multiselect("Thể loại (Genre)", sorted(df['Genre'].unique()), default=[])
platform_filter = st.sidebar.multiselect("Nền tảng (Platform)", sorted(df['Platform'].unique()), default=[])
year_range = st.sidebar.slider("Khoảng năm", int(df['Year'].min()), int(df['Year'].max()), (2000, 2016))
tier_filter = st.sidebar.selectbox("Quy mô Publisher", ["Tất cả","Major","Mid","Small"])

df_filtered = df.copy()
if genre_filter: df_filtered = df_filtered[df_filtered['Genre'].isin(genre_filter)]
if platform_filter: df_filtered = df_filtered[df_filtered['Platform'].isin(platform_filter)]
df_filtered = df_filtered[(df_filtered['Year']>=year_range[0]) & (df_filtered['Year']<=year_range[1])]
if tier_filter != "Tất cả": df_filtered = df_filtered[df_filtered['Publisher_Tier']==tier_filter]

st.sidebar.markdown(f"**Số game hiển thị:** {len(df_filtered):,} / {len(df):,}")

# ---------- Tieu de ----------
st.title("Video Game Global Sales Prediction Dashboard")
st.markdown("*Dự án ADY201m — Leakage-free prediction with 63 engineered features | Model: XGBoost tuned (RMSE=1.01M, R²=0.33) | Best test: LightGBM RMSE=0.947M, R²=0.42*")

tab1, tab2, tab3, tab4 = st.tabs(["📊 RQ1: Yếu tố ảnh hưởng", "🤖 RQ2: So sánh mô hình", "🎯 RQ3: Thế hệ console", "🔮 Dự đoán doanh số"])

# ============================================================
# TAB 1: RQ1 - Genre, Platform, Region anh huong
# ============================================================
with tab1:
    st.header("RQ1: Thể loại, nền tảng và khu vực nào ảnh hưởng nhiều nhất đến doanh số?")
    st.success("Kết luận: Lịch sử franchise + uy tín Publisher ảnh hưởng mạnh nhất; Platform ở mức trung bình; Genre trực tiếp yếu hơn kỳ vọng (đã kiểm soát Publisher & Franchise).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Doanh số trung bình theo thể loại")
        g_sales = df_filtered.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=True)
        fig = px.bar(x=g_sales.values, y=g_sales.index, orientation='h',
                     labels={'x':'Avg Global Sales (triệu bản)','y':'Genre'},
                     color=g_sales.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 nền tảng theo doanh số TB")
        p_sales = df_filtered.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=p_sales.index, y=p_sales.values,
                     labels={'x':'Platform','y':'Avg Global Sales (M)'}, color=p_sales.values)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tỷ lệ doanh số theo khu vực & thể loại")
    region = df_filtered.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
    region_pct = region.div(region.sum(axis=1), axis=0)
    fig = px.bar(region_pct, barmode='stack', color_discrete_sequence=['#3182ce','#38a169','#e53e3e','#d69e2e'],
                 labels={'value':'Tỷ lệ','variable':'Khu vực'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='Tỷ lệ doanh số')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Insight: Role-Playing có 35% doanh số từ Nhật (cao bất thường), trong khi Shooter tập trung >70% ở Bắc Mỹ + Châu Âu.")

    st.subheader("Top 10 feature quan trọng nhất (Permutation Importance)")
    imp = pd.read_csv('models/permutation_importance.csv', index_col=0).iloc[:,0].sort_values(ascending=True).tail(10)
    fig = px.bar(x=imp.values, y=imp.index, orientation='h',
                 labels={'x':'Mức tăng RMSE khi xáo trộn feature','y':'Feature'},
                 title="Feature Importance (XGBoost tuned)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: RQ2 - So sanh mo hinh
# ============================================================
with tab2:
    st.header("RQ2: Mô hình nào dự đoán doanh số chính xác nhất?")
    st.success("Kết luận: Gradient boosting trees (LightGBM/XGBoost/Random Forest) vượt xa mô hình tuyến tính. LightGBM tốt nhất test: RMSE=0.947M, R²=0.42 — vượt baseline bài báo Zhan 2026 (R²≈0.384).")

    results = pd.DataFrame({
        'Model': ['Linear (baseline)','Median (baseline)','kNN','Random Forest','XGBoost','LightGBM','XGBoost (tuned)'],
        'RMSE (M)': [2.037, 1.280, 1.074, 0.983, 1.025, 0.947, 1.015],
        'MAE (M)': [0.479, 0.457, 0.425, 0.418, 0.392, 0.360, 0.385],
        'R²': [-1.70, -0.07, 0.25, 0.37, 0.32, 0.42, 0.33],
    })
    st.dataframe(results.style.format({'RMSE (M)':'{:.3f}','MAE (M)':'{:.3f}','R²':'{:.2f}'})
                 .highlight_min(subset=['RMSE (M)','MAE (M)'], color='#c6f6d5')
                 .highlight_max(subset=['R²'], color='#c6f6d5'), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results, x='Model', y='RMSE (M)', color='RMSE (M)', color_continuous_scale='RdYlGn_r',
                     title="So sánh RMSE (thấp = tốt)")
        fig.add_hline(y=1.280, line_dash="dash", annotation_text="Median baseline")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(results, x='Model', y='R²', color='R²', color_continuous_scale='RdYlGn',
                     title="So sánh R² (cao = tốt)")
        fig.add_hline(y=0.384, line_dash="dash", annotation_text="Zhan 2026 baseline (R²≈0.384)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Kết quả Cross-Validation (TimeSeriesSplit 5 folds)")
    cv = pd.read_csv('models/cv_results.csv')
    st.dataframe(cv, use_container_width=True)
    st.caption("CV = TimeSeriesSplit(5) trên train ≤2012. Chọn XGBoost theo CV (đúng quy trình, không nhìn vào test).")

    st.subheader("Log tinh chỉnh (tuning)")
    tuning = pd.read_csv('models/tuning_log.csv')
    st.dataframe(tuning, use_container_width=True)

# ============================================================
# TAB 3: RQ3 - The he console
# ============================================================
with tab3:
    st.header("RQ3: Xu hướng doanh số theo thể loại có thay đổi qua các thế hệ console không?")
    st.success("Kết luận: Có — Role-Playing giảm từ 0.95M (Gen 5) xuống 0.55M (Gen 8), Shooter tăng từ 0.5M lên 1.1M. Bỏ feature thế hệ → RMSE tăng ~7.7% (0.947→1.02M).")

    pivot = df_filtered.pivot_table(index='Genre', columns='ConsoleGen', values='Global_Sales', aggfunc='mean')
    col_order = [c for c in ['Other','Gen_5','Gen_6','Gen_7','Gen_8'] if c in pivot.columns]
    pivot = pivot[col_order]
    fig = px.imshow(pivot, text_auto='.2f', color_continuous_scale='YlOrRd',
                    labels={'x':'Console Generation','y':'Genre','color':'Avg Sales (M)'},
                    title="Heatmap: Doanh số TB theo Genre × Console Generation")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Xu hướng doanh số theo năm (theo thể loại rộng)")
    yearly_genre = df_filtered.groupby(['Year','Genre_Broad'])['Global_Sales'].mean().reset_index()
    fig = px.line(yearly_genre, x='Year', y='Global_Sales', color='Genre_Broad', markers=True,
                  labels={'Global_Sales':'Avg Sales (M)','Genre_Broad':'Nhóm thể loại'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Insight: Nhóm Action (Shooter/Fighting) tăng mạnh từ Gen 6→Gen 7; RPG/Strategy suy giảm sau Gen 6; Sports/Racing ổn định.")

    st.subheader("Phân tích sai số theo thể loại (test set)")
    err = pd.read_csv('models/error_by_genre.csv', index_col=0).sort_values('mean', ascending=True)
    fig = px.bar(x=err['mean'], y=err.index, orientation='h',
                 labels={'x':'MAE (triệu bản)','y':'Genre'}, title="Sai số trung bình theo thể loại",
                 text=err['mean'].round(2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shooter (0.93M) và Sports (0.54M) khó dự đoán nhất (có hit lớn bất thường như Call of Duty, FIFA).")

# ============================================================
# TAB 4: Du doan doanh so
# ============================================================
with tab4:
    st.header("🔮 Công cụ dự đoán doanh số game mới")
    st.markdown("Nhập thông tin game → mô hình dự đoán doanh số toàn cầu (triệu bản). Model: **XGBoost tuned** (RMSE≈1.0M trên test).")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Năm phát hành", min_value=1995, max_value=2025, value=2015, step=1)
        genre = st.selectbox("Thể loại (Genre)", sorted(df['Genre'].unique()))
        platform = st.selectbox("Nền tảng (Platform)", sorted(df['Platform'].unique()))
        publisher = st.selectbox("Nhà phát hành (Publisher)", sorted(df['Publisher'].unique()))
        num_platforms = st.slider("Số nền tảng phát hành", 1, 10, 1)

    with col2:
        is_sequel = st.checkbox("Là phần tiếp theo (Sequel)?", value=False)
        franchise_max = st.number_input("Doanh số cao nhất của franchise trước đó (M)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        franchise_count = st.number_input("Số tựa game trước trong franchise", min_value=0, max_value=50, value=0, step=1)

    # --- Tu dong tinh rolling features tu du lieu lich su ---
    def lookup(group_col, val, year, col_prefix):
        """Tra cuu feature rolling 3 nam truoc cho mot nhom."""
        sub = df[(df[group_col]==val) & (df['Year']<year) & (df['Year']>=year-3)]
        if len(sub)==0:
            sub = df[df[group_col]==val]  # fallback: toan bo lich su
        return {
            f'{col_prefix}_Prev3Y_AvgSales': sub['Global_Sales'].mean() if len(sub)>0 else df['Global_Sales'].mean(),
            f'{col_prefix}_Prev3Y_Count': len(sub),
        }

    pub_feat  = lookup('Publisher', publisher, year, 'Pub')
    gen_feat  = lookup('Genre', genre, year, 'Genre')
    plat_feat = lookup('Platform', platform, year, 'Platform')

    # Platform lifecycle
    plat_launch = df[df['Platform']==platform]['Platform_Launch_Year'].median()
    years_since = year - plat_launch if pd.notna(plat_launch) else 5
    lifecycle = 'Early' if years_since<=2 else ('Mid' if years_since<=5 else 'Late')
    if platform=='PC': lifecycle = 'PC/Unknown'

    # Console gen
    console_gen = df[df['Platform']==platform]['ConsoleGen'].mode()
    console_gen = console_gen.iloc[0] if len(console_gen)>0 else 'Gen_7'

    # Manufacturer & tier
    manuf = df[df['Platform']==platform]['Platform_Manufacturer'].mode()
    manuf = manuf.iloc[0] if len(manuf)>0 else 'Other'
    pub_tier = df[df['Publisher']==publisher]['Publisher_Tier'].mode()
    pub_tier = pub_tier.iloc[0] if len(pub_tier)>0 else 'Small'

    # Market & ratios
    market_avg = df[(df['Year']<year)&(df['Year']>=year-3)]['Global_Sales'].mean()
    market_total = df[(df['Year']<year)&(df['Year']>=year-3)]['Global_Sales'].sum()

    input_data = pd.DataFrame([{
        'Year': year, 'Genre': genre, 'Platform': platform,
        'Publisher_Tier': pub_tier, 'Platform_Manufacturer': manuf,
        'Platform_Lifecycle_Stage': lifecycle, 'ConsoleGen': console_gen,
        'Is_Sequel': int(is_sequel), 'Is_Multiplatform': int(num_platforms>=2),
        'Num_Platforms': num_platforms,
        'Prev_Franchise_MaxSales': franchise_max, 'Prev_Franchise_TitleCount': franchise_count,
        'Pub_Prev3Y_AvgSales': pub_feat['Pub_Prev3Y_AvgSales'],
        'Pub_Prev3Y_Count': pub_feat['Pub_Prev3Y_Count'],
        'Genre_Prev3Y_AvgSales': gen_feat['Genre_Prev3Y_AvgSales'],
        'Genre_Prev3Y_Count': gen_feat['Genre_Prev3Y_Count'],
        'Platform_Prev3Y_AvgSales': plat_feat['Platform_Prev3Y_AvgSales'],
        'Platform_Prev3Y_Count': plat_feat['Platform_Prev3Y_Count'],
        'Years_Since_Platform_Launch': years_since if years_since>=0 else 0,
        'Genre_vs_Market_Avg': gen_feat['Genre_Prev3Y_AvgSales']/(market_avg+1e-6),
        'Publisher_MarketShare_Prev3Y': pub_feat['Pub_Prev3Y_AvgSales']*pub_feat['Pub_Prev3Y_Count']/(market_total+1e-6),
        'Genre_Trend_Momentum': 1.0,  # gia su trung tinh
        'PubGenre_Specialization': 1.0,
        'GenrePlatform_Fit': 1.0,
        'SameGenre_Releases_PrevYear': gen_feat['Genre_Prev3Y_Count']/3,
    }])

    # --- Canh bao ngoai pham vi train ---
    warnings_list = []
    for c in ['Year','Num_Platforms','Prev_Franchise_MaxSales']:
        lo, hi = train_ranges.get(c, (None, None))
        val = input_data[c].iloc[0]
        if lo is not None and (val < lo or val > hi):
            warnings_list.append(f"⚠️ **{c}** = {val} ngoài phạm vi train ({lo:.0f}–{hi:.0f}) → dự đoán kém tin cậy")
    if year > 2016:
        warnings_list.append("⚠️ Năm > 2016: dữ liệu train rất ít ở giai đoạn này (industry chuyển sang mobile/digital)")
    if warnings_list:
        for w in warnings_list: st.warning(w)

    if st.button("🚀 Dự đoán doanh số", type="primary"):
        pred = model.predict(input_data[CAT+NUM])[0]
        pred = max(0.01, pred)  # khong am
        mae_test = 0.385  # MAE cua XGBoost tuned tren test
        lo_ci = max(0.01, pred - 1.96*mae_test)
        hi_ci = pred + 1.96*mae_test
        st.balloons()
        st.subheader("Kết quả dự đoán")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Doanh số dự đoán", f"{pred:.2f} triệu bản")
        col_b.metric("Khoảng tin cậy 95%", f"{lo_ci:.2f} – {hi_ci:.2f} M")
        col_c.metric("Sai số kỳ vọng (MAE)", f"±{mae_test:.2f} M")

        # Phan loai
        if pred >= 5: level = "💎 Blockbuster (≥5M)"
        elif pred >= 1: level = "⭐ Hit (1–5M)"
        elif pred >= 0.5: level = "✅ Thành công vừa (0.5–1M)"
        else: level = "📦 Thành công thấp (<0.5M)"
        st.info(f"**Phân loại:** {level}")

        with st.expander("Xem chi tiết features đã dùng để dự đoán"):
            st.dataframe(input_data.T, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("ADY201m FA26 | Video Game Global Sales Prediction | Leakage-free methodology | Data: VGChartz via Kaggle (16,327 games, 1980–2020)")
