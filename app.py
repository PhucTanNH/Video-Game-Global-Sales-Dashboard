# ============================================================
# DASHBOARD STREAMLIT - Video Game Global Sales Prediction
# Run: streamlit run app.py
# ============================================================
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings('ignore')

APP_VERSION = "1.1.0"
LEGACY_VERSION = "1.0.0"

# ---------- Page configuration ----------
st.set_page_config(page_title=f"Video Game Sales Dashboard v{APP_VERSION}", layout="wide", page_icon="🎮")

# ---------- Load data & model (cached for performance) ----------
@st.cache_data
def load_data():
    return pd.read_excel('ADY201m_enhanced_v2.xlsx')

@st.cache_resource
def load_model():
    return joblib.load('models/best_model.joblib')

@st.cache_data
def load_model_meta():
    with open('models/model_meta.json', encoding='utf-8') as f:
        return json.load(f)

df = load_data()
model = load_model()
model_meta = load_model_meta()

# Keep inference feature order tied to the deployed model metadata.
CAT = model_meta['features_cat']
NUM = model_meta['features_num']

# Training ranges used to flag out-of-range inputs
TRAIN = df[df['Year']<=model_meta['train_years'][1]]
train_ranges = {c: (TRAIN[c].min(), TRAIN[c].max()) for c in NUM if c in TRAIN.columns}

# ---------- Sidebar: Filters ----------
st.sidebar.title("🎮 Filters")
genre_filter = st.sidebar.multiselect("Genre", sorted(df['Genre'].unique()), default=[])
platform_filter = st.sidebar.multiselect("Platform", sorted(df['Platform'].unique()), default=[])
year_range = st.sidebar.slider("Year range", int(df['Year'].min()), int(df['Year'].max()), (2000, 2016))
tier_filter = st.sidebar.selectbox("Publisher tier", ["All","Major","Mid","Small"])

df_filtered = df.copy()
if genre_filter: df_filtered = df_filtered[df_filtered['Genre'].isin(genre_filter)]
if platform_filter: df_filtered = df_filtered[df_filtered['Platform'].isin(platform_filter)]
df_filtered = df_filtered[(df_filtered['Year']>=year_range[0]) & (df_filtered['Year']<=year_range[1])]
if tier_filter != "All": df_filtered = df_filtered[df_filtered['Publisher_Tier']==tier_filter]

st.sidebar.markdown(f"**Games shown:** {len(df_filtered):,} / {len(df):,}")

# ---------- Title ----------
st.title("Video Game Global Sales Prediction Dashboard")
st.info(f"Current version: v{APP_VERSION} | Previous version: v{LEGACY_VERSION} (legacy)")
metrics = model_meta['metrics_test']
st.markdown(
    f"*ADY201m project — Model: {model_meta['model']} | Target: {model_meta['target']} (raw sales, millions) "
    f"| Inputs: {len(CAT) + len(NUM)} ({len(CAT)} categorical, {len(NUM)} numeric) "
    f"| Test RMSE={metrics['RMSE']:.3f}M, MAE={metrics['MAE']:.3f}M, R²={metrics['R2']:.2f}*"
)
with st.expander(f"What's new in v{APP_VERSION}"):
    st.markdown(
        "- Prediction features and metrics are read from the deployed model metadata.\n"
        "- Historical feature lookups use only information available before the selected release year.\n"
        "- Prediction uncertainty is labeled as an approximate range, not a calibrated 95% interval.\n"
        "- The feature details table now renders mixed numeric and categorical values reliably."
    )

tab1, tab2, tab3, tab4 = st.tabs(["📊 RQ1: Factors", "🤖 RQ2: Model Comparison", "🎯 RQ3: Console Generations", "🔮 Sales Prediction"])

# ============================================================
# TAB 1: RQ1 - Genre, platform, and regional impact
# ============================================================
with tab1:
    st.header("RQ1: Which genres, platforms, and regions have the greatest impact on sales?")
    st.success("Conclusion: Franchise history and publisher reputation have the strongest effects; platform has a moderate effect; genre has a weaker direct effect than expected after controlling for publisher and franchise.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average sales by genre")
        g_sales = df_filtered.groupby('Genre')['Global_Sales'].mean().sort_values(ascending=True)
        fig = px.bar(x=g_sales.values, y=g_sales.index, orientation='h',
                     labels={'x':'Average global sales (millions of units)','y':'Genre'},
                     color=g_sales.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 15 platforms by average sales")
        p_sales = df_filtered.groupby('Platform')['Global_Sales'].mean().sort_values(ascending=False).head(15)
        fig = px.bar(x=p_sales.index, y=p_sales.values,
                     labels={'x':'Platform','y':'Avg Global Sales (M)'}, color=p_sales.values)
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales share by region and genre")
    region = df_filtered.groupby('Genre')[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum()
    region_pct = region.div(region.sum(axis=1), axis=0)
    fig = px.bar(region_pct, barmode='stack', color_discrete_sequence=['#3182ce','#38a169','#e53e3e','#d69e2e'],
                 labels={'value':'Share','variable':'Region'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_title='Sales share')
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Insight: Role-Playing gets 35% of its sales from Japan (unusually high), while Shooter sales are concentrated in North America and Europe (>70%).")

    st.subheader("Top 10 most important features (permutation importance)")
    imp = pd.read_csv('models/permutation_importance.csv', index_col=0).iloc[:,0].sort_values(ascending=True).tail(10)
    fig = px.bar(x=imp.values, y=imp.index, orientation='h',
                 labels={'x':'Increase in RMSE after feature permutation','y':'Feature'},
                 title="Feature Importance (tuned XGBoost)")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: RQ2 - Model comparison
# ============================================================
with tab2:
    st.header("RQ2: Which model predicts sales most accurately?")
    st.success("Conclusion: Gradient-boosted trees (LightGBM/XGBoost/Random Forest) substantially outperform the linear model. LightGBM achieves the best test result: RMSE=0.947M, R²=0.42, exceeding the Zhan 2026 paper's baseline (R²≈0.384).")

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
                     title="RMSE Comparison (lower is better)")
        fig.add_hline(y=1.280, line_dash="dash", annotation_text="Median baseline")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(results, x='Model', y='R²', color='R²', color_continuous_scale='RdYlGn',
                     title="R² Comparison (higher is better)")
        fig.add_hline(y=0.384, line_dash="dash", annotation_text="Zhan 2026 baseline (R²≈0.384)")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cross-validation results (TimeSeriesSplit, 5 folds)")
    cv = pd.read_csv('models/cv_results.csv')
    st.dataframe(cv, use_container_width=True)
    st.caption("CV = TimeSeriesSplit(5) on the training data (≤2012). XGBoost was selected using CV, without using the test set.")

    st.subheader("Hyperparameter tuning log")
    tuning = pd.read_csv('models/tuning_log.csv')
    st.dataframe(tuning, use_container_width=True)

# ============================================================
# TAB 3: RQ3 - Console generations
# ============================================================
with tab3:
    st.header("RQ3: Do genre sales trends change across console generations?")
    st.success("Conclusion: Yes. Role-Playing sales decline from 0.95M (Gen 5) to 0.55M (Gen 8), while Shooter sales rise from 0.5M to 1.1M. Removing the generation feature increases RMSE by ~7.7% (0.947→1.02M).")

    pivot = df_filtered.pivot_table(index='Genre', columns='ConsoleGen', values='Global_Sales', aggfunc='mean')
    col_order = [c for c in ['Other','Gen_5','Gen_6','Gen_7','Gen_8'] if c in pivot.columns]
    pivot = pivot[col_order]
    fig = px.imshow(pivot, text_auto='.2f', color_continuous_scale='YlOrRd',
                    labels={'x':'Console Generation','y':'Genre','color':'Avg Sales (M)'},
                    title="Heatmap: Average Sales by Genre × Console Generation")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annual sales trends by broad genre")
    yearly_genre = df_filtered.groupby(['Year','Genre_Broad'])['Global_Sales'].mean().reset_index()
    fig = px.line(yearly_genre, x='Year', y='Global_Sales', color='Genre_Broad', markers=True,
                  labels={'Global_Sales':'Average sales (M)','Genre_Broad':'Broad genre'})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Insight: Action genres (Shooter/Fighting) rise sharply from Gen 6 to Gen 7; RPG/Strategy decline after Gen 6; Sports/Racing remain stable.")

    st.subheader("Prediction error by genre (test set)")
    err = pd.read_csv('models/error_by_genre.csv', index_col=0).sort_values('mean', ascending=True)
    fig = px.bar(x=err['mean'], y=err.index, orientation='h',
                 labels={'x':'MAE (millions of units)','y':'Genre'}, title="Mean prediction error by genre",
                 text=err['mean'].round(2))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Shooter (0.93M) and Sports (0.54M) are the hardest to predict, with unusually large hits such as Call of Duty and FIFA.")

# ============================================================
# TAB 4: Sales prediction
# ============================================================
with tab4:
    st.header("🔮 New Game Sales Predictor")
    st.markdown("Enter game details to predict global sales (millions of units). Model: **tuned XGBoost** (test RMSE≈1.0M).")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Release year", min_value=1995, max_value=2025, value=2015, step=1)
        genre = st.selectbox("Genre", sorted(df['Genre'].unique()))
        platform = st.selectbox("Platform", sorted(df['Platform'].unique()))
        publisher = st.selectbox("Publisher", sorted(df['Publisher'].unique()))
        num_platforms = st.slider("Number of release platforms", 1, 10, 1)

    with col2:
        is_sequel = st.checkbox("Is this a sequel?", value=False)
        franchise_max = st.number_input("Highest previous franchise sales (M)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
        franchise_count = st.number_input("Number of previous games in the franchise", min_value=0, max_value=50, value=0, step=1)

    # --- Automatically calculate rolling features from historical data ---
    def lookup(group_col, val, year, col_prefix):
        """Look up rolling features for a group over the previous three years."""
        sub = df[(df[group_col]==val) & (df['Year']<year) & (df['Year']>=year-3)]
        if len(sub)==0:
            sub = df[(df[group_col]==val) & (df['Year']<year)]
        historical_sales = df.loc[df['Year']<year, 'Global_Sales']
        return {
            f'{col_prefix}_Prev3Y_AvgSales': sub['Global_Sales'].mean() if len(sub)>0 else historical_sales.mean(),
            f'{col_prefix}_Prev3Y_Count': len(sub),
        }

    def historical_feature_value(feature, group_mask, year):
        history = df.loc[group_mask & (df['Year']<year), ['Year', feature]]
        if len(history)>0:
            latest_year = history['Year'].max()
            return history.loc[history['Year']==latest_year, feature].median()
        baseline = df.loc[df['Year']<year, feature].dropna()
        return baseline.median() if len(baseline)>0 else 0.0

    pub_feat  = lookup('Publisher', publisher, year, 'Pub')
    gen_feat  = lookup('Genre', genre, year, 'Genre')
    plat_feat = lookup('Platform', platform, year, 'Platform')

    # Platform lifecycle
    plat_launch = df[df['Platform']==platform]['Platform_Launch_Year'].median()
    years_since = year - plat_launch if pd.notna(plat_launch) else 5
    lifecycle = 'Early' if years_since<=2 else ('Mid' if years_since<=5 else 'Late')
    if platform=='PC': lifecycle = 'PC/Unknown'

    # Console generation
    console_gen = df[df['Platform']==platform]['ConsoleGen'].mode()
    console_gen = console_gen.iloc[0] if len(console_gen)>0 else 'Gen_7'

    # Manufacturer and publisher tier
    manuf = df[df['Platform']==platform]['Platform_Manufacturer'].mode()
    manuf = manuf.iloc[0] if len(manuf)>0 else 'Other'
    pub_tier = df[df['Publisher']==publisher]['Publisher_Tier'].mode()
    pub_tier = pub_tier.iloc[0] if len(pub_tier)>0 else 'Small'

    # Market values and ratios
    market_avg = df[(df['Year']<year)&(df['Year']>=year-3)]['Global_Sales'].mean()
    market_total = df[(df['Year']<year)&(df['Year']>=year-3)]['Global_Sales'].sum()

    genre_trend = historical_feature_value('Genre_Trend_Momentum', df['Genre']==genre, year)
    publisher_genre_fit = historical_feature_value(
        'PubGenre_Specialization', (df['Publisher']==publisher) & (df['Genre']==genre), year
    )
    genre_platform_fit = historical_feature_value(
        'GenrePlatform_Fit', (df['Genre']==genre) & (df['Platform']==platform), year
    )
    target_previous_year = year-1
    if df['Year'].min() <= target_previous_year <= df['Year'].max():
        same_genre_releases = int(((df['Genre']==genre) & (df['Year']==target_previous_year)).sum())
    else:
        recent_releases = df[(df['Genre']==genre) & (df['Year']<year)].groupby('Year').size().tail(3)
        same_genre_releases = recent_releases.mean() if len(recent_releases)>0 else 0

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
        'Genre_Trend_Momentum': genre_trend,
        'PubGenre_Specialization': publisher_genre_fit,
        'GenrePlatform_Fit': genre_platform_fit,
        'SameGenre_Releases_PrevYear': same_genre_releases,
    }])

    # --- Warn about inputs outside the training range ---
    warnings_list = []
    for c in ['Year','Num_Platforms','Prev_Franchise_MaxSales']:
        lo, hi = train_ranges.get(c, (None, None))
        val = input_data[c].iloc[0]
        if lo is not None and (val < lo or val > hi):
            warnings_list.append(f"⚠️ **{c}** = {val} is outside the training range ({lo:.0f}–{hi:.0f}) → prediction may be less reliable")
    if year > 2016:
        warnings_list.append("⚠️ Year > 2016: very little training data is available for this period (the industry shifted to mobile and digital).")
    if warnings_list:
        for w in warnings_list: st.warning(w)

    if st.button("🚀 Predict Sales", type="primary"):
        pred = model.predict(input_data[CAT+NUM])[0]
        pred = max(0.01, pred)  # Prevent negative predictions
        mae_test = metrics['MAE']
        lo_ci = max(0.01, pred - 1.96*mae_test)
        hi_ci = pred + 1.96*mae_test
        st.balloons()
        st.subheader("Prediction Results")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted sales", f"{pred:.2f} million units")
        col_b.metric("Approximate prediction range", f"{lo_ci:.2f} – {hi_ci:.2f} M")
        col_c.metric("Expected error (MAE)", f"±{mae_test:.2f} M")
        st.caption("Approximate range based on test MAE; it is not a calibrated 95% confidence interval.")

        # Classification
        if pred >= 5: level = "💎 Blockbuster (≥5M)"
        elif pred >= 1: level = "⭐ Hit (1–5M)"
        elif pred >= 0.5: level = "✅ Moderate success (0.5–1M)"
        else: level = "📦 Low sales (<0.5M)"
        st.info(f"**Classification:** {level}")

        with st.expander("View features used for this prediction"):
            st.dataframe(input_data.T.astype(str), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("ADY201m FA26 | Video Game Global Sales Prediction | Leakage-free methodology | Data: VGChartz via Kaggle (16,327 games, 1980–2020)")
