"""
  — Ethiopian Crop Yield Prediction
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import base64
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
# You can use emoji (default) or path to image file for page_icon
# To use custom image: save as 'logo.png' in streamlit_app folder, then use:
# page_icon="logo.png" or page_icon=Image.open("logo.png")

st.set_page_config(
    page_title=" ",
    page_icon="",  # Change to "logo.png" if you add a custom logo
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load artifacts ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)

# ── Custom CSS — Agricultural Color Theme ────────────────────────────────────
st.markdown("""
<style>
  /* Agricultural Color Theme */
  
  /* Main background - soft cream like wheat fields */
  .main { background-color: #FFF8DC !important; }
  
  /* Sidebar - rich soil brown with green accent */
  [data-testid="stSidebar"] { 
      background: linear-gradient(180deg, #2D5016 0%, #1a3d0a 100%) !important;
  }
  [data-testid="stSidebar"] * { 
      color: #FFFFFF !important; 
      text-shadow: none !important;
  }
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
      color: #FFFFFF !important;
      text-shadow: none !important;
  }
  
  /* Metric boxes - darker, more opaque for photo background */
  .metric-box {
      background: linear-gradient(135deg, rgba(85, 139, 47, 0.95) 0%, rgba(56, 102, 24, 0.95) 100%);
      border-left: 5px solid #FFD700;
      border-radius: 12px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 0.8rem;
      box-shadow: 0 6px 12px rgba(0,0,0,0.25);
      backdrop-filter: blur(10px);
      transition: transform 0.2s;
  }
  .metric-box:hover {
      transform: translateY(-3px);
      box-shadow: 0 8px 16px rgba(0,0,0,0.3);
  }
  .metric-box h4 { 
      color: #FFFFFF; 
      margin: 0; 
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      text-shadow: none;
      opacity: 1;
  }
  .metric-box h2 { 
      color: #FFFFFF; 
      margin: 0.3rem 0 0; 
      font-size: 1.8rem; 
      font-weight: 700;
      text-shadow: none;
      opacity: 1;
  }
  
  /* Section titles - stronger shadow for photo background */
  .section-title {
      font-size: 1.8rem; 
      font-weight: 700;
      color: #1a3d0a;
      background: rgba(255, 255, 255, 0.9);
      border-left: 6px solid #558B2F;
      border-bottom: 3px solid #F4A460;
      padding: 0.8rem 1.2rem; 
      margin-bottom: 1.5rem;
      text-shadow: none;
      box-shadow: 0 3px 8px rgba(0,0,0,0.15);
      border-radius: 6px;
  }
  
  /* Buttons - more opaque and vibrant */
  .stButton > button {
      background: linear-gradient(135deg, rgba(124, 179, 66, 0.95) 0%, rgba(85, 139, 47, 0.95) 100%) !important;
      color: white !important;
      border: 2px solid rgba(255, 215, 0, 0.6) !important;
      border-radius: 10px !important;
      font-weight: 700 !important;
      font-size: 1.05rem !important;
      padding: 0.7rem 1.8rem !important;
      box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
      backdrop-filter: blur(10px) !important;
      transition: all 0.3s !important;
      text-shadow: none !important;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, rgba(139, 195, 74, 0.98) 0%, rgba(104, 159, 56, 0.98) 100%) !important;
      transform: translateY(-3px) scale(1.02) !important;
      box-shadow: 0 6px 14px rgba(0,0,0,0.35) !important;
      border-color: rgba(255, 215, 0, 0.9) !important;
  }
  
  /* Radio buttons - wheat gold accent */
  [data-testid="stSidebar"] .stRadio > label {
      background-color: rgba(255, 215, 0, 0.1);
      border-radius: 6px;
      padding: 0.3rem 0.6rem;
      margin: 0.2rem 0;
  }
  
  /* Input fields - light cream background */
  .stNumberInput > div > div > input,
  .stSelectbox > div > div > select,
  .stSlider > div > div > div {
      background-color: #FFFAF0 !important;
      border: 2px solid #F4A460 !important;
      border-radius: 6px !important;
  }
  
  /* Success messages - darker green background */
  .stSuccess {
      background-color: rgba(45, 80, 22, 0.95) !important;
      border-left: 5px solid #FFD700 !important;
      color: #FFFFFF !important;
      backdrop-filter: blur(10px) !important;
      box-shadow: 0 3px 8px rgba(0,0,0,0.15) !important;
  }
  
  .stSuccess h3 {
      color: #FFFFFF !important;
      text-shadow: none !important;
      opacity: 1 !important;
  }
  
  /* Tabs - more opaque for photo background */
  .stTabs [data-baseweb="tab-list"] {
      background-color: rgba(245, 222, 179, 0.95);
      border-radius: 10px;
      padding: 0.6rem;
      box-shadow: 0 3px 8px rgba(0,0,0,0.15);
      backdrop-filter: blur(10px);
  }
  .stTabs [data-baseweb="tab"] {
      color: #2D5016 !important;
      font-weight: 700;
      padding: 0.6rem 1.2rem;
  }
  .stTabs [aria-selected="true"] {
      background-color: rgba(124, 179, 66, 0.95) !important;
      color: white !important;
      border-radius: 8px !important;
      box-shadow: 0 2px 6px rgba(0,0,0,0.2) !important;
  }
  
  /* Dataframe styling - opaque background */
  [data-testid="stDataFrame"] {
      border: 3px solid #558B2F !important;
      border-radius: 10px !important;
      background: rgba(255, 255, 255, 0.95) !important;
      box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
      backdrop-filter: blur(10px) !important;
  }
  
  /* Input fields - opaque with stronger borders */
  .stNumberInput > div > div > input,
  .stSelectbox > div > div > select,
  .stSlider {
      background-color: rgba(255, 250, 240, 0.95) !important;
      border: 2px solid #558B2F !important;
      border-radius: 8px !important;
      backdrop-filter: blur(10px) !important;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
  }
  
  /* Markdown containers - no effects */
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] h4,
  [data-testid="stMarkdownContainer"] h3 {
      text-shadow: none !important;
  }
  
  /* Metric widgets - enhanced with darker background */
  [data-testid="stMetric"] {
      background: rgba(45, 80, 22, 0.9) !important;
      padding: 1rem !important;
      border-radius: 8px !important;
      border-left: 4px solid #FFD700 !important;
      box-shadow: 0 3px 8px rgba(0,0,0,0.15) !important;
      backdrop-filter: blur(10px) !important;
  }
  
  /* Metric labels and values - pure white text, no effects */
  [data-testid="stMetric"] label {
      color: #FFFFFF !important;
      font-weight: 600 !important;
      text-shadow: none !important;
      opacity: 1 !important;
  }
  
  [data-testid="stMetric"] [data-testid="stMetricValue"] {
      color: #FFFFFF !important;
      font-weight: 700 !important;
      font-size: 1.5rem !important;
      text-shadow: none !important;
      opacity: 1 !important;
  }
  
  [data-testid="stMetric"] [data-testid="stMetricDelta"] {
      color: #FFD700 !important;
      text-shadow: none !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts (continued) ───────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, 'best_model.pkl'))

@st.cache_data
def load_json(filename):
    with open(os.path.join(BASE_DIR, filename)) as f:
        return json.load(f)

@st.cache_data
def load_dataset():
    path = os.path.join(BASE_DIR, 'Etho-Agri Dataset_Enhanced.xlsx')
    if not os.path.exists(path):
        return None
    df = pd.read_excel(path)
    df = df.rename(columns={
        'Region': 'region', 'crop type': 'crop_type', 'Year': 'year',
        'Production(kg)': 'production_kg', 'Yeild (kg/ha)': 'yield_kg_ha',
        ' Area cultivated(Ha)': 'area_ha', 'Area_Filled': 'area_filled',
        'Region_Avg_Yield': 'region_avg_yield', 'Crop_Avg_Yield': 'crop_avg_yield',
    })
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df[df['yield_kg_ha'].notna() & (df['yield_kg_ha'] > 0)]
    return df

model     = load_model()
feat_cols = load_json('feature_cols.json')
encodings = load_json('encodings.json')
metrics   = load_json('metrics.json')
stats     = load_json('stats.json')
df_raw    = load_dataset()

REGIONS = encodings['regions_list']
CROPS   = encodings['crops_list']

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    # Optional: Display logo image if available
    logo_path = os.path.join(BASE_DIR, 'logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    
    st.markdown("##  ")
    st.markdown("*Ethiopian Crop Yield Prediction*")
    st.divider()
    page = st.radio(
        "Navigate",
        [" Home", " EDA", " Predict", " Model Performance", " About"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown(f"**Best Model:** {metrics['best_model']}")
    best = next(r for r in metrics['test_results'] if r['Model'] == metrics['best_model'])
    st.markdown(f"**R²:** `{best['R2']}`")
    st.markdown(f"**RMSE:** `{best['RMSE']:,}` kg/ha")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == " Home":
    # Optional: Display banner logo/photo if available
    banner_path = os.path.join(BASE_DIR, 'banner.png')
    if os.path.exists(banner_path):
        st.image(banner_path, use_container_width=True)
    
    st.markdown('<p class="section-title"> EthioYield prediction — Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Machine learning-powered **crop yield prediction** for Ethiopian regional agriculture.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><h4>Total Records</h4><h2>{stats["total_rows"]:,}</h2></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><h4>Regions</h4><h2>{len(REGIONS)}</h2></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><h4>Crop Types</h4><h2>{len(CROPS)}</h2></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><h4>Year Range</h4><h2>{stats["base_year"]}–2022</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    if df_raw is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("####  Average Yield Trend by Crop")
            trend = df_raw.groupby(['year', 'crop_type'])['yield_kg_ha'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#FFF8DC')
            ax.set_facecolor('#FFFEF0')
            # Agricultural color palette for crops
            crop_colors = ['#7CB342', '#F4A460', '#8B4513', '#FFD700', '#2D5016', '#87CEEB', '#D2691E']
            for i, crop in enumerate(df_raw['crop_type'].dropna().unique()):
                sub = trend[trend['crop_type'] == crop]
                ax.plot(sub['year'], sub['yield_kg_ha'], marker='o', markersize=4, 
                       label=crop, color=crop_colors[i % len(crop_colors)], linewidth=2)
            ax.set_xlabel('Year', fontweight='bold', color='#2D5016'); 
            ax.set_ylabel('Avg Yield (kg/ha)', fontweight='bold', color='#2D5016')
            ax.legend(fontsize=7, ncol=2, framealpha=0.9); ax.grid(alpha=0.3, color='#F4A460')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            st.markdown("####  Average Yield by Region")
            reg = df_raw.groupby('region')['yield_kg_ha'].mean().sort_values()
            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#FFF8DC')
            ax.set_facecolor('#FFFEF0')
            # Gradient from light green to dark green
            colors = plt.cm.YlGn(np.linspace(0.3, 0.9, len(reg)))
            bars = ax.barh(reg.index, reg.values, color=colors, edgecolor='#2D5016', linewidth=1.5)
            ax.bar_label(bars, fmt='%.0f', padding=3, fontsize=8, color='#2D5016', fontweight='bold')
            ax.set_xlabel('Avg Yield (kg/ha)', fontweight='bold', color='#2D5016')
            ax.grid(axis='x', alpha=0.3, color='#F4A460')
            plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("Place `Etho-Agri Dataset_Enhanced.xlsx` in the app folder to see charts.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == " EDA":
    st.markdown('<p class="section-title"> Exploratory Data Analysis</p>', unsafe_allow_html=True)

    if df_raw is None:
        st.warning("Dataset not found. Place the Excel file in the app folder.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Yield by Crop", "Heatmap", "Production", "Correlation"])

    with tab1:
        st.markdown("#### Yield Distribution by Crop Type")
        p99 = df_raw.groupby('crop_type')['yield_kg_ha'].transform(lambda x: x.quantile(0.99))
        df_plot = df_raw[df_raw['yield_kg_ha'] <= p99]
        order = df_plot.groupby('crop_type')['yield_kg_ha'].median().sort_values(ascending=False).index.tolist()
        fig, ax = plt.subplots(figsize=(11, 4), facecolor='#FFF8DC')
        ax.set_facecolor('#FFFEF0')
        crop_palette = ['#7CB342', '#F4A460', '#8B4513', '#FFD700', '#2D5016', '#87CEEB', '#D2691E', '#9ACD32']
        for i, crop in enumerate(order):
            data = df_plot[df_plot['crop_type'] == crop]['yield_kg_ha']
            ax.boxplot(data, positions=[i], widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=crop_palette[i % len(crop_palette)], alpha=0.8, edgecolor='#2D5016', linewidth=1.5),
                       medianprops=dict(color='#8B4513', linewidth=2.5),
                       whiskerprops=dict(color='#2D5016', linewidth=1.5),
                       capprops=dict(color='#2D5016', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=3, alpha=0.4, markerfacecolor='#F4A460'))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, color='#2D5016', fontweight='bold')
        ax.set_ylabel('Yield (kg/ha)', fontweight='bold', color='#2D5016')
        ax.grid(axis='y', alpha=0.3, color='#F4A460')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("#### Average Yield — Region × Crop")
        pivot = df_raw.pivot_table(values='yield_kg_ha', index='region',
                                    columns='crop_type', aggfunc='mean').round(0)
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='#FFF8DC')
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGn',
                    linewidths=0.8, linecolor='white', ax=ax,
                    cbar_kws={'label': 'Yield (kg/ha)'})
        ax.set_xlabel('Crop', fontweight='bold', color='#2D5016')
        ax.set_ylabel('Region', fontweight='bold', color='#2D5016')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("#### Total Production by Region")
        prod = df_raw.groupby(['region','crop_type'])['production_kg'].sum().unstack(fill_value=0).div(1e6)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#FFF8DC')
        ax.set_facecolor('#FFFEF0')
        # Agricultural color map for stacked bars
        crop_colors_stack = ['#7CB342', '#F4A460', '#FFD700', '#8B4513', '#2D5016', '#87CEEB', '#D2691E']
        prod.plot(kind='bar', stacked=True, ax=ax, color=crop_colors_stack, edgecolor='white', linewidth=1.2)
        ax.set_ylabel('Production (Million kg)', fontweight='bold', color='#2D5016')
        ax.legend(bbox_to_anchor=(1,1), fontsize=8, framealpha=0.95)
        ax.grid(axis='y', alpha=0.3, color='#F4A460')
        plt.xticks(rotation=30, ha='right', color='#2D5016', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        st.markdown("#### Feature Correlation Matrix")
        cols = ['yield_kg_ha','area_filled','production_kg','year','region_avg_yield','crop_avg_yield']
        cols = [c for c in cols if c in df_raw.columns]
        corr = df_raw[cols].replace([np.inf,-np.inf], np.nan).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#FFF8DC')
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, linewidths=0.8, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Predict":
    st.markdown('<p class="section-title"> Predict Crop Yield</p>', unsafe_allow_html=True)
    st.markdown("Fill in the details below to get an instant yield prediction.")

    col1, col2, col3 = st.columns(3)
    with col1:
        region   = st.selectbox("Region", REGIONS)
        crop     = st.selectbox("Crop Type", CROPS)
        year     = st.slider("Year", 1996, 2030, 2023)
    with col2:
        area     = st.number_input("Area Cultivated (Ha)", min_value=1.0, value=5000.0, step=100.0)
        lag1     = st.number_input("Last Year's Yield (kg/ha)", min_value=0.0, value=1500.0, step=50.0)
        lag2     = st.number_input("2 Years Ago Yield (kg/ha)", min_value=0.0, value=1400.0, step=50.0)
    with col3:
        lag3     = st.number_input("3 Years Ago Yield (kg/ha)", min_value=0.0, value=1300.0, step=50.0)
        production = st.number_input("Expected Production (kg)", min_value=0.0, value=7500000.0, step=100000.0)
        is_recent  = st.selectbox("Is Recent (post-2010)?", [1, 0])

    st.markdown("---")

    if st.button(" Predict Yield", use_container_width=True):
        # Build feature vector matching FEATURE_COLS
        region_enc_val = list(encodings['region_map'].values()).index(region) \
                         if region in encodings['region_map'].values() else 0
        crop_enc_val   = list(encodings['crop_map'].values()).index(crop) \
                         if crop in encodings['crop_map'].values() else 0

        base_year    = stats['base_year']
        roll_mean_2  = (lag1 + lag2) / 2
        roll_mean_3  = (lag1 + lag2 + lag3) / 3
        decade       = (year // 10) * 10

        region_avg = df_raw[df_raw['region'] == region]['yield_kg_ha'].mean() if df_raw is not None else 1200
        crop_avg   = df_raw[df_raw['crop_type'] == crop]['yield_kg_ha'].mean() if df_raw is not None else 1200

        feature_map = {
            'year'              : year,
            'year_trend'        : year - base_year,
            'year_since_start'  : year - base_year,
            'is_recent'         : is_recent,
            'decade_1990'       : int(decade == 1990),
            'decade_2000'       : int(decade == 2000),
            'decade_2010'       : int(decade == 2010),
            'decade_2020'       : int(decade == 2020),
            'region_enc'        : region_enc_val,
            'crop_enc'          : crop_enc_val,
            'region_code'       : region_enc_val,
            'crop_code'         : crop_enc_val,
            'log_area'          : np.log1p(area),
            'log_production'    : np.log1p(production),
            'area_efficiency'   : production / (area + 1e-6),
            'area_missing_flag' : 0,
            'region_avg_yield'  : region_avg,
            'crop_avg_yield'    : crop_avg,
            'region_crop_ratio' : region_avg / (crop_avg + 1e-6),
            'region_growth_rate': 0.05,
            'crop_trend'        : 0.02,
            'yield_lag1'        : lag1,
            'yield_lag2'        : lag2,
            'yield_lag3'        : lag3,
            'yield_roll_mean_2' : roll_mean_2,
            'yield_roll_mean_3' : roll_mean_3,
        }

        input_df = pd.DataFrame([{col: feature_map.get(col, 0) for col in feat_cols}])
        prediction = model.predict(input_df)[0]
        prediction = max(0, prediction)

        st.success(f"###  Predicted Yield: **{prediction:,.0f} kg/ha**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Yield", f"{prediction:,.0f} kg/ha")
        c2.metric("National Average", f"{stats['yield_mean']:,.0f} kg/ha",
                  delta=f"{prediction - stats['yield_mean']:+,.0f}")
        c3.metric("Region", region)

        # Gauge-style bar
        st.markdown("#### How does this compare to historical yields?")
        fig, ax = plt.subplots(figsize=(9, 2), facecolor='#FFF8DC')
        ax.set_facecolor('#FFFEF0')
        bars = ax.barh(['Historical Min', 'Historical Mean', 'Prediction', 'Historical Max'],
                [stats['yield_min'], stats['yield_mean'], prediction, stats['yield_max']],
                color=['#D2691E','#F4A460','#7CB342','#FFD700'], edgecolor='#2D5016', linewidth=1.5)
        ax.axvline(prediction, color='#2D5016', linewidth=3, linestyle='--', label='Your Prediction')
        ax.set_xlabel('Yield (kg/ha)', fontweight='bold', color='#2D5016')
        ax.legend()
        ax.grid(axis='x', alpha=0.3, color='#F4A460')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Model Performance":
    st.markdown('<p class="section-title"> Model Performance</p>', unsafe_allow_html=True)

    results_df = pd.DataFrame(metrics['test_results']).sort_values('RMSE')

    st.markdown("#### All Models — Test Set Metrics")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### RMSE Comparison")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#FFF8DC')
        ax.set_facecolor('#FFFEF0')
        colors = ['#7CB342' if m == metrics['best_model'] else '#D2B48C'
                  for m in results_df['Model']]
        bars = ax.barh(results_df['Model'], results_df['RMSE'],
                       color=colors, edgecolor='#2D5016', linewidth=1.5)
        ax.bar_label(bars, fmt='%.0f', padding=3, color='#2D5016', fontweight='bold')
        ax.set_xlabel('RMSE (kg/ha) — lower is better', fontweight='bold', color='#2D5016')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3, color='#F4A460')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("#### R² Comparison")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='#FFF8DC')
        ax.set_facecolor('#FFFEF0')
        colors2 = ['#FFD700' if m == metrics['best_model'] else '#D2B48C'
                   for m in results_df['Model']]
        bars2 = ax.barh(results_df['Model'], results_df['R2'],
                        color=colors2, edgecolor='#2D5016', linewidth=1.5)
        ax.bar_label(bars2, fmt='%.4f', padding=3, color='#2D5016', fontweight='bold')
        ax.set_xlabel('R² — higher is better', fontweight='bold', color='#2D5016')
        ax.invert_yaxis(); ax.grid(axis='x', alpha=0.3, color='#F4A460')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("####  Best Model Summary")
    best = next(r for r in metrics['test_results'] if r['Model'] == metrics['best_model'])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model",  best['Model'])
    m2.metric("RMSE",   f"{best['RMSE']:,} kg/ha")
    m3.metric("R²",     str(best['R2']))
    m4.metric("MAPE",   f"{best['MAPE']}%")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == " About":
    st.markdown('<p class="section-title"> About This Project</p>', unsafe_allow_html=True)
    st.markdown("""
    ###  EthioYield AI

    This application predicts **crop yield (kg/ha)** for Ethiopian regions using
    machine learning trained on historical agricultural data from **1996 to 2022**.

    ---
    ####  Dataset
    - **Source:** Ethiopian Agricultural Data (Etho-Agri Dataset Enhanced)
    - **Regions:** 10 Ethiopian regions
    - **Crops:** Teff, Barley, Wheat, Maize, Sorghum, Millet, Oats
    - **Features:** Area, production, lag yields, rolling averages, regional statistics

    ---
    ####  Models Trained
    | Model | Description |
    |-------|-------------|
    | Random Forest | Ensemble of 300 decision trees |
    | XGBoost | Gradient boosting with histogram trees |
    | LightGBM | Fast gradient boosting (best performer) |
    | Voting Ensemble | Average of all three models |

    ---
    ####  Key Features Used
    - **Lag features** — yield from previous 1, 2, 3 years
    - **Rolling averages** — 2-year and 3-year smoothed yield
    - **Log transforms** — area and production (reduces skew)
    - **Label encoding** — region and crop type
    - **Temporal features** — year trend, decade dummies

    ---
    ####  Evaluation Metrics
    - **RMSE** — Root Mean Square Error
    - **MAE** — Mean Absolute Error
    - **R²** — Coefficient of Determination
    - **MAPE** — Mean Absolute Percentage Error

    ---
    *Built with Python · Scikit-learn · LightGBM · XGBoost · Streamlit*
    """)
