import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set Streamlit page config first thing
st.set_page_config(
    page_title="🚗 EV Adoption Forecaster", 
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Load model ===
model = joblib.load('forecasting_ev_model.pkl')

# === Enhanced Styling ===
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .title-container {
            text-align: center;
            padding: 30px 0;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            margin: 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-box {
            background: rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        
        .stSelectbox > div > div {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
        }
        
        .stButton > button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border-radius: 25px;
            border: none;
            padding: 10px 30px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
    </style>
""", unsafe_allow_html=True)

# === Enhanced Header ===
st.markdown("""
    <div class="title-container">
        <h1 style='font-size: 3rem; font-weight: 700; color: #FFFFFF; margin: 0;'>
            🚗 EV Adoption Forecaster
        </h1>
        <p style='font-size: 1.2rem; color: #FFFFFF; margin: 10px 0 0 0; opacity: 0.9;'>
            Advanced Electric Vehicle Adoption Prediction for Washington State Counties
        </p>
        <div style='display: flex; justify-content: center; gap: 20px; margin-top: 15px;'>
            <span style='background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; font-size: 0.9rem;'>
                📊 AI-Powered Forecasting
            </span>
            <span style='background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; font-size: 0.9rem;'>
                🎯 3-Year Predictions
            </span>
            <span style='background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; font-size: 0.9rem;'>
                📈 Real-time Analytics
            </span>
        </div>
    </div>
""", unsafe_allow_html=True)

# === Sidebar with enhanced features ===
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h3 style='color: white; margin: 0;'>🎛️ Control Panel</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Add some interactive features
    st.markdown("### 📊 Dashboard Features")
    show_statistics = st.checkbox("📈 Show Statistics Dashboard", value=True)
    show_comparison = st.checkbox("🔍 Enable County Comparison", value=True)
    show_insights = st.checkbox("💡 Show AI Insights", value=True)
    
    st.markdown("### ⚙️ Display Options")
    chart_theme = st.selectbox("🎨 Chart Theme", ["plotly", "plotly_white", "plotly_dark"])
    forecast_period = st.slider("📅 Forecast Period (months)", 12, 60, 36)

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("preprocessed_ev_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# === Enhanced County Selection ===
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h3 style='color: white; margin: 0;'>🏛️ Select Your County</h3>
        </div>
    """, unsafe_allow_html=True)
    
    county_list = sorted(df['County'].dropna().unique().tolist())
    county = st.selectbox("Choose a county to analyze:", county_list, 
                         help="Select a county to view its EV adoption forecast")

if county not in df['County'].unique():
    st.error(f"County '{county}' not found in dataset.")
    st.stop()

# === Enhanced Data Processing ===
county_df = df[df['County'] == county].sort_values("Date")
county_code = county_df['county_encoded'].iloc[0]

# === Statistics Dashboard ===
if show_statistics:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>📊 {county} County Statistics</h2>
        </div>
    """.format(county=county), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_evs = county_df['Electric Vehicle (EV) Total'].sum()
        st.metric("🚗 Total EVs", f"{total_evs:,}")
    
    with col2:
        avg_monthly = county_df['Electric Vehicle (EV) Total'].mean()
        st.metric("📈 Avg Monthly", f"{avg_monthly:.1f}")
    
    with col3:
        max_month = county_df.loc[county_df['Electric Vehicle (EV) Total'].idxmax(), 'Date']
        st.metric("📅 Peak Month", max_month.strftime("%b %Y"))
    
    with col4:
        growth_rate = ((county_df['Electric Vehicle (EV) Total'].iloc[-1] - county_df['Electric Vehicle (EV) Total'].iloc[0]) / 
                      county_df['Electric Vehicle (EV) Total'].iloc[0] * 100) if county_df['Electric Vehicle (EV) Total'].iloc[0] > 0 else 0
        st.metric("📊 Growth Rate", f"{growth_rate:.1f}%")

# === Enhanced Forecasting ===
historical_ev = list(county_df['Electric Vehicle (EV) Total'].values[-6:])
cumulative_ev = list(np.cumsum(historical_ev))
months_since_start = county_df['months_since_start'].max()
latest_date = county_df['Date'].max()

future_rows = []
forecast_horizon = forecast_period

for i in range(1, forecast_horizon + 1):
    forecast_date = latest_date + pd.DateOffset(months=i)
    months_since_start += 1
    lag1, lag2, lag3 = historical_ev[-1], historical_ev[-2], historical_ev[-3]
    roll_mean = np.mean([lag1, lag2, lag3])
    pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
    pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
    recent_cumulative = cumulative_ev[-6:]
    ev_growth_slope = np.polyfit(range(len(recent_cumulative)), recent_cumulative, 1)[0] if len(recent_cumulative) == 6 else 0

    new_row = {
        'months_since_start': months_since_start,
        'county_encoded': county_code,
        'ev_total_lag1': lag1,
        'ev_total_lag2': lag2,
        'ev_total_lag3': lag3,
        'ev_total_roll_mean_3': roll_mean,
        'ev_total_pct_change_1': pct_change_1,
        'ev_total_pct_change_3': pct_change_3,
        'ev_growth_slope': ev_growth_slope
    }

    pred = model.predict(pd.DataFrame([new_row]))[0]
    future_rows.append({"Date": forecast_date, "Predicted EV Total": round(pred)})

    historical_ev.append(pred)
    if len(historical_ev) > 6:
        historical_ev.pop(0)

    cumulative_ev.append(cumulative_ev[-1] + pred)
    if len(cumulative_ev) > 6:
        cumulative_ev.pop(0)

# === Enhanced Visualization with Plotly ===
st.markdown("---")
st.markdown("""
    <div style='text-align: center; margin: 20px 0;'>
        <h2 style='color: white; margin: 0;'>📈 EV Adoption Forecast</h2>
    </div>
""", unsafe_allow_html=True)

# Prepare data for plotting
historical_cum = county_df[['Date', 'Electric Vehicle (EV) Total']].copy()
historical_cum['Source'] = 'Historical'
historical_cum['Cumulative EV'] = historical_cum['Electric Vehicle (EV) Total'].cumsum()

forecast_df = pd.DataFrame(future_rows)
forecast_df['Source'] = 'Forecast'
forecast_df['Cumulative EV'] = forecast_df['Predicted EV Total'].cumsum() + historical_cum['Cumulative EV'].iloc[-1]

combined = pd.concat([
    historical_cum[['Date', 'Cumulative EV', 'Source']],
    forecast_df[['Date', 'Cumulative EV', 'Source']]
], ignore_index=True)

# Create enhanced Plotly chart
fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=historical_cum['Date'],
    y=historical_cum['Cumulative EV'],
    mode='lines+markers',
    name='Historical Data',
    line=dict(color='#4CAF50', width=3),
    marker=dict(size=8)
))

# Forecast data
fig.add_trace(go.Scatter(
    x=forecast_df['Date'],
    y=forecast_df['Cumulative EV'],
    mode='lines+markers',
    name='Forecast',
    line=dict(color='#FF6B6B', width=3, dash='dash'),
    marker=dict(size=8)
))

fig.update_layout(
    title=f"EV Adoption Forecast for {county} County",
    xaxis_title="Date",
    yaxis_title="Cumulative EV Count",
    template=chart_theme,
    height=500,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# === Enhanced Insights ===
if show_insights:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>💡 AI-Generated Insights</h2>
        </div>
    """, unsafe_allow_html=True)
    
    historical_total = historical_cum['Cumulative EV'].iloc[-1]
    forecasted_total = forecast_df['Cumulative EV'].iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if historical_total > 0:
            forecast_growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
            trend = "📈" if forecast_growth_pct > 0 else "📉"
            st.success(f"""
            **{trend} Growth Prediction**
            
            Expected growth: **{forecast_growth_pct:.1f}%** over {forecast_period} months
            
            Current EVs: {historical_total:,}
            Predicted EVs: {forecasted_total:,}
            """)
        else:
            st.warning("Insufficient historical data for growth analysis.")
    
    with col2:
        avg_monthly_growth = (forecast_df['Predicted EV Total'].mean())
        st.info(f"""
        **📊 Monthly Trends**
        
        Average monthly growth: **{avg_monthly_growth:.1f}** EVs
        Peak month: **{forecast_df.loc[forecast_df['Predicted EV Total'].idxmax(), 'Date'].strftime('%B %Y')}**
        """)

# === Enhanced County Comparison ===
if show_comparison:
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; margin: 20px 0;'>
            <h2 style='color: white; margin: 0;'>🔍 Multi-County Comparison</h2>
        </div>
    """, unsafe_allow_html=True)
    
    multi_counties = st.multiselect(
        "Select up to 3 counties to compare:", 
        county_list, 
        max_selections=3,
        default=[county] if county in county_list else None
    )
    
    if multi_counties:
        comparison_data = []
        
        for cty in multi_counties:
            cty_df = df[df['County'] == cty].sort_values("Date")
            cty_code = cty_df['county_encoded'].iloc[0]
            
            hist_ev = list(cty_df['Electric Vehicle (EV) Total'].values[-6:])
            cum_ev = list(np.cumsum(hist_ev))
            months_since = cty_df['months_since_start'].max()
            last_date = cty_df['Date'].max()
            
            future_rows_cty = []
            for i in range(1, forecast_horizon + 1):
                forecast_date = last_date + pd.DateOffset(months=i)
                months_since += 1
                lag1, lag2, lag3 = hist_ev[-1], hist_ev[-2], hist_ev[-3]
                roll_mean = np.mean([lag1, lag2, lag3])
                pct_change_1 = (lag1 - lag2) / lag2 if lag2 != 0 else 0
                pct_change_3 = (lag1 - lag3) / lag3 if lag3 != 0 else 0
                recent_cum = cum_ev[-6:]
                ev_slope = np.polyfit(range(len(recent_cum)), recent_cum, 1)[0] if len(recent_cum) == 6 else 0
                
                new_row = {
                    'months_since_start': months_since,
                    'county_encoded': cty_code,
                    'ev_total_lag1': lag1,
                    'ev_total_lag2': lag2,
                    'ev_total_lag3': lag3,
                    'ev_total_roll_mean_3': roll_mean,
                    'ev_total_pct_change_1': pct_change_1,
                    'ev_total_pct_change_3': pct_change_3,
                    'ev_growth_slope': ev_slope
                }
                pred = model.predict(pd.DataFrame([new_row]))[0]
                future_rows_cty.append({"Date": forecast_date, "Predicted EV Total": round(pred)})
                
                hist_ev.append(pred)
                if len(hist_ev) > 6:
                    hist_ev.pop(0)
                
                cum_ev.append(cum_ev[-1] + pred)
                if len(cum_ev) > 6:
                    cum_ev.pop(0)
            
            hist_cum = cty_df[['Date', 'Electric Vehicle (EV) Total']].copy()
            hist_cum['Cumulative EV'] = hist_cum['Electric Vehicle (EV) Total'].cumsum()
            
            fc_df = pd.DataFrame(future_rows_cty)
            fc_df['Cumulative EV'] = fc_df['Predicted EV Total'].cumsum() + hist_cum['Cumulative EV'].iloc[-1]
            
            combined_cty = pd.concat([
                hist_cum[['Date', 'Cumulative EV']],
                fc_df[['Date', 'Cumulative EV']]
            ], ignore_index=True)
            
            combined_cty['County'] = cty
            comparison_data.append(combined_cty)
        
        # Create comparison chart
        comp_df = pd.concat(comparison_data, ignore_index=True)
        
        fig_comp = go.Figure()
        
        colors = ['#4CAF50', '#FF6B6B', '#FFD93D']
        for i, (cty, group) in enumerate(comp_df.groupby('County')):
            fig_comp.add_trace(go.Scatter(
                x=group['Date'],
                y=group['Cumulative EV'],
                mode='lines+markers',
                name=cty,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))
        
        fig_comp.update_layout(
            title="Multi-County EV Adoption Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative EV Count",
            template=chart_theme,
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Growth summary
        growth_summaries = []
        for cty in multi_counties:
            cty_df_comp = comp_df[comp_df['County'] == cty].reset_index(drop=True)
            historical_total = cty_df_comp['Cumulative EV'].iloc[len(cty_df_comp) - forecast_horizon - 1]
            forecasted_total = cty_df_comp['Cumulative EV'].iloc[-1]
            
            if historical_total > 0:
                growth_pct = ((forecasted_total - historical_total) / historical_total) * 100
                growth_summaries.append(f"**{cty}**: {growth_pct:.1f}%")
            else:
                growth_summaries.append(f"**{cty}**: N/A")
        
        st.success(f"📊 **Forecasted Growth**: {' | '.join(growth_summaries)}")

# === Footer ===
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 30px;'>
        <p style='color: white; margin: 0; font-size: 1.1rem;'>
            🚀 <strong>EV Adoption Forecaster</strong> | Powered by AI & Machine Learning
        </p>
        <p style='color: white; margin: 5px 0 0 0; opacity: 0.8;'>
            Prepared for AICTE Internship Cycle 2 by S4F | Advanced Analytics & Predictive Modeling
        </p>
    </div>
""", unsafe_allow_html=True)
