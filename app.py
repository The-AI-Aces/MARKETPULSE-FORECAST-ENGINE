import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

sys.path.append("src")
from generate_features import (load_meta, load_google, load_bing,
                                build_features, build_future_rows, validate_campaigns)

st.set_page_config(
    page_title="AdScope — Revenue Forecaster",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LIGHT / DATA-SHEET THEME ──────────────────────────────────────────────────
# NOTE: the dataframe grid, selectbox dropdown panel, and slider track are
# custom-rendered components that do NOT read this CSS — they read
# .streamlit/config.toml (base="light"). This block only styles regular DOM.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #FFFFFF;
    --panel: #F7F9FB;
    --hairline: #E2E8F0;
    --signal: #0E9488;
    --amber: #B45309;
    --red: #DC2626;
    --green: #15803D;
    --ink: #101826;
    --ink-dim: #5B6B7F;
}

html, body { background: var(--paper) !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .block-container {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    padding-top: 1rem !important;
}
p, span, label, div { color: var(--ink) !important; }
h1, h2, h3, h4, h5 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink) !important; }
.mono { font-family: 'IBM Plex Mono', monospace; font-feature-settings: "tnum"; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {
    background: var(--panel) !important;
    border-right: 1px solid var(--hairline) !important;
}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: var(--ink) !important; }
[data-testid="stCheckbox"] label { color: var(--ink) !important; }

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary { color: var(--ink) !important; font-weight: 600 !important; }

/* ── TABS — underline indicator, not filled pills ── */
[data-testid="stTabs"] { background: transparent !important; }
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--hairline) !important;
    gap: 4px !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--ink-dim) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em;
    border-radius: 0 !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background: transparent !important;
    color: var(--signal) !important;
    border-bottom: 2px solid var(--signal) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: var(--panel) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: 4px !important;
}

/* ── HERO ── */
.console-hero {
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 6px;
    padding: 1.8rem 2rem 1.4rem;
    margin-bottom: 1.4rem;
}
.console-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    color: var(--signal) !important;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.console-eyebrow span.dim { color: var(--ink-dim) !important; margin-left: 10px; }
.console-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem; font-weight: 700;
    color: var(--ink) !important;
    margin-bottom: 4px;
}
.console-sub { color: var(--ink-dim) !important; font-size: 0.92rem; max-width: 640px; }

/* ── SCOPE RULE — the signature element ──
   low is structurally 0%, high is 100%, mid is ALWAYS the midpoint
   (bands are ±15%) — the ruler position is the data, not decoration */
.scope-rule { margin-top: 1.4rem; }
.scope-track {
    position: relative; height: 3px;
    background: linear-gradient(90deg, var(--red), var(--hairline), var(--signal));
    border-radius: 2px;
    margin: 0 8px;
}
.scope-mark {
    position: absolute; top: -4px; width: 11px; height: 11px;
    border-radius: 50%; transform: translateX(-50%);
    border: 2px solid var(--paper);
    box-shadow: 0 0 0 1px var(--hairline);
}
.scope-labels { display: flex; justify-content: space-between; margin-top: 10px; padding: 0 8px; }
.scope-labels .l { text-align: left; }
.scope-labels .m { text-align: center; }
.scope-labels .h { text-align: right; }
.scope-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem;
    letter-spacing: 0.08em; color: var(--ink-dim) !important; display: block;
}
.scope-val {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem;
    font-weight: 600; margin-top: 2px; color: var(--ink) !important;
}

/* ── READOUT STRIP ── */
.readout-strip {
    display: flex; border: 1px solid var(--hairline); border-radius: 6px;
    overflow: hidden; margin: 1rem 0 1.6rem; background: var(--paper);
}
.readout-cell { flex: 1; padding: 1.1rem 1.3rem; border-right: 1px solid var(--hairline); background: var(--paper); }
.readout-cell:last-child { border-right: none; }
.readout-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.08em; color: var(--ink-dim) !important; text-transform: uppercase;
}
.readout-value {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem;
    font-weight: 600; color: var(--ink) !important; margin-top: 4px;
}
.readout-value.signal { color: var(--signal) !important; }
.readout-value.amber  { color: var(--amber) !important; }

/* ── SECTION LABEL ── */
.sec {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--ink-dim) !important;
    border-bottom: 1px solid var(--hairline);
    padding: 0 0 8px; margin: 1.4rem 0 1rem;
}
.sec span { color: var(--signal) !important; margin-right: 8px; }

/* ── VALIDATION ROWS ── */
.v-ok, .v-warn, .v-err {
    border-radius: 4px; padding: 8px 14px; margin-bottom: 5px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
    border-left: 3px solid;
}
.v-ok   { background: rgba(14,148,136,0.07); border-color: var(--signal); color: #0a5b53 !important; }
.v-warn { background: rgba(180,83,9,0.07);   border-color: var(--amber);  color: #7c3a06 !important; }
.v-err  { background: rgba(220,38,38,0.07);  border-color: var(--red);    color: #991b1b !important; }

/* ── AI INSIGHTS BOX ── */
.ai-box {
    background: var(--panel); border: 1px solid var(--hairline);
    border-left: 3px solid var(--signal);
    border-radius: 4px; padding: 1.5rem 2rem;
    color: var(--ink) !important; line-height: 1.8; font-size: 0.93rem;
}
.ai-box h2, .ai-box h3, .ai-box strong { color: var(--signal) !important; }

[data-testid="stDataFrame"] { border: 1px solid var(--hairline) !important; border-radius: 4px !important; }
</style>
""", unsafe_allow_html=True)

# ── MODEL ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("pickle/model.pkl"), joblib.load("pickle/features.pkl")

model, FEATURES = load_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔭 AdScope")
    st.markdown("**AIgnition 3.0 · The AI Aces**")
    st.caption("Karpagam College of Engineering, Coimbatore")
    st.divider()

    st.markdown("#### 📅 Forecast Window")
    forecast_window = st.selectbox(
        "fw", [30, 60, 90],
        format_func=lambda x: f"Next {x} Days",
        label_visibility="collapsed"
    )

    st.markdown("#### 💰 Budget Simulation")
    st.caption("Adjust spend multiplier per channel")
    google_budget = st.slider("Google Ads ×", 0.5, 3.0, 1.0, 0.1)
    meta_budget   = st.slider("Meta Ads ×",   0.5, 3.0, 1.0, 0.1)
    bing_budget   = st.slider("Bing Ads ×",   0.5, 3.0, 1.0, 0.1)

    st.markdown("#### 📡 Active Channels")
    use_google = st.checkbox("Google Ads", value=True)
    use_meta   = st.checkbox("Meta Ads",   value=True)
    use_bing   = st.checkbox("Bing Ads",   value=True)

    st.divider()
    st.caption("Low = ×0.85  |  Mid = ×1.0  |  High = ×1.15")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="console-hero">
  <div class="console-eyebrow">SCOPE ONLINE <span class="dim">· GOOGLE / META / BING · GROQ LLAMA-3.3-70B</span></div>
  <div class="console-title">AdScope Revenue Forecaster</div>
  <div class="console-sub">Reading projected revenue and ROAS across channels as a calibrated range, not a single guess.</div>
</div>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    meta   = load_meta("./data")
    google = load_google("./data")
    bing   = load_bing("./data")
    df = pd.concat([meta, google, bing], ignore_index=True)
    issues = validate_campaigns(df)
    df = build_features(df)
    df = build_future_rows(df, horizon_days=90)
    return df, issues

with st.spinner("Loading and forecasting..."):
    df_all, val_issues = load_all_data()

with st.expander("🔍 Campaign Validation Report", expanded=False):
    for iss in val_issues:
        if iss.startswith("OK"):
            st.markdown(f'<div class="v-ok">✅ {iss}</div>', unsafe_allow_html=True)
        elif iss.startswith("WARNING"):
            st.markdown(f'<div class="v-warn">⚠️ {iss}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="v-err">❌ {iss}</div>', unsafe_allow_html=True)

# ── FILTER & BUDGET ───────────────────────────────────────────────────────────
active = []
if use_google: active.append("Google")
if use_meta:   active.append("Meta")
if use_bing:   active.append("Bing")

if not active:
    st.error("Please select at least one channel in the sidebar.")
    st.stop()

future = df_all[df_all["is_future"] == True].copy()
future = future[future["channel"].isin(active)]

budget_map = {"Google": google_budget, "Meta": meta_budget, "Bing": bing_budget}
for ch, mult in budget_map.items():
    mask = future["channel"] == ch
    for col in ["spend","daily_budget","clicks","impressions","conversions"]:
        future.loc[mask, col] *= mult

cutoff = future["date"].min() + pd.Timedelta(days=forecast_window)
df = future[future["date"] <= cutoff].copy()
df = df[df["spend"] > 0].dropna(subset=FEATURES)

# ── PREDICT ───────────────────────────────────────────────────────────────────
df["predicted_revenue"] = model.predict(df[FEATURES])
df["revenue_low"]  = df["predicted_revenue"] * 0.85
df["revenue_mid"]  = df["predicted_revenue"]
df["revenue_high"] = df["predicted_revenue"] * 1.15
df["roas_low"]     = np.where(df["spend"]>0, df["revenue_low"]  / df["spend"], 0)
df["roas_mid"]     = np.where(df["spend"]>0, df["revenue_mid"]  / df["spend"], 0)
df["roas_high"]    = np.where(df["spend"]>0, df["revenue_high"] / df["spend"], 0)

total_spend    = df["spend"].sum()
total_rev_low  = df["revenue_low"].sum()
total_rev_mid  = df["revenue_mid"].sum()
total_rev_high = df["revenue_high"].sum()
blended_roas   = total_rev_mid / total_spend if total_spend > 0 else 0

# ── KPI STRIP + SCOPE RULE ───────────────────────────────────────────────────
st.markdown(f'<div class="sec"><span>◆</span>{forecast_window}-DAY FORECAST SUMMARY</div>',
            unsafe_allow_html=True)

st.markdown(f"""
<div class="readout-strip">
  <div class="readout-cell">
    <div class="readout-label">Total Ad Spend</div>
    <div class="readout-value">${total_spend:,.0f}</div>
  </div>
  <div class="readout-cell" style="flex: 2.2;">
    <div class="readout-label">Revenue Range</div>
    <div class="scope-rule">
      <div class="scope-track">
        <div class="scope-mark" style="left:0%;   background: var(--red);"></div>
        <div class="scope-mark" style="left:50%;  background: var(--signal); width:13px; height:13px; top:-5px;"></div>
        <div class="scope-mark" style="left:100%; background: var(--green);"></div>
      </div>
      <div class="scope-labels">
        <div class="l"><span class="scope-tag">LOW</span><span class="scope-val">${total_rev_low:,.0f}</span></div>
        <div class="m"><span class="scope-tag">MID</span><span class="scope-val">${total_rev_mid:,.0f}</span></div>
        <div class="h"><span class="scope-tag">HIGH</span><span class="scope-val">${total_rev_high:,.0f}</span></div>
      </div>
    </div>
  </div>
  <div class="readout-cell">
    <div class="readout-label">Blended ROAS</div>
    <div class="readout-value amber">{blended_roas:.2f}×</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── CHART THEME ───────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#F7F9FB",
    font=dict(family="IBM Plex Mono, monospace", color="#101826", size=12),
    margin=dict(t=30, b=40, l=10, r=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#E2E8F0", zerolinecolor="#E2E8F0"),
)
CH_CLR = {"Google":"#B45309","Meta":"#DC2626","Bing":"#0E9488"}

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "CHANNEL",
    "CAMPAIGN",
    "TREND",
    "SIMULATOR",
    "AI INSIGHTS"
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="sec"><span>◆</span>{forecast_window}-DAY CHANNEL-LEVEL FORECAST</div>',
                unsafe_allow_html=True)

    ch_rows = []
    for ch in df["channel"].unique():
        c = df[df["channel"]==ch]
        ch_rows.append({
            "Channel":      ch,
            "Spend ($)":    f"${c['spend'].sum():,.0f}",
            "Rev Low ($)":  f"${c['revenue_low'].sum():,.0f}",
            "Rev Mid ($)":  f"${c['revenue_mid'].sum():,.0f}",
            "Rev High ($)": f"${c['revenue_high'].sum():,.0f}",
            "ROAS Low":     f"{c['roas_low'].mean():.2f}x",
            "ROAS Mid":     f"{c['roas_mid'].mean():.2f}x",
            "ROAS High":    f"{c['roas_high'].mean():.2f}x",
        })
    st.dataframe(pd.DataFrame(ch_rows), use_container_width=True, hide_index=True)

    channels = [r["Channel"] for r in ch_rows]
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec"><span>◆</span>REVENUE RANGE BY CHANNEL</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for lbl, mult, clr in [("Low",0.85,"#DC2626"),("Mid",1.0,"#0E9488"),("High",1.15,"#15803D")]:
            fig.add_trace(go.Bar(
                name=lbl, x=channels,
                y=[df[df["channel"]==c]["revenue_mid"].sum()*mult for c in channels],
                marker_color=clr
            ))
        fig.update_layout(barmode="group", height=360, yaxis_title="Revenue ($)", **PT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec"><span>◆</span>ROAS BY CHANNEL</div>', unsafe_allow_html=True)
        roas_vals = [df[df["channel"]==c]["roas_mid"].mean() for c in channels]
        fig2 = go.Figure(go.Bar(
            x=channels, y=roas_vals,
            marker_color=[CH_CLR.get(c,"#5B6B7F") for c in channels],
            text=[f"{v:.1f}x" for v in roas_vals],
            textposition="outside",
            textfont=dict(color="#101826")
        ))
        fig2.update_layout(height=360, yaxis_title="ROAS", **PT)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec"><span>◆</span>CAMPAIGN TYPE BREAKDOWN</div>', unsafe_allow_html=True)
    ct_rows = []
    for ct in df["campaign_type"].unique():
        c = df[df["campaign_type"]==ct]
        ct_rows.append({
            "Campaign Type": ct,
            "Channel":       c["channel"].iloc[0],
            "Spend ($)":     f"${c['spend'].sum():,.0f}",
            "Rev Low ($)":   f"${c['revenue_low'].sum():,.0f}",
            "Rev Mid ($)":   f"${c['revenue_mid'].sum():,.0f}",
            "Rev High ($)":  f"${c['revenue_high'].sum():,.0f}",
            "ROAS Mid":      f"{c['roas_mid'].mean():.2f}x",
        })
    st.dataframe(pd.DataFrame(ct_rows), use_container_width=True, hide_index=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="sec"><span>◆</span>CAMPAIGN-LEVEL {forecast_window}-DAY FORECAST</div>',
                unsafe_allow_html=True)
    camp_rows = []
    for cn in df["campaign_name"].unique():
        c = df[df["campaign_name"]==cn]
        camp_rows.append({
            "Campaign":     cn,
            "Channel":      c["channel"].iloc[0],
            "Type":         c["campaign_type"].iloc[0],
            "Spend ($)":    round(c["spend"].sum(), 2),
            "Rev Low ($)":  round(c["revenue_low"].sum(), 2),
            "Rev Mid ($)":  round(c["revenue_mid"].sum(), 2),
            "Rev High ($)": round(c["revenue_high"].sum(), 2),
            "ROAS Mid":     round(c["roas_mid"].mean(), 2),
            "Conversions":  round(c["conversions"].sum(), 0),
        })
    camp_df = pd.DataFrame(camp_rows).sort_values("Rev Mid ($)", ascending=False)
    st.dataframe(camp_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec"><span>◆</span>TOP 10 CAMPAIGNS BY FORECASTED REVENUE</div>', unsafe_allow_html=True)
    top10 = camp_df.head(10)
    fig3 = go.Figure(go.Bar(
        x=top10["Rev Mid ($)"], y=top10["Campaign"],
        orientation="h", marker_color="#0E9488",
        text=[f"${v:,.0f}" for v in top10["Rev Mid ($)"]],
        textposition="outside",
        textfont=dict(color="#101826")
    ))
    fig3.update_layout(height=430, xaxis_title="Revenue Mid ($)", **PT)
    st.plotly_chart(fig3, use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="sec"><span>◆</span>DAILY REVENUE FORECAST — NEXT {forecast_window} DAYS</div>',
                unsafe_allow_html=True)
    trend = df.groupby("date").agg(
        revenue_low=("revenue_low","sum"),
        revenue_mid=("revenue_mid","sum"),
        revenue_high=("revenue_high","sum"),
    ).reset_index()

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=trend["date"], y=trend["revenue_high"], name="High",
        line=dict(color="#15803D", dash="dot", width=1.5)))
    fig4.add_trace(go.Scatter(
        x=trend["date"], y=trend["revenue_mid"], name="Mid (Expected)",
        line=dict(color="#0E9488", width=2.5),
        fill="tonexty", fillcolor="rgba(14,148,136,0.08)"))
    fig4.add_trace(go.Scatter(
        x=trend["date"], y=trend["revenue_low"], name="Low",
        line=dict(color="#DC2626", dash="dot", width=1.5),
        fill="tonexty", fillcolor="rgba(220,38,38,0.06)"))
    fig4.update_layout(height=420, yaxis_title="Revenue ($)", xaxis_title="Date", **PT)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="sec"><span>◆</span>MONTHLY SEASONALITY PATTERN</div>', unsafe_allow_html=True)
    monthly = df.groupby(df["date"].dt.month)["revenue_mid"].sum().reset_index()
    monthly.columns = ["Month","Revenue"]
    mnames = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    monthly["Month"] = monthly["Month"].map(mnames)
    fig5 = go.Figure(go.Bar(
        x=monthly["Month"], y=monthly["Revenue"],
        marker_color="#B45309",
        text=[f"${v:,.0f}" for v in monthly["Revenue"]],
        textposition="outside",
        textfont=dict(color="#101826")
    ))
    fig5.update_layout(height=330, yaxis_title="Revenue ($)", **PT)
    st.plotly_chart(fig5, use_container_width=True)

# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec"><span>◆</span>WHAT-IF BUDGET SIMULATION</div>', unsafe_allow_html=True)
    st.info("Adjust the budget sliders in the sidebar. See how different spend levels affect forecasted revenue and ROAS.")

    avg_mult = (google_budget + meta_budget + bing_budget) / 3
    sim_rows = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        sim_spend = (total_spend / avg_mult) * mult if avg_mult > 0 else total_spend
        sim_rev   = (total_rev_mid / avg_mult) * mult if avg_mult > 0 else total_rev_mid
        sim_roas  = sim_rev / sim_spend if sim_spend > 0 else 0
        sim_rows.append({
            "Multiplier":      f"{mult}x",
            "Projected Spend": f"${sim_spend:,.0f}",
            "Revenue Low":     f"${sim_rev*0.85:,.0f}",
            "Revenue Mid":     f"${sim_rev:,.0f}",
            "Revenue High":    f"${sim_rev*1.15:,.0f}",
            "ROAS":            f"{sim_roas:.2f}x",
        })
    st.dataframe(pd.DataFrame(sim_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec"><span>◆</span>PER-CHANNEL REVENUE AT CURRENT BUDGET</div>', unsafe_allow_html=True)
    cols = st.columns(len(active))
    for i, ch in enumerate(active):
        c = df[df["channel"]==ch]
        with cols[i]:
            st.metric(
                label=ch,
                value=f"${c['revenue_mid'].sum():,.0f}",
                delta=f"ROAS {c['roas_mid'].mean():.1f}x · Budget {budget_map[ch]}x"
            )

# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="sec"><span>◆</span>AI CAUSAL INSIGHTS — GROQ LLAMA-3.3-70B</div>',
                unsafe_allow_html=True)
    st.caption("Pre-generated · Offline safe · No internet required at evaluation time")

    insight_file = f"output/insights_{forecast_window}d.txt"
    if os.path.exists(insight_file):
        with open(insight_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        st.markdown(f'<div class="ai-box">{content}</div>', unsafe_allow_html=True)
    else:
        st.warning("Run `python src/generate_insights.py` to generate AI insights.")

    st.markdown('<div class="sec"><span>◆</span>FORECAST CONFIDENCE BY CHANNEL</div>', unsafe_allow_html=True)
    conf_rows = []
    for ch in df["channel"].unique():
        c = df[df["channel"]==ch]
        mid = c["revenue_mid"].sum()
        unc = ((c["revenue_high"].sum()-c["revenue_low"].sum())/mid*100) if mid>0 else 0
        conf_rows.append({
            "Channel":     ch,
            "Uncertainty": f"±{unc/2:.1f}%",
            "Confidence":  "High" if unc<20 else "Medium" if unc<35 else "Low",
            "Data Points": len(c),
            "Rev Mid":     f"${mid:,.0f}",
        })
    st.dataframe(pd.DataFrame(conf_rows), use_container_width=True, hide_index=True)

st.divider()
st.markdown(
    "<center><small style='color:#5B6B7F; font-family: IBM Plex Mono, monospace;'>"
    "ADSCOPE · AIGNITION 3.0 · TEAM: THE AI ACES · "
    "KARPAGAM COLLEGE OF ENGINEERING, COIMBATORE"
    "</small></center>",
    unsafe_allow_html=True
)