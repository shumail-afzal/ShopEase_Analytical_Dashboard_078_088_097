# ============================================================
# SHOPEASE ANALYTICAL DASHBOARD
# Group ID: 078_088_097 | Fixed Seed: 78088097
# Run: streamlit run stream_project.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly is used for hover, zoom, drill-down and interactive charts.
# Install once if needed: pip install plotly
import plotly.express as px
import plotly.graph_objects as go

# -------------------------- CONFIG ---------------------------
GROUP_ID = "078_088_097"
SEED = 78088097
SAMPLE_SIZE = 2500

random.seed(SEED)
np.random.seed(SEED)

st.set_page_config(
    page_title="ShopEase Analytical Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- STYLE ---------------------------
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 8% 2%, rgba(88, 86, 214, .10), transparent 22rem),
            radial-gradient(circle at 95% 8%, rgba(0, 173, 181, .08), transparent 24rem);
    }
    .block-container {padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1500px;}
    .hero {
        padding: 1.45rem 1.55rem;
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 22px;
        margin-bottom: 1rem;
        background: linear-gradient(120deg, rgba(88,86,214,.13), rgba(0,173,181,.08), rgba(127,127,127,.03));
    }
    .hero h1 {margin:0; font-size:2.15rem; letter-spacing:-.03em;}
    .hero p {margin:.38rem 0 0 0; opacity:.78; font-size:1.02rem;}
    .insight {
        padding: .9rem 1rem;
        border-left: 4px solid rgba(88,86,214,.75);
        background: rgba(127,127,127,.07);
        border-radius: 10px;
        margin: .45rem 0;
    }
    .small-note {opacity:.72; font-size:.86rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: 13px 15px;
        border-radius: 16px;
        background: rgba(127,127,127,.04);
        box-shadow: 0 5px 20px rgba(0,0,0,.035);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        transition: .18s ease;
        border-color: rgba(88,86,214,.40);
    }

    /* CONTROL CENTER */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(31,34,48,.98), rgba(20,22,32,.98));
        border-right: 1px solid rgba(255,255,255,.08);
    }
    section[data-testid="stSidebar"] * {color: #F5F7FB;}
    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div {
        background: rgba(255,255,255,.07);
        border-radius: 12px;
        border-color: rgba(255,255,255,.12);
    }
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background: rgba(105,101,255,.70);
    }
    .control-hero {
        padding: 1rem;
        border-radius: 16px;
        background: linear-gradient(135deg, rgba(105,101,255,.35), rgba(0,173,181,.22));
        border: 1px solid rgba(255,255,255,.12);
        margin: .25rem 0 .9rem 0;
    }
    .control-hero h3 {margin:0; color:white;}
    .control-hero p {margin:.3rem 0 0 0; color:rgba(255,255,255,.72); font-size:.82rem;}
    .filter-badge {
        padding:.55rem .7rem; border-radius:10px; margin-bottom:.55rem;
        background:rgba(255,255,255,.055); border:1px solid rgba(255,255,255,.08);
        font-size:.8rem; color:rgba(255,255,255,.76);
    }
    div[data-testid="stTabs"] button {font-weight:650;}
</style>
""", unsafe_allow_html=True)

# ----------------------- DATA PIPELINE -----------------------
def parse_discount(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    try:
        return float(s[:-1]) / 100 if s.endswith("%") else float(s)
    except Exception:
        return np.nan

@st.cache_data
def load_and_prepare(path):
    raw = pd.read_csv(path)
    original_rows = len(raw)

    # Clean text
    text_cols = ["Gender", "City", "Category", "Product",
                 "PaymentMethod", "OrderStatus"]
    for c in text_cols:
        raw[c] = raw[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)

    # Numeric conversion
    for c in ["CustomerAge", "Quantity", "UnitPrice", "Rating", "TotalAmount"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw["Discount"] = raw["Discount"].apply(parse_discount)

    # Dates
    raw["OrderDate"] = pd.to_datetime(raw["OrderDate"], errors="coerce")
    raw["DeliveryDate"] = pd.to_datetime(raw["DeliveryDate"], errors="coerce")

    # Data quality flags BEFORE cleaning
    raw["InvalidQuantity"] = raw["Quantity"].isna() | (raw["Quantity"] <= 0)
    raw["InvalidPrice"] = raw["UnitPrice"].isna() | (raw["UnitPrice"] <= 0)
    raw["InvalidAge"] = raw["CustomerAge"].isna() | ~raw["CustomerAge"].between(15, 100)
    raw["InvalidDiscount"] = raw["Discount"].isna() | ~raw["Discount"].between(0, 1)
    raw["InvalidRating"] = raw["Rating"].notna() & ~raw["Rating"].between(1, 5)
    raw["MissingOrderDate"] = raw["OrderDate"].isna()

    quality_flags = [
        "InvalidQuantity", "InvalidPrice", "InvalidAge",
        "InvalidDiscount", "InvalidRating", "MissingOrderDate"
    ]
    raw["QualityIssue"] = raw[quality_flags].any(axis=1)

    # Keep valid analytical rows
    clean = raw[
        (~raw["InvalidQuantity"]) &
        (~raw["InvalidPrice"]) &
        (~raw["InvalidDiscount"]) &
        (~raw["MissingOrderDate"])
    ].copy()

    # Fill non-critical missing values
    clean["CustomerAge"] = clean["CustomerAge"].where(clean["CustomerAge"].between(15,100))
    clean["Rating"] = clean["Rating"].where(clean["Rating"].between(1,5))

    # Derived variables
    clean["GrossSales"] = clean["Quantity"] * clean["UnitPrice"]
    clean["NetSales"] = clean["GrossSales"] * (1 - clean["Discount"])
    clean["DiscountValue"] = clean["GrossSales"] - clean["NetSales"]
    clean["DeliveryDays"] = (clean["DeliveryDate"] - clean["OrderDate"]).dt.total_seconds() / 86400
    clean.loc[clean["DeliveryDays"] < 0, "DeliveryDays"] = np.nan

    clean["YearMonth"] = clean["OrderDate"].dt.to_period("M").astype(str)
    clean["Month"] = clean["OrderDate"].dt.month_name().str[:3]
    clean["Weekday"] = clean["OrderDate"].dt.day_name().str[:3]
    clean["Hour"] = clean["OrderDate"].dt.hour

    clean["AgeBand"] = pd.cut(
        clean["CustomerAge"],
        bins=[0, 24, 34, 44, 54, 64, 200],
        labels=["≤24", "25–34", "35–44", "45–54", "55–64", "65+"]
    )

    clean["DiscountBand"] = pd.cut(
        clean["Discount"],
        bins=[-0.001, 0, .05, .10, .20, 1],
        labels=["0%", "1–5%", "6–10%", "11–20%", "20%+"]
    )

    # Fixed sample AFTER deterministic cleaning
    if len(clean) < SAMPLE_SIZE:
        raise ValueError(f"Only {len(clean)} valid rows remain; cannot sample {SAMPLE_SIZE}.")
    sample = clean.sample(n=SAMPLE_SIZE, random_state=SEED).copy()
    sample = sample.sort_values("OrderDate").reset_index(drop=True)

    quality_summary = {
        "original_rows": original_rows,
        "valid_rows": len(clean),
        "sample_rows": len(sample),
        "quality_issue_rows_raw": int(raw["QualityIssue"].sum()),
        "duplicates_raw": int(raw.duplicated().sum())
    }
    return sample, raw, quality_summary

try:
    df, raw_df, quality = load_and_prepare("shopease_raw_orders.csv")
except FileNotFoundError:
    st.error("Place `shopease_raw_orders.csv` in the same folder as this Python file.")
    st.stop()
except Exception as e:
    st.error(f"Data preparation error: {e}")
    st.stop()

# ----------------------- SIDEBAR FILTERS ---------------------
st.sidebar.markdown("""
<div class="control-hero">
    <h3>🎛️ Dashboard Controls</h3>
    <p>Use these filters to explore the dataset and update the dashboard visuals.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    f'<div class="filter-badge">🔐 Group <b>{GROUP_ID}</b> &nbsp;•&nbsp; Seed <b>{SEED}</b><br>'
    f'🎯 Reproducible analytical sample: <b>{SAMPLE_SIZE:,}</b> rows</div>',
    unsafe_allow_html=True
)

with st.sidebar.expander("📅 TIME WINDOW", expanded=True):
    min_date = df["OrderDate"].min().date()
    max_date = df["OrderDate"].max().date()
    date_range = st.date_input(
        "Order date range", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

def multi(label, col, parent=st.sidebar):
    opts = sorted(df[col].dropna().astype(str).unique().tolist())
    return parent.multiselect(label, opts, default=opts)

with st.sidebar.expander("🧩 CATEGORICAL FILTERS", expanded=True):
    categories = multi("Product category", "Category", st)
    cities = multi("Customer city", "City", st)
    genders = multi("Gender", "Gender", st)

with st.sidebar.expander("🛒 TRANSACTION FILTERS", expanded=False):
    statuses = multi("Order status", "OrderStatus", st)
    payments = multi("Payment method", "PaymentMethod", st)

with st.sidebar.expander("📊 VISUAL SETTINGS", expanded=True):
    top_n = st.slider("Top N entities", 3, 15, 7)
    metric_choice = st.selectbox(
        "Primary business metric",
        ["Net Sales", "Orders", "Customers", "Quantity", "Average Rating"]
    )
    chart_theme = st.selectbox(
        "Chart appearance",
        ["plotly_dark", "plotly_white", "presentation"],
        index=0,
        format_func=lambda x: {
            "plotly_dark":"🌙 Executive Dark",
            "plotly_white":"☀️ Clean Light",
            "presentation":"🎤 Presentation"
        }[x]
    )

filtered = df.copy()
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
    filtered = filtered[(filtered["OrderDate"] >= start) & (filtered["OrderDate"] < end)]

filtered = filtered[
    filtered["Category"].astype(str).isin(categories) &
    filtered["City"].astype(str).isin(cities) &
    filtered["Gender"].astype(str).isin(genders) &
    filtered["OrderStatus"].astype(str).isin(statuses) &
    filtered["PaymentMethod"].astype(str).isin(payments)
].copy()

st.sidebar.markdown(
    f'<div class="filter-badge">👁️ Current view: <b>{len(filtered):,}</b> records'
    f'<br>Filters update every dashboard tab instantly.</div>',
    unsafe_allow_html=True
)

if filtered.empty:
    st.warning("No records match the current filters. Change the selections in the Control Center.")
    st.stop()

# ----------------------- HELPER FUNCTIONS --------------------
def safe_pct(a, b):
    return (a / b * 100) if b not in [0, None] and pd.notna(b) else 0

def metric_series(data, group):
    if metric_choice == "Net Sales":
        return data.groupby(group)["NetSales"].sum()
    if metric_choice == "Orders":
        return data.groupby(group)["OrderID"].nunique()
    if metric_choice == "Customers":
        return data.groupby(group)["CustomerID"].nunique()
    if metric_choice == "Quantity":
        return data.groupby(group)["Quantity"].sum()
    return data.groupby(group)["Rating"].mean()

def fmt_money(x):
    return f"₹{x:,.0f}"

def plotly_layout(fig, height=410):
    fig.update_layout(
        template=chart_theme,
        height=height,
        margin=dict(l=15, r=15, t=52, b=18),
        legend_title_text="",
        hovermode="closest",
        title_font=dict(size=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def business_insights(data):
    out = []
    if len(data):
        cat = data.groupby("Category")["NetSales"].sum().sort_values(ascending=False)
        if len(cat):
            out.append(f"🏆 **{cat.index[0]}** is the leading category with {safe_pct(cat.iloc[0], cat.sum()):.1f}% of filtered sales.")
        city = data.groupby("City")["NetSales"].sum().sort_values(ascending=False)
        if len(city):
            out.append(f"📍 **{city.index[0]}** generates the highest sales among selected cities ({fmt_money(city.iloc[0])}).")
        if data["Rating"].notna().any():
            r = data.groupby("Category")["Rating"].mean().dropna().sort_values(ascending=False)
            if len(r):
                out.append(f"⭐ **{r.index[0]}** has the strongest average rating at {r.iloc[0]:.2f}/5.")
        d = data.groupby("DiscountBand", observed=True)["NetSales"].mean().dropna().sort_values(ascending=False)
        if len(d):
            out.append(f"🏷️ Orders in the **{d.index[0]} discount band** have the highest average order-level sales ({fmt_money(d.iloc[0])}).")
        delivered = data[data["OrderStatus"].str.lower().eq("delivered")]
        if delivered["DeliveryDays"].notna().any():
            out.append(f"🚚 Delivered orders take a median **{delivered['DeliveryDays'].median():.1f} days** in the current view.")
    return out

# ---------------------------- HERO ---------------------------
st.markdown("""
<div class="hero">
<h1>🛍️ ShopEase Analytical Dashboard</h1>
<p>Interactive analysis of customer, sales, product and order data.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    f"Group ID: {GROUP_ID}  •  Seed: {SEED}  •  "
    f"Showing {len(filtered):,} of the fixed {len(df):,}-record analytical sample"
)
st.markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 16px 0;">
  <span style="padding:6px 10px;border-radius:10px;background:rgba(88,86,214,.10);">📊 Bar → Categories</span>
  <span style="padding:6px 10px;border-radius:10px;background:rgba(0,173,181,.10);">📈 Line → Time</span>
  <span style="padding:6px 10px;border-radius:10px;background:rgba(230,150,50,.10);">🔵 Scatter → Numerical</span>
  <span style="padding:6px 10px;border-radius:10px;background:rgba(150,90,180,.10);">🔥 Heat Map → Patterns</span>
</div>
""", unsafe_allow_html=True)


# ----------------------------- TABS --------------------------
overview, visual_lab, customers, products, operations, data_lab = st.tabs([
    "🏠 Overview",
    "✨ Visual Analytics",
    "👥 Customer Analysis",
    "🧺 Product Analysis",
    "🚚 Order & Delivery Analysis",
    "🧹 Data Quality"
])

# ============================================================
# TAB 1 — EXECUTIVE OVERVIEW
# ============================================================
with overview:
    sales = filtered["NetSales"].sum()
    orders = filtered["OrderID"].nunique()
    customers_n = filtered["CustomerID"].nunique()
    aov = sales / orders if orders else 0
    avg_rating = filtered["Rating"].mean()
    delivered_rate = safe_pct(
        filtered["OrderStatus"].str.lower().eq("delivered").sum(), len(filtered)
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Net Sales", fmt_money(sales))
    c2.metric("Orders", f"{orders:,}")
    c3.metric("Customers", f"{customers_n:,}")
    c4.metric("AOV", fmt_money(aov))
    c5.metric("Avg Rating", f"{avg_rating:.2f}/5" if pd.notna(avg_rating) else "N/A")
    c6.metric("Delivered", f"{delivered_rate:.1f}%")

    st.subheader("💡 Key Findings")
    insight_cols = st.columns(2)
    for i, insight in enumerate(business_insights(filtered)):
        with insight_cols[i % 2]:
            st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.6, 1])
    with left:
        trend_metric = st.selectbox(
            "Trend metric", ["NetSales", "Orders", "Quantity"],
            format_func=lambda x: {"NetSales":"Net Sales","Orders":"Orders","Quantity":"Units"}[x],
            key="trend_metric"
        )
        temp = filtered.copy()
        temp["Date"] = temp["OrderDate"].dt.date
        if trend_metric == "NetSales":
            trend = temp.groupby("Date")["NetSales"].sum().reset_index()
            y = "NetSales"
        elif trend_metric == "Orders":
            trend = temp.groupby("Date")["OrderID"].nunique().reset_index(name="Orders")
            y = "Orders"
        else:
            trend = temp.groupby("Date")["Quantity"].sum().reset_index()
            y = "Quantity"
        fig = px.area(trend, x="Date", y=y, title=f"{trend_metric.replace('NetSales','Net Sales')} over time")
        plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        cat = metric_series(filtered, "Category").sort_values(ascending=False).reset_index(name="Value")
        fig = px.pie(cat, names="Category", values="Value", hole=.55, title=f"{metric_choice} contribution")
        fig.update_traces(textposition="inside", textinfo="percent+label")
        plotly_layout(fig, 390)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        rank_dimension = st.selectbox("Rank by", ["Product", "City", "PaymentMethod"], key="rank_dim")
        rank = metric_series(filtered, rank_dimension).sort_values(ascending=False).head(top_n).reset_index(name="Value")
        fig = px.bar(rank, x="Value", y=rank_dimension, orientation="h",
                     title=f"Top {top_n} {rank_dimension}s by {metric_choice}")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        status = filtered["OrderStatus"].value_counts().reset_index()
        status.columns = ["Status", "Orders"]
        fig = px.bar(status, x="Status", y="Orders", title="Order health")
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# VISUAL ANALYTICS — SIR-TAUGHT CHART TYPES
# ============================================================
with visual_lab:
    st.subheader("✨ Visual Analysis")
    st.caption(
        "A dedicated visual layer for categorical and non-categorical analysis: "
        "bar chart, line chart, scatter plot and heat maps."
    )

    # BAR: categorical vs numerical
    st.markdown("### 1. Bar Chart — Categorical Comparison")
    bc1, bc2 = st.columns([1, 1])
    with bc1:
        bar_dimension = st.selectbox(
            "Categorical variable",
            ["Category", "City", "PaymentMethod", "OrderStatus", "Gender"],
            key="bar_dimension"
        )
    with bc2:
        bar_measure = st.selectbox(
            "Measure",
            ["Net Sales", "Orders", "Quantity", "Average Rating"],
            key="bar_measure"
        )

    if bar_measure == "Net Sales":
        bar_data = filtered.groupby(bar_dimension)["NetSales"].sum()
    elif bar_measure == "Orders":
        bar_data = filtered.groupby(bar_dimension)["OrderID"].nunique()
    elif bar_measure == "Quantity":
        bar_data = filtered.groupby(bar_dimension)["Quantity"].sum()
    else:
        bar_data = filtered.groupby(bar_dimension)["Rating"].mean()

    bar_data = bar_data.sort_values(ascending=False).head(top_n).reset_index(name="Value")
    fig = px.bar(
        bar_data, x=bar_dimension, y="Value", text_auto=".3s",
        title=f"{bar_measure} by {bar_dimension}",
        hover_data={"Value":":,.2f"}
    )
    plotly_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

    # LINE: time series
    st.markdown("### 2. Line Chart — Time-Series Trend")
    line_measure = st.radio(
        "Trend measure", ["Net Sales", "Orders", "Quantity"],
        horizontal=True, key="visual_line"
    )
    timeline = filtered.assign(Date=filtered["OrderDate"].dt.date)
    if line_measure == "Net Sales":
        line_data = timeline.groupby("Date")["NetSales"].sum().reset_index(name="Value")
    elif line_measure == "Orders":
        line_data = timeline.groupby("Date")["OrderID"].nunique().reset_index(name="Value")
    else:
        line_data = timeline.groupby("Date")["Quantity"].sum().reset_index(name="Value")

    fig = px.line(
        line_data, x="Date", y="Value", markers=True,
        title=f"{line_measure} movement across time"
    )
    fig.update_traces(line=dict(width=3))
    plotly_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

    # SCATTER: numeric vs numeric
    st.markdown("### 3. Scatter Plot — Numerical Relationship")
    numeric_scatter = ["CustomerAge", "Quantity", "UnitPrice", "Discount", "Rating", "NetSales", "DeliveryDays"]
    sc1, sc2, sc3 = st.columns(3)
    scatter_x = sc1.selectbox("X-axis numerical variable", numeric_scatter, index=2, key="scatter_x")
    scatter_y = sc2.selectbox("Y-axis numerical variable", numeric_scatter, index=5, key="scatter_y")
    scatter_color = sc3.selectbox(
        "Colour by category", ["Category", "Gender", "PaymentMethod", "OrderStatus"],
        key="scatter_color"
    )
    scatter_data = filtered[[scatter_x, scatter_y, scatter_color, "Product", "OrderID"]].dropna()
    fig = px.scatter(
        scatter_data, x=scatter_x, y=scatter_y, color=scatter_color,
        hover_name="Product", hover_data=["OrderID"],
        opacity=.72, title=f"{scatter_x} vs {scatter_y} • segmented by {scatter_color}"
    )
    plotly_layout(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    # HEAT MAPS: categorical + numeric and pure numeric correlation
    st.markdown("### 4. Heat Maps — Pattern & Correlation Intelligence")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown("##### Categorical Heat Map")
        heat_row = st.selectbox(
            "Rows", ["Category", "PaymentMethod", "OrderStatus", "Gender"],
            index=0, key="heat_row"
        )
        heat_col = st.selectbox(
            "Columns", ["City", "PaymentMethod", "OrderStatus", "Gender"],
            index=0, key="heat_col"
        )
        heat_measure = st.selectbox(
            "Cell measure", ["Net Sales", "Orders", "Quantity"],
            key="heat_measure"
        )

        if heat_measure == "Net Sales":
            hp = filtered.pivot_table(index=heat_row, columns=heat_col, values="NetSales", aggfunc="sum", fill_value=0)
        elif heat_measure == "Orders":
            hp = filtered.pivot_table(index=heat_row, columns=heat_col, values="OrderID", aggfunc=pd.Series.nunique, fill_value=0)
        else:
            hp = filtered.pivot_table(index=heat_row, columns=heat_col, values="Quantity", aggfunc="sum", fill_value=0)

        fig = px.imshow(
            hp, aspect="auto", text_auto=".2s",
            title=f"{heat_measure}: {heat_row} × {heat_col}",
            labels=dict(color=heat_measure)
        )
        plotly_layout(fig, 500)
        st.plotly_chart(fig, use_container_width=True)

    with h2:
        st.markdown("##### Numerical Correlation Heat Map")
        corr_cols = st.multiselect(
            "Numerical variables",
            ["CustomerAge","Quantity","UnitPrice","Discount","Rating","NetSales","DeliveryDays"],
            default=["CustomerAge","Quantity","UnitPrice","Discount","Rating","NetSales"],
            key="visual_corr_cols"
        )
        if len(corr_cols) >= 2:
            corr_matrix = filtered[corr_cols].corr()
            fig = px.imshow(
                corr_matrix, text_auto=".2f", zmin=-1, zmax=1,
                color_continuous_scale="RdBu_r",
                title="Correlation strength between numerical variables"
            )
            plotly_layout(fig, 500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least two numerical variables.")

    st.info(
        "Chart logic: bar charts compare categories; line charts show ordered time trends; "
        "scatter plots examine relationships between numerical variables; heat maps reveal "
        "category combinations and numerical correlations."
    )

# ============================================================
# TAB 2 — CUSTOMER INTELLIGENCE
# ============================================================
with customers:
    st.subheader("Customer Analysis")
    st.caption("Understand who buys, how much they spend, and which customers matter most.")

    cust = filtered.groupby("CustomerID").agg(
        Orders=("OrderID", "nunique"),
        Monetary=("NetSales", "sum"),
        Units=("Quantity", "sum"),
        AvgRating=("Rating", "mean"),
        LastOrder=("OrderDate", "max")
    ).reset_index()
    ref_date = filtered["OrderDate"].max() + pd.Timedelta(days=1)
    cust["Recency"] = (ref_date - cust["LastOrder"]).dt.days

    # RFM scores, robust to ties
    if len(cust) >= 5:
        cust["R"] = pd.qcut(cust["Recency"].rank(method="first"), 4, labels=[4,3,2,1]).astype(int)
        cust["F"] = pd.qcut(cust["Orders"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
        cust["M"] = pd.qcut(cust["Monetary"].rank(method="first"), 4, labels=[1,2,3,4]).astype(int)
        cust["RFMScore"] = cust["R"] + cust["F"] + cust["M"]
        cust["Segment"] = pd.cut(
            cust["RFMScore"], bins=[0,5,8,10,12],
            labels=["At Risk / Low", "Regular", "Loyal", "Champions"]
        )
    else:
        cust["Segment"] = "Insufficient data"

    a, b, c = st.columns(3)
    a.metric("Unique Customers", f"{len(cust):,}")
    b.metric("Sales / Customer", fmt_money(cust["Monetary"].mean()))
    c.metric("Repeat Customers", f"{safe_pct((cust['Orders']>1).sum(),len(cust)):.1f}%")

    l, r = st.columns(2)
    with l:
        age_sales = filtered.groupby("AgeBand", observed=True).agg(
            Sales=("NetSales","sum"), Customers=("CustomerID","nunique")
        ).reset_index()
        fig = px.bar(age_sales, x="AgeBand", y="Sales", hover_data=["Customers"],
                     title="Sales by age segment")
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        seg = cust.groupby("Segment", observed=True).agg(
            Customers=("CustomerID","count"), Sales=("Monetary","sum")
        ).reset_index()
        fig = px.treemap(seg, path=["Segment"], values="Sales", color="Customers",
                         title="RFM customer value map")
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Customer value explorer")
    min_orders = st.slider("Minimum customer orders", 1, max(1, int(cust["Orders"].max())), 1)
    show_cust = cust[cust["Orders"] >= min_orders].sort_values("Monetary", ascending=False).head(top_n)
    st.dataframe(
        show_cust[["CustomerID","Orders","Monetary","Units","AvgRating","Recency","Segment"]],
        use_container_width=True, hide_index=True
    )

    fig = px.scatter(
        cust, x="Orders", y="Monetary", size="Units", color="Segment",
        hover_name="CustomerID", title="Customer frequency × monetary value"
    )
    plotly_layout(fig, 470)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 — PRODUCT INTELLIGENCE
# ============================================================
with products:
    st.subheader("Product & Sales Analysis")

    selected_category = st.selectbox(
        "Drill into category", ["All"] + sorted(filtered["Category"].dropna().unique().tolist())
    )
    prod_df = filtered if selected_category == "All" else filtered[filtered["Category"] == selected_category]

    p = prod_df.groupby(["Category","Product"]).agg(
        Sales=("NetSales","sum"),
        Units=("Quantity","sum"),
        Orders=("OrderID","nunique"),
        AvgPrice=("UnitPrice","mean"),
        AvgDiscount=("Discount","mean"),
        Rating=("Rating","mean")
    ).reset_index()
    p["SalesPerOrder"] = p["Sales"] / p["Orders"].replace(0,np.nan)

    l, r = st.columns(2)
    with l:
        top = p.sort_values("Sales", ascending=False).head(top_n)
        fig = px.bar(top, x="Sales", y="Product", color="Category", orientation="h",
                     hover_data=["Units","Orders","Rating","AvgDiscount"],
                     title=f"Top {top_n} products by sales")
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        plotly_layout(fig, 450)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        fig = px.scatter(
            p, x="Units", y="Sales", size="Orders", color="Category",
            hover_name="Product", hover_data=["AvgPrice","AvgDiscount","Rating"],
            title="Product portfolio matrix: demand × revenue"
        )
        plotly_layout(fig, 450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Discount effectiveness")
    disc = filtered.groupby("DiscountBand", observed=True).agg(
        AvgSales=("NetSales","mean"),
        TotalSales=("NetSales","sum"),
        Orders=("OrderID","nunique"),
        AvgUnits=("Quantity","mean"),
        AvgRating=("Rating","mean")
    ).reset_index()
    fig = px.bar(
        disc, x="DiscountBand", y="AvgSales", text_auto=".2s",
        hover_data=["TotalSales","Orders","AvgUnits","AvgRating"],
        title="Does a deeper discount actually create larger orders?"
    )
    plotly_layout(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Category × city opportunity heatmap")
    pivot = filtered.pivot_table(
        index="Category", columns="City", values="NetSales", aggfunc="sum", fill_value=0
    )
    fig = px.imshow(pivot, aspect="auto", text_auto=".2s",
                    title="Sales concentration across category and city")
    plotly_layout(fig, 470)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4 — OPERATIONS
# ============================================================
with operations:
    st.subheader("Order & Delivery Analysis")

    delivered = filtered[filtered["OrderStatus"].str.lower().eq("delivered")].copy()
    avg_delivery = delivered["DeliveryDays"].mean()
    median_delivery = delivered["DeliveryDays"].median()
    cancellation = safe_pct(
        filtered["OrderStatus"].str.lower().str.contains("cancel", na=False).sum(), len(filtered)
    )

    a,b,c,d = st.columns(4)
    a.metric("Avg Delivery", f"{avg_delivery:.1f} days" if pd.notna(avg_delivery) else "N/A")
    b.metric("Median Delivery", f"{median_delivery:.1f} days" if pd.notna(median_delivery) else "N/A")
    c.metric("Cancellation Rate", f"{cancellation:.1f}%")
    d.metric("Discount Cost", fmt_money(filtered["DiscountValue"].sum()))

    l,r = st.columns(2)
    with l:
        city_ops = delivered.groupby("City").agg(
            AvgDeliveryDays=("DeliveryDays","mean"),
            Orders=("OrderID","nunique"),
            Rating=("Rating","mean")
        ).reset_index().sort_values("AvgDeliveryDays")
        fig = px.scatter(
            city_ops, x="AvgDeliveryDays", y="Rating", size="Orders",
            hover_name="City", title="Delivery speed vs customer rating"
        )
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        pm = filtered.groupby(["PaymentMethod","OrderStatus"]).size().reset_index(name="Orders")
        fig = px.bar(pm, x="PaymentMethod", y="Orders", color="OrderStatus",
                     barmode="stack", title="Payment method × order outcome")
        plotly_layout(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Order activity pattern")
    activity = filtered.groupby(["Weekday","Hour"]).size().reset_index(name="Orders")
    weekday_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    activity["Weekday"] = pd.Categorical(activity["Weekday"], categories=weekday_order, ordered=True)
    matrix = activity.pivot(index="Weekday", columns="Hour", values="Orders").fillna(0)
    fig = px.imshow(matrix, aspect="auto", title="When do customers place orders?",
                    labels=dict(x="Hour of day", y="Day", color="Orders"))
    plotly_layout(fig, 390)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 6 — DATA QUALITY & EXPORT
# ============================================================
with data_lab:
    st.subheader("Data Quality & Reproducibility")
    a,b,c,d = st.columns(4)
    a.metric("Raw Rows", f"{quality['original_rows']:,}")
    b.metric("Rows Flagged", f"{quality['quality_issue_rows_raw']:,}")
    c.metric("Valid Pool", f"{quality['valid_rows']:,}")
    d.metric("Fixed Sample", f"{quality['sample_rows']:,}")

    st.markdown("#### Why this matters")
    st.write(
        "The dashboard first standardizes text, converts dates/numerics, harmonizes percentage discounts, "
        "flags invalid values, removes analytically unusable rows, creates derived measures, and only then "
        f"draws the reproducible {SAMPLE_SIZE:,}-row sample using random_state={SEED}."
    )

    quality_table = pd.DataFrame({
        "Check":[
            "Invalid / non-positive quantity",
            "Invalid / non-positive unit price",
            "Invalid age",
            "Invalid discount",
            "Invalid rating",
            "Missing order date",
            "Exact duplicate raw rows"
        ],
        "Rows":[
            int(raw_df["InvalidQuantity"].sum()),
            int(raw_df["InvalidPrice"].sum()),
            int(raw_df["InvalidAge"].sum()),
            int(raw_df["InvalidDiscount"].sum()),
            int(raw_df["InvalidRating"].sum()),
            int(raw_df["MissingOrderDate"].sum()),
            quality["duplicates_raw"]
        ]
    })
    st.dataframe(quality_table, use_container_width=True, hide_index=True)

    st.markdown("#### Filtered analytical data")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download current filtered view",
        csv,
        "shopease_filtered_analysis.csv",
        "text/csv"
    )

# --------------------------- FOOTER --------------------------
st.divider()
st.caption(
    "ShopEase Analytical Dashboard • Python | Pandas | NumPy | Random | "
    "Matplotlib | Seaborn | SciPy | Statsmodels | Streamlit | Plotly"
)
