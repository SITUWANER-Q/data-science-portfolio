import io
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Macau Weather Dashboard (1999-2019)",
    layout="wide",
    initial_sidebar_state="expanded",
)
# -----------------------------
# Color scheme (cohesive theme)
# -----------------------------
PRIMARY_WARM = "#E67E22"  # orange for temperature
PRIMARY_WARM_DARK = "#D35400"
PRIMARY_COOL = "#2980B9"  # blue for rainfall
PRIMARY_COOL_TEAL = "#1ABC9C"  # teal alternative for rainfall
PRIMARY_SUN = "#F1C40F"  # yellow for sunshine
ACCENT = "#8E44AD"  # deep purple for highlights
BG_LIGHT = "#F7F7F7"
PAPER_LIGHT = "#FFFFFF"

def apply_plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        plot_bgcolor=PAPER_LIGHT,
        paper_bgcolor=PAPER_LIGHT,
        hoverlabel=dict(bgcolor="#FCFCFC"),
        margin=dict(l=30, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
# -----------------------------
# Data loading / preparation
# -----------------------------
EXPECTED_COLUMNS = [
    "date",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "sunshine_hours",
]
data: Optional[pd.DataFrame] = None
@st.cache_data(show_spinner=False)
def generate_sample_data(start_year: int = 1999, end_year: int = 2019) -> pd.DataFrame:
    rng = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    num_days = len(rng)

    # Seasonal temperature pattern with noise
    day_of_year = rng.dayofyear.values
    temp_seasonal = 22 + 8 * np.sin(2 * np.pi * (day_of_year - 30) / 365.25)
    temp_trend = 0.015 * (rng.year - start_year)  # slight warming trend
    temperature_c = temp_seasonal + temp_trend + np.random.normal(0, 1.5, num_days)

    # Humidity generally higher in warmer, wetter months
    humidity_base = 75 - 5 * np.cos(2 * np.pi * (day_of_year - 30) / 365.25)
    humidity_pct = np.clip(humidity_base + np.random.normal(0, 5, num_days), 40, 100)

    # Rainfall: monsoon-like seasonality, skewed distribution with heavy tails
    month = rng.month.values
    monsoon_factor = np.where((month >= 5) & (month <= 9), 1.8, 0.8)
    rainfall_shape = 1.3 * monsoon_factor
    rainfall_scale = 4.0 * monsoon_factor
    rainfall_mm = np.random.gamma(rainfall_shape, rainfall_scale, num_days)
    # Introduce zero-rain days
    dry_prob = np.where((month >= 11) | (month <= 3), 0.75, 0.45)
    is_dry = np.random.rand(num_days) < dry_prob
    rainfall_mm[is_dry] = 0.0

    # Sunshine hours: more in spring/autumn, less during monsoon and winter
    sun_base = 6 + 2.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
    sun_rain_penalty = np.clip(1 - (rainfall_mm / 40.0), 0.2, 1.0)
    sunshine_hours = np.clip(sun_base * sun_rain_penalty + np.random.normal(0, 0.7, num_days), 0, 12)

    df = pd.DataFrame(
        {
            "date": rng,
            "temperature_c": temperature_c,
            "rainfall_mm": rainfall_mm,
            "humidity_pct": humidity_pct,
            "sunshine_hours": sunshine_hours,
        }
    )

    return enrich_calendar_columns(df)


def enrich_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    return df


def read_uploaded_xlsx(file_bytes: bytes) -> Optional[pd.DataFrame]:
    try:
        # Try sheet named "Macau_weather_dataset" first; otherwise fallback to first sheet
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_name = "Macau_weather_dataset" if "Macau_weather_dataset" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name)
    except Exception:
        return None

    # Normalize column names
    def norm(s: str) -> str:
        s = str(s).strip().lower()
        s = s.replace("º", "°")
        s = " ".join(s.split())
        return s

    df.rename(columns={c: norm(c) for c in df.columns}, inplace=True)

    # Expected source columns (normalized) -> target
    rename_map = {
        "date": "date",
        "mean(°c)": "temperature_c",
        "mean °c": "temperature_c",
        "mean(c)": "temperature_c",
        "mean maximum (°c)": "temp_max_c",
        "mean minimum (°c)": "temp_min_c",
        "total rainfall (mm)": "rainfall_mm",
        "mean relative humidity (%)": "humidity_pct",
        "insolation duration (hour)": "sunshine_hours",
    }

    # Apply map where possible
    for src, dst in rename_map.items():
        if src in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    # If average temperature not present, try to derive from max/min
    if "temperature_c" not in df.columns and {"temp_max_c", "temp_min_c"}.issubset(df.columns):
        df["temperature_c"] = (pd.to_numeric(df["temp_max_c"], errors="coerce") + pd.to_numeric(df["temp_min_c"], errors="coerce")) / 2.0

    # Validate required columns
    required = ["date", "temperature_c", "rainfall_mm", "humidity_pct", "sunshine_hours"]
    if not all(c in df.columns for c in required):
        return None

    # Coerce types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["temperature_c", "humidity_pct", "sunshine_hours", "rainfall_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Handle non-numeric flags like "VST"
    df["rainfall_mm"] = df["rainfall_mm"].fillna(0)
    df["sunshine_hours"] = df["sunshine_hours"].fillna(0)

    # Drop rows without essential fields
    df = df.dropna(subset=["date", "temperature_c", "humidity_pct"]).copy()

    return enrich_calendar_columns(df[["date", "temperature_c", "rainfall_mm", "humidity_pct", "sunshine_hours"]])


# -----------------------------
# Sidebar: Data, filters, and notes
# -----------------------------
with st.sidebar:
    st.markdown("**Data**")
    uploaded = st.file_uploader(
        "Upload Excel (.xlsx/.xls) containing sheet 'Macau_weather_dataset' (optional)",
        type=["xlsx", "xls"],
        help=(
            "Data is daily granularity. We'll derive year/month and aggregate as needed. "
            "If no file is uploaded, sample data (1999–2019) will be used."
        ),
    )

    if uploaded is not None:
        parsed = read_uploaded_xlsx(uploaded.read())
        if parsed is None or parsed.empty:
            st.warning("Could not parse the uploaded Excel file. Using sample data instead.")
            data = generate_sample_data()
        else:
            data = parsed
            st.success("Using uploaded dataset.")
    else:
        st.info("No file uploaded. Using sample data.")
        data = generate_sample_data()

    st.markdown("---")
    st.markdown("**Filters**")
    all_years = sorted(data["year"].unique().tolist())
    selected_years = st.multiselect("Year(s)", options=all_years, default=all_years)

    month_name_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    selected_months = st.multiselect(
        "Month(s)", options=month_name_order, default=month_name_order
    )

    st.markdown("---")
    st.markdown("**Instructions**")
    st.caption(
        "Hover over points for tooltips. Use filters above to focus by year/month. "
        "Charts update dynamically; use them to explore trends, seasonality, and distributions."
    )


# Filter data as per selections (drill-down via filters)
filtered = data[(data["year"].isin(selected_years)) & (data["month_name"].isin(selected_months))]


# -----------------------------
# Title
# -----------------------------
st.title("Macau Weather Dashboard")
st.caption(
    "Exploring temperature, rainfall, humidity, and sunshine patterns in Macau (1999-2019)"
)


# -----------------------------
# Helper aggregations
# -----------------------------
annual = (
    data.groupby("year", as_index=False)
    .agg(avg_temperature_c=("temperature_c", "mean"), total_rainfall_mm=("rainfall_mm", "sum"))
)

annual_filtered = (
    filtered.groupby("year", as_index=False)
    .agg(avg_temperature_c=("temperature_c", "mean"), total_rainfall_mm=("rainfall_mm", "sum"))
)

monthly_by_year = (
    data.groupby(["year", "month", "month_name"], as_index=False)
    .agg(
        avg_temperature_c=("temperature_c", "mean"),
        total_rainfall_mm=("rainfall_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
        avg_sunshine_hours=("sunshine_hours", "mean"),
    )
)

monthly_by_year_filtered = (
    filtered.groupby(["year", "month", "month_name"], as_index=False)
    .agg(
        avg_temperature_c=("temperature_c", "mean"),
        total_rainfall_mm=("rainfall_mm", "sum"),
        avg_humidity_pct=("humidity_pct", "mean"),
        avg_sunshine_hours=("sunshine_hours", "mean"),
    )
)


# -----------------------------
# Layout: 3 columns (Left/Center/Right)
# -----------------------------
col_left, col_center, col_right = st.columns(3)


# -----------------------------
# Section 1: Annual Trends (Left)
# -----------------------------
with col_left:
    st.subheader("Section 1: Annual Trends")

    # Annual Temperature Variation
    fig_temp_line = px.line(
        annual,
        x="year",
        y="avg_temperature_c",
        title="Annual Temperature Variation",
        markers=True,
        color_discrete_sequence=[PRIMARY_WARM],
        labels={"avg_temperature_c": "Avg Temp (°C)", "year": "Year"},
    )
    # Add trendline via lowess using plotly express (statsmodels optional) -> fallback to simple rolling mean
    try:
        fig_temp_trend = px.scatter(
            annual, x="year", y="avg_temperature_c", trendline="lowess", trendline_color_override=PRIMARY_WARM_DARK
        )
        # Extract the trendline trace
        for tr in fig_temp_trend.data:
            if tr.mode == "lines":
                fig_temp_line.add_trace(tr)
    except Exception:
        # Simple moving average fallback
        temp_ma = annual["avg_temperature_c"].rolling(window=5, min_periods=1, center=True).mean()
        fig_temp_line.add_trace(
            go.Scatter(x=annual["year"], y=temp_ma, mode="lines", name="Trend", line=dict(color=PRIMARY_WARM_DARK, width=3))
        )
    fig_temp_line = apply_plotly_theme(fig_temp_line)
    st.plotly_chart(fig_temp_line, use_container_width=True)

    # Annual Rainfall Variation
    fig_rain_line = px.line(
        annual,
        x="year",
        y="total_rainfall_mm",
        title="Annual Rainfall Variation",
        markers=True,
        color_discrete_sequence=[PRIMARY_COOL],
        labels={"total_rainfall_mm": "Total Rainfall (mm)", "year": "Year"},
    )
    fig_rain_line = apply_plotly_theme(fig_rain_line)
    st.plotly_chart(fig_rain_line, use_container_width=True)

    # Temperature vs. Rainfall Scatter
    fig_scatter = px.scatter(
        annual,
        x="avg_temperature_c",
        y="total_rainfall_mm",
        color="year",
        color_continuous_scale=px.colors.sequential.Viridis_r,
        title="Temperature vs. Rainfall (Annual)",
        labels={
            "avg_temperature_c": "Avg Temp (°C)",
            "total_rainfall_mm": "Total Rainfall (mm)",
            "year": "Year",
        },
        hover_data={"year": True},
    )
    fig_scatter = apply_plotly_theme(fig_scatter)
    st.plotly_chart(fig_scatter, use_container_width=True)


# -----------------------------
# Section 2: Seasonal Patterns (Center)
# -----------------------------
with col_center:
    st.subheader("Section 2: Seasonal Patterns")

    # Box Plots: Monthly Temperature Distribution (use filtered daily data)
    month_category_order = month_name_order
    daily_for_box = filtered.copy()
    daily_for_box["month_name"] = pd.Categorical(daily_for_box["month_name"], categories=month_category_order, ordered=True)
    # Sequential color list from cool to warm across months
    month_colors = [
        "#2E86C1",  # Jan - cool
        "#3498DB",
        "#1ABC9C",
        "#16A085",
        "#27AE60",
        "#2ECC71",
        "#F1C40F",
        "#F39C12",
        "#E67E22",
        "#D35400",
        "#E74C3C",
        "#C0392B",  # Dec - warm
    ]
    fig_box = px.box(
        daily_for_box,
        x="month_name",
        y="temperature_c",
        color="month_name",
        color_discrete_sequence=month_colors,
        category_orders={"month_name": month_category_order},
        title="Monthly Temperature Distribution",
        labels={"month_name": "Month", "temperature_c": "Daily Temp (°C)"},
        points=False,
    )
    fig_box.update_layout(showlegend=False)
    fig_box = apply_plotly_theme(fig_box)
    st.plotly_chart(fig_box, use_container_width=True)

    # Stacked Area: Monthly Rainfall Contribution per Year (filtered)
    # Build per-year monthly totals and normalize by annual total to show contribution
    monthly_totals = (
        monthly_by_year_filtered.groupby(["year", "month", "month_name"], as_index=False)["total_rainfall_mm"].sum()
    )
    annual_totals = monthly_totals.groupby("year", as_index=False)["total_rainfall_mm"].sum().rename(
        columns={"total_rainfall_mm": "annual_total_mm"}
    )
    monthly_totals = monthly_totals.merge(annual_totals, on="year", how="left")
    monthly_totals["contribution"] = np.where(
        monthly_totals["annual_total_mm"] > 0,
        monthly_totals["total_rainfall_mm"] / monthly_totals["annual_total_mm"],
        0.0,
    )

    monthly_totals["month_name"] = pd.Categorical(
        monthly_totals["month_name"], categories=month_category_order, ordered=True
    )
    fig_area = px.area(
        monthly_totals.sort_values(["year", "month"]),
        x="year",
        y="contribution",
        color="month_name",
        color_discrete_sequence=[
            "#D6EAF8",
            "#AED6F1",
            "#85C1E9",
            "#5DADE2",
            "#48C9B0",
            "#45B39D",
            "#2ECC71",
            "#28B463",
            "#1ABC9C",
            "#17A589",
            "#148F77",
            "#117864",
        ],
        title="Monthly Rainfall Contribution to Annual Total",
        labels={"contribution": "Contribution (fraction)", "year": "Year", "month_name": "Month"},
        groupnorm="percent",
    )
    fig_area.update_yaxes(tickformat=",.0%")
    fig_area = apply_plotly_theme(fig_area)
    st.plotly_chart(fig_area, use_container_width=True)

    # Heatmap: Temperature and Humidity Correlation by Month (filtered)
    corr_by_month = []
    for name in month_name_order:
        sub = filtered[filtered["month_name"] == name]
        if len(sub) >= 5 and sub["temperature_c"].std() > 1e-6 and sub["humidity_pct"].std() > 1e-6:
            corr = float(np.corrcoef(sub["temperature_c"], sub["humidity_pct"])[0, 1])
        else:
            corr = np.nan
        corr_by_month.append(corr)

    z = np.array([corr_by_month])
    fig_heat = px.imshow(
        z,
        x=month_name_order,
        y=["Correlation"],
        color_continuous_scale=px.colors.diverging.RdBu,
        origin="lower",
        zmin=-1,
        zmax=1,
        aspect="auto",
        labels=dict(color="Pearson r"),
        title="Temperature vs. Humidity Correlation by Month",
    )
    fig_heat.update_yaxes(showticklabels=False)
    fig_heat = apply_plotly_theme(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------
# Section 3: Sunshine & Rainfall Details (Right)
# -----------------------------
with col_right:
    st.subheader("Section 3: Sunshine Duration & Detailed Rainfall")

    # Bar Chart: Average Monthly Sunshine Duration (filtered)
    sunshine_monthly = (
        monthly_by_year_filtered.groupby(["month", "month_name"], as_index=False)["avg_sunshine_hours"].mean()
    )
    sunshine_monthly["month_name"] = pd.Categorical(
        sunshine_monthly["month_name"], categories=month_name_order, ordered=True
    )
    fig_sun_bar = px.bar(
        sunshine_monthly.sort_values("month"),
        x="month_name",
        y="avg_sunshine_hours",
        title="Average Monthly Sunshine Duration",
        color_discrete_sequence=[PRIMARY_SUN],
        labels={"month_name": "Month", "avg_sunshine_hours": "Sunshine (hours/day)"},
    )
    fig_sun_bar = apply_plotly_theme(fig_sun_bar)
    st.plotly_chart(fig_sun_bar, use_container_width=True)

    # Histogram: Daily Rainfall Distribution (filtered)
    fig_rain_hist = px.histogram(
        filtered,
        x="rainfall_mm",
        nbins=50,
        title="Daily Rainfall Distribution",
        color_discrete_sequence=[PRIMARY_COOL],
        labels={"rainfall_mm": "Daily Rainfall (mm)"},
    )
    fig_rain_hist.update_xaxes(range=[0, max(1.0, float(filtered["rainfall_mm"].max()))])
    fig_rain_hist = apply_plotly_theme(fig_rain_hist)
    st.plotly_chart(fig_rain_hist, use_container_width=True)

    # Optional Map: Placeholder unless geospatial data provided
    st.caption(
        "SituWaner MC566111"
    )
    if {"lat", "lon"}.issubset(filtered.columns):
        # Aggregate rainfall by location (example: mean rainfall)
        loc = filtered.groupby(["lat", "lon"], as_index=False)["rainfall_mm"].mean()
        fig_map = px.scatter_mapbox(
            loc,
            lat="lat",
            lon="lon",
            size="rainfall_mm",
            color="rainfall_mm",
            color_continuous_scale=px.colors.sequential.Blues,
            zoom=11,
            height=400,
            title="Average Rainfall by Location",
        )
        fig_map.update_layout(mapbox_style="carto-positron")
        fig_map = apply_plotly_theme(fig_map)
        st.plotly_chart(fig_map, use_container_width=True)


# -----------------------------
# Footer (source and date)
# -----------------------------
st.markdown("---")
st.markdown(
    f"Data Source: Macao Meteorological and Geophysical Bureau. | Date of Analysis: {datetime.now().strftime('%Y-%m-%d')}"
)

