# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components

# Try import Prophet; we'll fallback if it's not available
USE_PROPHET = True
try:
   from prophet import Prophet
except Exception:
    USE_PROPHET = False

st.set_page_config(page_title="Migration & Seasonal Photo Predictor", layout="wide")
st.title("📸 Migration & Seasonal Photo Predictor — Streamlit (Prophet-enabled)")

# ---------- Helpers ----------
def fetch_ebird_geo(species_code: str, lat: float, lon: float, dist_km: int, back_days: int, api_key: str):
    """
    Use eBird geo recent endpoint: returns DataFrame with columns date, lat, lon, count, locName
    Note: eBird rate limits apply. Use caching and don't call too often.
    """
    url = "https://api.ebird.org/v2/data/obs/geo/recent/" + species_code
    headers = {"X-eBirdApiToken": api_key} if api_key else {}
    params = {"lat": lat, "lng": lon, "dist": dist_km, "back": back_days}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"eBird API error {r.status_code}: {r.text}")
    data = r.json()
    if not data:
        return pd.DataFrame(columns=["date","lat","lon","count","locName"])
    rows = []
    for obs in data:
        obs_date = obs.get("obsDt","").split(" ")[0]
        rows.append({
            "date": obs_date,
            "lat": obs.get("lat"),
            "lon": obs.get("lng"),
            "count": obs.get("howMany") if obs.get("howMany") is not None else 1,
            "locName": obs.get("locName","")
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=3600)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize names
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    # Identify columns
    date_col = next((c for c in df.columns if "date" in c), None)
    lat_col = next((c for c in df.columns if c in ("lat","latitude","latitude_deg")), None)
    lon_col = next((c for c in df.columns if c in ("lon","lng","longitude","longitude_deg")), None)
    count_col = next((c for c in df.columns if "count" in c or "howmany" in c), None)
    if not date_col or not lat_col or not lon_col:
        raise ValueError("CSV must contain date, latitude and longitude columns (names can include date, lat/latitude, lon/lng).")
    df = df.rename(columns={date_col:"date", lat_col:"lat", lon_col:"lon"})
    if count_col:
        df = df.rename(columns={count_col:"count"})
    else:
        df["count"] = 1
    df["date"] = pd.to_datetime(df["date"])
    return df[["date","lat","lon","count"]]

def weekly_aggregate(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame()
    df2 = df.copy()
    df2["week_start"] = df2["date"].dt.to_period("W").apply(lambda r: r.start_time)
    weekly = df2.groupby("week_start", as_index=False)["count"].sum().rename(columns={"count":"y"})
    weekly = weekly.sort_values("week_start")
    return weekly

def seasonal_average_by_week(df: pd.DataFrame):
    """Return mean & std counts per calendar week across years"""
    if df.empty:
        return pd.DataFrame(columns=["week","mean_count","std_count","n_years"])
    tmp = df.copy()
    tmp["year"] = tmp["date"].dt.year
    tmp["weeknum"] = tmp["date"].dt.isocalendar().week
    agg = tmp.groupby(["year","weeknum"], as_index=False)["count"].sum()
    seasonal = agg.groupby("weeknum")["count"].agg(["mean","std","count"]).rename(columns={"mean":"mean_count","std":"std_count","count":"n_records"})
    seasonal = seasonal.reset_index().rename(columns={"weeknum":"week"})
    # count of years per week:
    years_per_week = agg.groupby("weeknum")["year"].nunique().reset_index().rename(columns={"weeknum":"week","year":"n_years"})
    seasonal = seasonal.merge(years_per_week, on="week", how="left").fillna(0)
    return seasonal[["week","mean_count","std_count","n_years"]]

@st.cache_data(ttl=3600)
def prophet_forecast(weekly_df: pd.DataFrame, periods_weeks: int = 52):
    """
    weekly_df: columns ['week_start','y']
    returns: forecast dataframe with columns ds,yhat,yhat_lower,yhat_upper
    """
    if weekly_df.empty:
        return pd.DataFrame()
    # Prepare prophet input
    dfp = weekly_df.rename(columns={"week_start":"ds"})
    dfp = dfp[["ds","y"]].copy()
    # Prophet wants ds as datetime and y numeric
    try:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(dfp)
        future = model.make_future_dataframe(periods=periods_weeks, freq="W")
        forecast = model.predict(future)
        return forecast[["ds","yhat","yhat_lower","yhat_upper"]]
    except Exception as e:
        st.warning(f"Prophet failed: {e}")
        return pd.DataFrame()

def predict_using_seasonal(seasonal_df: pd.DataFrame, weeks_ahead:int=52):
    today = date.today()
    max_mean = seasonal_df["mean_count"].max() if not seasonal_df.empty else 0.0
    rows = []
    for i in range(weeks_ahead):
        dt = today + timedelta(weeks=i)
        wk = int(dt.isocalendar().week)
        monday = dt - timedelta(days=dt.weekday())
        mean_count = float(seasonal_df.loc[seasonal_df["week"]==wk,"mean_count"].squeeze()) if wk in seasonal_df["week"].values else 0.0
        std_count = float(seasonal_df.loc[seasonal_df["week"]==wk,"std_count"].squeeze()) if wk in seasonal_df["week"].values else 0.0
        score = mean_count / max_mean if max_mean>0 else 0.0
        rows.append({"week_start":monday,"week":wk,"mean_count":mean_count,"std_count":std_count,"score":score})
    return pd.DataFrame(rows)

def make_heatmap_html(df: pd.DataFrame, zoom_start=6):
    if df.empty:
        return "<p>No data</p>"
    avg_lat = float(df["lat"].mean())
    avg_lon = float(df["lon"].mean())
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=zoom_start, tiles="CartoDB positron")
    heat_data = df[["lat","lon","count"]].values.tolist()
    HeatMap(heat_data, radius=8, blur=12, max_zoom=8).add_to(m)
    return m.get_root().render()

# ---------- UI ----------
st.sidebar.header("Input / Config")
mode = st.sidebar.radio("Mode", ["eBird geo fetch (recommended)", "Upload CSV"])
st.sidebar.markdown("---")
st.sidebar.info("Using eBird API key: uitbv83kll1n for live fetching.")
st.sidebar.markdown("---")
if mode == "eBird geo fetch (recommended)":
    st.subheader("Fetch sightings from eBird (geo search)")
    species_code = st.text_input("Species code (eBird species code)", value="comsan")
    lat = st.number_input("Latitude", value=22.57, format="%.5f")
    lon = st.number_input("Longitude", value=88.36, format="%.5f")
    dist_km = st.slider("Radius (km)", 5, 200, 50, step=5)
    back_days = st.slider("Days back to fetch", 30, 365, 365, step=30)
    fetch_btn = st.button("Fetch & Forecast")
    if fetch_btn:
        api_key = "uitbv83kll1n"
        try:
            df = fetch_ebird_geo(species_code.strip(), lat, lon, dist_km, back_days, api_key)
        except Exception as e:
            st.error(f"Failed to fetch eBird: {e}")
            df = pd.DataFrame(columns=["date","lat","lon","count"])
        if df.empty:
            st.warning("No sightings returned. Try larger radius or upload CSV as fallback.")
        else:
            st.success(f"Fetched {len(df)} records.")
            st.dataframe(df.head(8))
            # aggregate and seasonal
            weekly = weekly_aggregate(df)
            seasonal = seasonal_average_by_week(df)
            # Prophet forecast attempt
            forecast = None
            if USE_PROPHET:
                with st.spinner("Training Prophet model (may take a few seconds)..."):
                    forecast = prophet_forecast(weekly, periods_weeks=52)
            if forecast is not None and not forecast.empty:
                # present top weeks from prophet (future-only)
                future_only = forecast[forecast["ds"] >= pd.Timestamp(date.today())].copy()
                future_only["weeknum"] = future_only["ds"].dt.isocalendar().week
                top = future_only.sort_values("yhat", ascending=False).head(6)
                top["start"] = top["ds"].dt.strftime("%Y-%m-%d")
                top["end"] = (top["ds"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
                st.subheader("Top predicted weeks (Prophet forecast)")
                st.table(top[["start","end","yhat","yhat_lower","yhat_upper"]].rename(columns={
                    "yhat":"predicted_count","yhat_lower":"low","yhat_upper":"high"
                }))
            else:
                st.info("Prophet unavailable or failed; falling back to seasonal average.")
                preds = predict_using_seasonal(seasonal, weeks_ahead=52)
                top = preds.sort_values("mean_count", ascending=False).head(6)
                top["start"] = top["week_start"].dt.strftime("%Y-%m-%d")
                top["end"] = (top["week_start"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
                st.subheader("Top weeks (Seasonal average fallback)")
                st.table(top[["start","end","mean_count","score"]].rename(columns={"mean_count":"avg_count"}))
            # plot seasonal profile
            st.subheader("Seasonal profile (mean sightings per calendar week)")
            seasonal_plot = seasonal.set_index("week")["mean_count"]
            st.line_chart(seasonal_plot)
            # heatmap
            st.subheader("Sightings heatmap")
            html_map = make_heatmap_html(df, zoom_start=7)
            components.html(html_map, height=600)
            # downloads
            st.download_button("Download sightings CSV", data=df.to_csv(index=False).encode("utf-8"), file_name=f"{species_code}_sightings.csv")
            st.download_button("Download heatmap HTML", data=html_map.encode("utf-8"), file_name=f"{species_code}_heatmap.html")

else:
    st.subheader("Upload a CSV of sightings (must include date, lat, lon; count optional)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    process_btn = st.button("Process & Forecast")
    if process_btn:
        if uploaded is None:
            st.error("Please upload a CSV file.")
        else:
            try:
                df = load_csv(uploaded)
                st.success(f"Loaded {len(df)} records.")
                st.dataframe(df.head(8))
                weekly = weekly_aggregate(df)
                seasonal = seasonal_average_by_week(df)
                forecast = None
                if USE_PROPHET:
                    with st.spinner("Training Prophet model (may take a few seconds)..."):
                        forecast = prophet_forecast(weekly, periods_weeks=52)
                if forecast is not None and not forecast.empty:
                    future_only = forecast[forecast["ds"] >= pd.Timestamp(date.today())].copy()
                    future_only["start"] = future_only["ds"].dt.strftime("%Y-%m-%d")
                    future_only["end"] = (future_only["ds"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
                    top = future_only.sort_values("yhat", ascending=False).head(6)
                    st.subheader("Top predicted weeks (Prophet forecast)")
                    st.table(top[["start","end","yhat","yhat_lower","yhat_upper"]].rename(columns={
                        "yhat":"predicted_count","yhat_lower":"low","yhat_upper":"high"
                    }))
                else:
                    preds = predict_using_seasonal(seasonal, weeks_ahead=52)
                    top = preds.sort_values("mean_count", ascending=False).head(6)
                    top["start"] = top["week_start"].dt.strftime("%Y-%m-%d")
                    top["end"] = (top["week_start"] + pd.Timedelta(days=6)).dt.strftime("%Y-%m-%d")
                    st.subheader("Top weeks (Seasonal average fallback)")
                    st.table(top[["start","end","mean_count","score"]].rename(columns={"mean_count":"avg_count"}))
                # seasonal plot + heatmap
                st.subheader("Seasonal profile")
                st.line_chart(seasonal.set_index("week")["mean_count"])
                st.subheader("Sightings heatmap")
                html_map = make_heatmap_html(df, zoom_start=7)
                components.html(html_map, height=600)
                st.download_button("Download cleaned CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="cleaned_sightings.csv")
                st.download_button("Download heatmap HTML", data=html_map.encode("utf-8"), file_name="heatmap.html")
            except Exception as e:
                st.error(f"Failed to process uploaded CSV: {e}")

st.markdown("---")
st.markdown("**Notes:** Prophet gives improved forecasts (trend + seasonality). If Prophet isn't available on the host, the app falls back to simple seasonal averages. The eBird API key is set directly in the code for reliable fetching.")