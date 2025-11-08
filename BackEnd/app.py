import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit

try:
    from dotenv import load_dotenv
except ImportError:  # Fallback to a minimal loader if python-dotenv is missing.
    def load_dotenv(path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "dwell_times_complete_1.csv")
ENV_FILE = os.path.join(SCRIPT_DIR, ".env")


@st.cache_resource(show_spinner="Loading data and training XGBoost model...")
def load_model_bundle():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    df = df[(df["BDT"] > 0) & (df["BDT"] <= 700)].copy()
    need_cols = ["trip_id", "lat", "long", "ts_time", "occupancy", "congestion", "BDT"]
    df = df.dropna(subset=[c for c in need_cols if c in df.columns])

    ts = pd.to_datetime(df["ts_time"], format="%H:%M:%S", errors="coerce")
    df = df.loc[ts.notna()].copy()
    df["ts_dt"] = ts[ts.notna()]
    hour_series = df["ts_dt"].dt.hour
    hour_min = int(hour_series.min())
    hour_max = int(hour_series.max())

    seconds = (
        df["ts_dt"].dt.hour * 3600 + df["ts_dt"].dt.minute * 60 + df["ts_dt"].dt.second
    )
    df["hour_float"] = seconds / 3600.0
    df["occupancy"] = df["occupancy"].astype(int)
    df["congestion"] = df["congestion"].astype(int)

    df = df.sort_values(["trip_id", "ts_dt"]).copy()
    df["stop_seq"] = df.groupby("trip_id").cumcount().astype(np.int32)
    df["prev_bdt"] = df.groupby("trip_id")["BDT"].shift(1)
    df["rolling_med_bdt"] = (
        df.groupby("trip_id")["BDT"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    )
    df["prev_ts"] = df.groupby("trip_id")["ts_dt"].shift(1)
    df["intra_trip_gap_sec"] = (df["ts_dt"] - df["prev_ts"]).dt.total_seconds()

    median_bdt = df["BDT"].median()
    gap_median = df["intra_trip_gap_sec"].median()
    df["prev_bdt"] = df["prev_bdt"].fillna(median_bdt)
    df["rolling_med_bdt"] = df["rolling_med_bdt"].fillna(median_bdt)
    df["intra_trip_gap_sec"] = df["intra_trip_gap_sec"].fillna(gap_median)

    coords_rad = np.radians(df[["lat", "long"]].to_numpy())
    eps_rad = 1.0 / 6371000.0
    db = DBSCAN(eps=eps_rad, min_samples=5, metric="haversine")
    df["stop_id"] = db.fit_predict(coords_rad)

    if "Suburb" in df.columns:
        df["suburb_cluster"] = df["Suburb"].astype(str) + "_" + df["stop_id"].astype(str)
    else:
        df["suburb_cluster"] = df["stop_id"].astype(str)
    df["suburb_cluster_id"] = pd.factorize(df["suburb_cluster"])[0]

    groups = df["trip_id"].values
    targets = df["BDT"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(splitter.split(df, targets, groups=groups))
    train_subset = df.iloc[train_idx].copy()

    global_mean = train_subset["BDT"].mean()
    stop_mean = train_subset.groupby("suburb_cluster_id")["BDT"].mean()
    df["stop_mean_bdt"] = df["suburb_cluster_id"].map(stop_mean).fillna(global_mean)

    numeric_clean = [
        "Estimated resident population (no.)",
        "Population density (persons/km2)",
        "Median age - persons (years)",
        "Median_tot_hhd_inc_weekly",
        "rainfall_mm",
    ]
    for col in numeric_clean:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=True)
                .str.replace(" ", "", regex=True)
                .replace("-", np.nan)
                .astype(float)
            )

    df["rain_flag"] = 0
    if "rainfall_mm" in df.columns:
        df["rain_flag"] = (df["rainfall_mm"] > 0).astype(int)

    base_feats = [
        "hour_float",
        "occupancy",
        "congestion",
        "stop_seq",
        "prev_bdt",
        "rolling_med_bdt",
        "intra_trip_gap_sec",
        "lat",
        "long",
        "suburb_cluster_id",
        "stop_mean_bdt",
        "rain_flag",
    ]
    exclude = set(base_feats + ["BDT", "trip_id", "ts_time", "ts_dt", "prev_ts", "Suburb"])
    extra_cols = [c for c in df.columns if c not in exclude]
    extra_numeric = [c for c in extra_cols if df[c].dtype.kind in "biufc"]
    feat_cols = base_feats + extra_numeric

    train_df = df.iloc[train_idx].copy()
    missing_cols = [c for c in feat_cols if c not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Feature columns missing from training frame: {missing_cols}")
    train_X = train_df[feat_cols]
    train_y = np.log1p(train_df["BDT"].to_numpy())

    xgb_params = dict(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="mae",
    )
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(train_X, train_y)

    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    importance = pd.Series(gain).sort_values(ascending=False)

    full_feature_frame = df[feat_cols].copy()

    return {
        "model": model,
        "features": feat_cols,
        "importance": importance,
        "dataframe": df.reset_index(drop=True),
        "feature_frame": full_feature_frame.reset_index(drop=True),
        "hour_range": (hour_min, hour_max),
    }


@st.cache_data(show_spinner=False)
def cached_predictions(params: Tuple[int, int, float, float]):
    bundle = load_model_bundle()
    feature_frame = bundle["feature_frame"].copy()

    occupancy, congestion, hour_float, rainfall = params
    if "occupancy" in feature_frame.columns:
        feature_frame["occupancy"] = occupancy
    if "congestion" in feature_frame.columns:
        feature_frame["congestion"] = congestion
    if "hour_float" in feature_frame.columns:
        feature_frame["hour_float"] = hour_float
    if "rainfall_mm" in feature_frame.columns:
        feature_frame["rainfall_mm"] = rainfall
    if "rain_flag" in feature_frame.columns:
        feature_frame["rain_flag"] = int(rainfall > 0)

    preds = bundle["model"].predict(feature_frame)
    dwell = np.expm1(preds)

    base = bundle["dataframe"][["lat", "long", "Suburb"]].copy()
    base["predicted_bdt"] = dwell

    summary = {
        "mean": float(np.mean(dwell)),
        "max": float(np.max(dwell)),
        "min": float(np.min(dwell)),
    }
    # Normalize dwell times and amplify contrast to produce a more vivid heatmap.
    span = dwell.max() - dwell.min()
    if span > 0:
        normalized = (dwell - dwell.min()) / span
    else:
        normalized = np.zeros_like(dwell)
    heat_weights = np.power(normalized, 5) * 500.0  # emphasize high dwell hotspots
    base["weight"] = heat_weights
    return base, summary


def render_google_heatmap(data: pd.DataFrame, api_key: str, *, height: int = 360):
    center_lat = float(data["lat"].mean())
    center_lng = float(data["long"].mean())
    heatmap_data = [
        {"lat": float(row.lat), "lng": float(row.long), "weight": float(row.weight)}
        for row in data.itertuples()
    ]
    payload = json.dumps(heatmap_data)

    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body, #map {{
        height: 100%;
        margin: 0;
        padding: 0;
      }}
    </style>
    <script>
      const heatmapPoints = {payload};
      function initMap() {{
        const center = {{ lat: {center_lat}, lng: {center_lng} }};
        const map = new google.maps.Map(document.getElementById("map"), {{
          zoom: 11,
          center: center,
          mapTypeId: "roadmap"
        }});
        const heatmapData = heatmapPoints.map((p) => {{
          return {{ location: new google.maps.LatLng(p.lat, p.lng), weight: p.weight }};
        }});
        const heatmap = new google.maps.visualization.HeatmapLayer({{
          data: heatmapData,
          radius: 30
        }});
        heatmap.setMap(map);
      }}
    </script>
    <script async defer src="https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=visualization&callback=initMap"></script>
  </head>
  <body>
    <div id="map"></div>
  </body>
</html>
"""
    st.components.v1.html(html, height=height)


def main():
    load_dotenv(ENV_FILE)
    api_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        st.warning("Google Maps API key is missing. Set GOOGLE_MAPS_API_KEY in .env.")

    st.set_page_config(page_title="Bus Dwell Time Explorer", layout="wide")
    st.title("Bus Dwell Time Explorer")
    st.caption("Interactively explore predicted dwell durations powered by XGBoost.")

    bundle = load_model_bundle()
    importance = bundle["importance"].head(5)
    hour_min, hour_max = bundle["hour_range"]
    st.session_state.setdefault("loading_status", "idle")

    left, right = st.columns([3, 1], gap="large")
    with right:
        st.subheader("Feature Influence")
        imp_df = (
            importance.rename("gain")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        st.dataframe(imp_df, use_container_width=True, height=160)

        st.markdown("---")
        st.subheader("Inspector")

        default_hour = int(np.clip(8, hour_min, hour_max))
        default_params: Dict[str, int] = {
            "occupancy": 1,
            "congestion": 0,
            "hour": default_hour,
            "rainfall": 0,
        }
        for key, val in default_params.items():
            st.session_state.setdefault(key, val)

        with st.form("analyze_form", clear_on_submit=False):
            occ = st.slider(
                "Occupancy level",
                min_value=0,
                max_value=5,
                value=int(st.session_state["occupancy"]),
                step=1,
                key="occupancy_input",
            )
            cong = st.slider(
                "Congestion flag",
                min_value=0,
                max_value=1,
                value=int(st.session_state["congestion"]),
                step=1,
                key="congestion_input",
            )
            hour = st.slider(
                "Hour of day",
                min_value=hour_min,
                max_value=hour_max,
                value=int(np.clip(st.session_state["hour"], hour_min, hour_max)),
                step=1,
                key="hour_input",
            )
            rain = st.slider(
                "Rainfall (mm)",
                min_value=0,
                max_value=10,
                value=int(st.session_state["rainfall"]),
                step=1,
                key="rain_input",
            )
            loading_placeholder = st.empty()
            submitted = st.form_submit_button("Analyze", type="primary")

        params_tuple = (occ, cong, float(hour), float(rain))
        if submitted:
            st.session_state["loading_status"] = "loading"

        status = st.session_state.get("loading_status", "idle")
        if status == "loading":
            loading_placeholder.warning("Analyzing... please wait.")
        elif status == "done":
            loading_placeholder.success("Analysis complete.")
        else:
            loading_placeholder.empty()

        if submitted or "latest_prediction" not in st.session_state:
            st.session_state["occupancy"] = occ
            st.session_state["congestion"] = cong
            st.session_state["hour"] = hour
            st.session_state["rainfall"] = rain
            if submitted:
                with st.spinner("Running XGBoost inference..."):
                    data, summary = cached_predictions(params_tuple)
            else:
                data, summary = cached_predictions(params_tuple)
            if "latest_prediction" in st.session_state:
                st.session_state["previous_prediction"] = st.session_state[
                    "latest_prediction"
                ]
            st.session_state["latest_prediction"] = {
                "data": data,
                "summary": summary,
                "params": params_tuple,
            }
            st.session_state["loading_status"] = "done"
            loading_placeholder.success("Analysis complete.")

        latest_prediction = st.session_state["latest_prediction"]
        summary = latest_prediction["summary"]
        params_tuple = latest_prediction["params"]

        st.metric("Predicted mean dwell (s)", f"{summary['mean']:.1f}")
        st.metric("Predicted max dwell (s)", f"{summary['max']:.1f}")
        st.metric("Predicted min dwell (s)", f"{summary['min']:.1f}")

        st.write("Last analyzed parameters:")
        st.json(
            {
                "occupancy": st.session_state["occupancy"],
                "congestion": st.session_state["congestion"],
                "hour": st.session_state["hour"],
                "rainfall_mm": st.session_state["rainfall"],
            },
            expanded=False,
        )

    with left:
        st.subheader("Predicted dwell time heatmaps")
        if "latest_prediction" in st.session_state:
            if api_key:
                current_hour = int(st.session_state["latest_prediction"]["params"][2])
                occ_val = int(st.session_state["latest_prediction"]["params"][0])
                cong_val = int(st.session_state["latest_prediction"]["params"][1])
                rain_val = int(st.session_state["latest_prediction"]["params"][3])
                st.markdown(
                    f"**Current parameters — {current_hour:02d}:00 | "
                    f"occupancy={occ_val}, congestion={cong_val}, rainfall={rain_val}mm**"
                )
                render_google_heatmap(
                    st.session_state["latest_prediction"]["data"], api_key
                )

                if "previous_prediction" in st.session_state:
                    prev_hour = int(
                        st.session_state["previous_prediction"]["params"][2]
                    )
                    prev_occ = int(
                        st.session_state["previous_prediction"]["params"][0]
                    )
                    prev_cong = int(
                        st.session_state["previous_prediction"]["params"][1]
                    )
                    prev_rain = int(
                        st.session_state["previous_prediction"]["params"][3]
                    )
                    st.markdown(
                        f"**Previous parameters — {prev_hour:02d}:00 | "
                        f"occupancy={prev_occ}, congestion={prev_cong}, rainfall={prev_rain}mm**"
                    )
                    render_google_heatmap(
                        st.session_state["previous_prediction"]["data"], api_key
                    )
                else:
                    st.info("이전 분석이 없습니다. 파라미터를 조정하고 Analyze를 눌러 비교해 보세요.")
            else:
                st.error("Google Maps API 키가 없어서 히트맵을 표시할 수 없습니다.")
        else:
            st.info("파라미터를 조정한 뒤 Analyze 버튼을 눌러 히트맵을 생성하세요.")


if __name__ == "__main__":
    main()
