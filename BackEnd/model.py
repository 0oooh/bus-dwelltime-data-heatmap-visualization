import pandas as pd, numpy as np
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt


# 로컬 파일 경로 설정
import os

# 스크립트 파일과 같은 디렉토리에 있는 데이터 파일 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, "dwell_times_complete_1.csv")

# 파일이 존재하는지 확인
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {DATA_FILE}")

# === 0) LOAD & BASIC CLEAN ===
df = pd.read_csv(DATA_FILE)
df = df[(df["BDT"] > 0) & (df["BDT"] <= 700)].copy()
need = ["trip_id","lat","long","ts_time","occupancy","congestion","BDT"]
df = df.dropna(subset=[c for c in need if c in df.columns])

# parse time
ts = pd.to_datetime(df["ts_time"], format="%H:%M:%S", errors="coerce")
df = df.loc[ts.notna()].copy()
df["ts_dt"] = ts[ts.notna()]

# === 1) TIME FEATURES ===
sec = df["ts_dt"].dt.hour*3600 + df["ts_dt"].dt.minute*60 + df["ts_dt"].dt.second
df["hour_float"] = sec/3600.0
df["occupancy"] = df["occupancy"].astype(int)
df["congestion"] = df["congestion"].astype(int)

# === 2) TRIP STRUCTURE FEATURES ===
df = df.sort_values(["trip_id","ts_dt"]).copy()
df["stop_seq"] = df.groupby("trip_id").cumcount().astype(np.int32)
df["prev_bdt"] = df.groupby("trip_id")["BDT"].shift(1)
df["rolling_med_bdt"] = (
    df.groupby("trip_id")["BDT"]
      .transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
)
df["prev_ts"] = df.groupby("trip_id")["ts_dt"].shift(1)
df["intra_trip_gap_sec"] = (df["ts_dt"] - df["prev_ts"]).dt.total_seconds()

for c, fill in [
    ("prev_bdt", df["BDT"].median()),
    ("rolling_med_bdt", df["BDT"].median()),
    ("intra_trip_gap_sec", df["intra_trip_gap_sec"].median()),
]:
    df[c] = df[c].fillna(fill)

# === 3) STOP CLUSTERING (DBSCAN 1m) ===
coords_rad = np.radians(df[["lat","long"]].to_numpy())
eps_rad = 1.0 / 6371000.0
db = DBSCAN(eps=eps_rad, min_samples=5, metric="haversine")
df["stop_id"] = db.fit_predict(coords_rad)



# === 3.1) SUBURB + CLUSTER HYBRID FEATURE ===
if "Suburb" in df.columns:
    df["suburb_cluster"] = df["Suburb"].astype(str) + "_" + df["stop_id"].astype(str)
else:
    df["suburb_cluster"] = df["stop_id"].astype(str)

df["suburb_cluster_id"] = pd.factorize(df["suburb_cluster"])[0]

# === 4) TRAIN/TEST SPLIT ===
y_all = df["BDT"].values
groups = df["trip_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, y_all, groups=groups))
train_df = df.iloc[tr_idx].copy()
test_df  = df.iloc[te_idx].copy()

# === 5) TRAIN-ONLY STOP MEANS (LEAKAGE SAFE, USING HYBRID) ===
global_mean = train_df["BDT"].mean()
stop_mean = train_df.groupby("suburb_cluster_id")["BDT"].mean()
train_df["stop_mean_bdt"] = train_df["suburb_cluster_id"].map(stop_mean).fillna(global_mean)
test_df["stop_mean_bdt"]  = test_df["suburb_cluster_id"].map(stop_mean).fillna(global_mean)

# === 6) DEMOGRAPHIC CLEANING ===
num_fix_cols = [
    "Estimated resident population (no.)",
    "Population density (persons/km2)",
    "Median age - persons (years)",
    "Median_tot_hhd_inc_weekly",
    "rainfall_mm"
]
for c in num_fix_cols:
    for d in (df, train_df, test_df):
        if c in d.columns:
            d[c] = (
                d[c].astype(str)
                .str.replace(",", "", regex=True)
                .str.replace(" ", "", regex=True)
                .replace("-", np.nan)
                .astype(float)
            )

# === 7) RAIN FLAG ===    Boolean value if rain_mm >0 -> 1, otherwise 0.

for d in (df, train_df, test_df):
    d["rain_flag"] = (d["rainfall_mm"] > 0).astype(int)

# === 8) FEATURE SETS ===
base_feats = [
    "hour_float","occupancy","congestion",
    "stop_seq","prev_bdt","rolling_med_bdt","intra_trip_gap_sec",
    "lat","long","suburb_cluster_id","stop_mean_bdt","rain_flag"
]
exclude_cols = set(base_feats + ["BDT","trip_id","ts_time","ts_dt","prev_ts","Suburb","suburb"])
extra_cols = [c for c in df.columns if c not in exclude_cols]
extra_numeric = [c for c in extra_cols if df[c].dtype.kind in "biufc"]
FEATS_ENR = base_feats + extra_numeric

print(f"\nBaseline: {len(base_feats)} features")
print(f"Enriched: {len(FEATS_ENR)} features (includes {len(extra_numeric)} new numeric ones): {extra_numeric}\n")

X_train_base = train_df[base_feats]
X_test_base  = test_df[base_feats]
X_train_enr  = train_df[FEATS_ENR]
X_test_enr   = test_df[FEATS_ENR]
y_train = train_df["BDT"].to_numpy()
y_test  = test_df["BDT"].to_numpy()

# === 9) METRIC FUNCTION ===
def show(name, y_true, y_pred):
    print(f"[{name}] MAE={mean_absolute_error(y_true, y_pred):.2f} | "
          f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.2f} | "
          f"R²={r2_score(y_true, y_pred):.4f}")

# === 10) XGBOOST ===
print("--- XGBoost Comparison ---")
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

xgb_params = dict(
    n_estimators=2000, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, eval_metric="mae"
)
xgb_base = xgb.XGBRegressor(**xgb_params)
xgb_enr  = xgb.XGBRegressor(**xgb_params)
xgb_base.fit(X_train_base, y_train_log, eval_set=[(X_test_base, y_test_log)], verbose=False)
xgb_enr.fit(X_train_enr,  y_train_log, eval_set=[(X_test_enr,  y_test_log)], verbose=False)

pred_base = np.expm1(xgb_base.predict(X_test_base))
pred_enr  = np.expm1(xgb_enr.predict(X_test_enr))
show("XGB baseline", y_test, pred_base)
show("XGB complete-1", y_test, pred_enr)

# === 11) LIGHTGBM ===
print("\n--- LightGBM Comparison ---")
for dset in [X_train_base, X_test_base, X_train_enr, X_test_enr]:
    if "suburb_cluster_id" in dset.columns:
        dset["suburb_cluster_id"] = dset["suburb_cluster_id"].astype("category")

lgb_params = dict(
    objective="mae", metric="l1", n_estimators=4000, learning_rate=0.03,
    num_leaves=64, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, random_state=42
)
lgb_base = lgb.LGBMRegressor(**lgb_params)
lgb_enr  = lgb.LGBMRegressor(**lgb_params)
lgb_base.fit(X_train_base, np.log1p(y_train), eval_set=[(X_test_base, np.log1p(y_test))],
             callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
lgb_enr.fit(X_train_enr, np.log1p(y_train), eval_set=[(X_test_enr, np.log1p(y_test))],
            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
pred_base = np.expm1(lgb_base.predict(X_test_base))
pred_enr  = np.expm1(lgb_enr.predict(X_test_enr))
show("LGBM baseline", y_test, pred_base)
show("LGBM complete-1", y_test, pred_enr)

# === 12) FEATURE IMPORTANCES ===
gain = xgb_enr.get_booster().get_score(importance_type="gain")
imp = pd.Series(gain).sort_values(ascending=False)
print("\nTop XGBoost feature importances:")
print(imp.head(15))



# === 13) SUBURB CLUSTER DENSITY PLOT ===
import seaborn as sns

# count how many unique clusters each suburb has
if "Suburb" in df.columns:
    suburb_counts = (
        df.groupby("Suburb")["suburb_cluster_id"]
        .nunique()
        .sort_values(ascending=False)
        .head(5)
    )

    print("\nTop 5 suburbs with the most clusters:")
    print(suburb_counts)

    # subset only those top suburbs
    top_suburbs = suburb_counts.index.tolist()
    df_top = df[df["Suburb"].isin(top_suburbs)]

    # scatter plot: latitude vs longitude, colour by suburb
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_top,
        x="long",
        y="lat",
        hue="Suburb",
        s=10,
        palette="tab10",
        alpha=0.8
    )
    plt.title("Top 5 Suburbs with Most DBSCAN Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(title="Suburb", loc="best")
    plt.show()
else:
    print("⚠️ 'Suburb' column not found — skipping suburb cluster plot.")
