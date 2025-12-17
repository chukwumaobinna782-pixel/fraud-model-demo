# model.py
import joblib, os, pathlib, pandas as pd, xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parent
model = joblib.load("xgb_fraud_2025.bin")
feature_names = joblib.load("feature_names.joblib")

def preprocess(df_raw):
    """Exactly the same steps you used in the notebook."""
    df = df_raw.copy()
    direct = ['amount_usd', 'dist_to_home_km']
    signal = ['card_present', 'foreign_ip', 'risky_ip_country', 'uncommon_os']
    df['hour_of_day'] = pd.to_datetime(df['trans_ts']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['trans_ts']).dt.dayofweek
    df['weekend'] = (df['day_of_week'] >= 5).astype(int)
    time = ['hour_of_day', 'day_of_week', 'weekend']
    cat = ['merchant_category', 'device_os', 'device_lang', 'ip_country', 'ip_isp']
    for c in cat:
        df[c] = df[c].astype('category')
    velocity = ['cust_prev_tx', 'time_since_prev_tx_sec',
                'very_quick_succession', 'amount_to_hist_avg_ratio']
    feats = direct + signal + time + cat + velocity
    return df[feats]

def predict_fraud(df_raw):
    X = preprocess(df_raw)
    prob = model.predict_proba(X)[:, 1]
    return prob
