"""
Kairos v5 — Flask Backend
Regime-Aware Financial Trend Classification
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import yfinance as yf
import time
import warnings

try:
    import joblib
    import torch
    import xgboost as xgb
    from model import ProposedModel, CNNBaseline, LSTMBaseline, CNNPred2DBaseline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static")

# Restrict CORS to same-origin in production; allow all only in dev
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
CORS(app, origins=ALLOWED_ORIGINS)

# ── Model Loading ─────────────────────────────────────────────────────────────
SCALER_PATH = "kairos_v5_scaler.pkl"

MODELS = {
    'PROPOSED': None,
    '1D-CNN': None,
    'CNNPRED-2D': None,
    'LSTM': None,
    'XGBOOST': None,
    'DUMMY': None
}
kairos_scaler = None
device = None

if ML_AVAILABLE:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        if os.path.exists(SCALER_PATH):
            kairos_scaler = joblib.load(SCALER_PATH)
            
            # 1. Proposed Model
            if os.path.exists("kairos_v5_model.pth"):
                m = ProposedModel(C=7, T=60).to(device)
                m.load_state_dict(torch.load("kairos_v5_model.pth", map_location=device))
                m.eval()
                MODELS['PROPOSED'] = m
            
            # 2. 1D-CNN
            if os.path.exists("kairos_v5_cnn.pth"):
                m = CNNBaseline(C=7).to(device)
                m.load_state_dict(torch.load("kairos_v5_cnn.pth", map_location=device))
                m.eval()
                MODELS['1D-CNN'] = m
                
            # 3. CNNPred-2D
            if os.path.exists("kairos_v5_cnnpred.pth"):
                m = CNNPred2DBaseline(C=7, T=60).to(device)
                m.load_state_dict(torch.load("kairos_v5_cnnpred.pth", map_location=device))
                m.eval()
                MODELS['CNNPRED-2D'] = m
                
            # 4. LSTM
            if os.path.exists("kairos_v5_lstm.pth"):
                m = LSTMBaseline(C=7).to(device)
                m.load_state_dict(torch.load("kairos_v5_lstm.pth", map_location=device))
                m.eval()
                MODELS['LSTM'] = m
                
            # 5. XGBoost
            if os.path.exists("kairos_v5_xgb.json"):
                m = xgb.XGBClassifier()
                m.load_model("kairos_v5_xgb.json")
                MODELS['XGBOOST'] = m
                
            # 6. Dummy
            if os.path.exists("kairos_v5_dummy.pkl"):
                MODELS['DUMMY'] = joblib.load("kairos_v5_dummy.pkl")

            print("✅ Loaded all available machine learning models.")
        else:
            print("⚠️ Scaler not found. Falling back to rule-based signal.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Error loading models: {e}. Falling back to rule-based signal.")
else:
    print("⚠️ PyTorch/XGBoost not installed. Falling back to rule-based signal.")

# ── Constants (mirrors notebook) ──────────────────────────────────────────────
LOOKBACK   = 60
HORIZON    = 5
VOL_WINDOW = 30
CLASS_NAMES = ["Down", "Neutral", "Up"]
CLASS_COLORS = ["#9c3b3b", "#8c8c8c", "#4a7c59"]

# FROZEN feature group: 'Base price+tech (7)' — MUST match notebook training order
BASE_FEATURE_COLUMNS = [
    "log_ret_close", "log_ret_open", "log_ret_high", "log_ret_low",
    "rolling_vol_20", "rsi_14", "macd_norm",
]

# ── Feature Engineering (mirrors notebook CELL 3 exactly) ─────────────────────
def compute_features(close, high, low, open_, volume, vix_close):
    feat = pd.DataFrame(index=close.index)

    # 1-4: OHLC log-returns
    feat["log_ret_close"] = np.log(close / close.shift(1))
    feat["log_ret_open"]  = np.log(open_ / open_.shift(1))
    feat["log_ret_high"]  = np.log(high  / high.shift(1))
    feat["log_ret_low"]   = np.log(low   / low.shift(1))

    # 5: 20-day rolling realized volatility
    feat["rolling_vol_20"] = (
        feat["log_ret_close"].rolling(window=20, min_periods=20).std()
    )

    # 6: RSI-14 normalized to [0, 1]  ← notebook divides by 100!
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_l = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_g / (avg_l + 1e-10)
    feat["rsi_14"] = (100 - (100 / (1 + rs))) / 100.0

    # 7: MACD signal normalized by price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    feat["macd_norm"] = (ema12 - ema26) / close

    # ── Extra features (for display / reality calc, NOT fed to model) ─────
    feat["log_vix"] = np.log(vix_close.clip(lower=1e-6))

    vol_mean = volume.rolling(window=20, min_periods=20).mean()
    vol_std  = volume.rolling(window=20, min_periods=20).std()
    feat["vol_shock_20"] = (volume - vol_mean) / (vol_std + 1e-8)

    feat["momentum_5d"] = np.log(close / close.shift(5))

    ema50  = close.ewm(span=50,  adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    feat["ema_distance"] = (ema50 - ema200) / close

    return feat.dropna()

def generate_labels(close_series, index, lambda_val=1.0, horizon=5, vol_window=30):
    close_series = pd.Series(close_series, index=index).astype(float)
    future_ret = np.log(close_series.shift(-horizon) / close_series)
    past_ret   = np.log(close_series / close_series.shift(1))
    sigma_t    = past_ret.rolling(window=vol_window, min_periods=vol_window).std()
    epsilon    = lambda_val * sigma_t
    
    labels = pd.Series(np.nan, index=index, dtype=float)
    labels[future_ret > epsilon]  = 2
    labels[future_ret < -epsilon] = 0
    neutral_mask = (future_ret >= -epsilon) & (future_ret <= epsilon)
    labels[neutral_mask] = 1
    return labels

def rule_based_signal(feat_row, close_series, current_idx):
    score = 0.0
    rsi = feat_row.get("rsi_14", 50.0) / 100.0
    if rsi < 0.35: score -= 1.5
    elif rsi > 0.65: score += 1.5
    mom = feat_row.get("momentum_5d", 0.0)
    score += mom * 30
    ema = feat_row.get("ema_distance", 0.0)
    score += ema * 20
    macd = feat_row.get("macd_norm", 0.0)
    score += macd * 80
    vol = feat_row.get("rolling_vol_20", 0.01)
    neutral_band = vol * 12
    log_vix = feat_row.get("log_vix", 3.0)
    if log_vix > 3.3: score -= 0.6
    elif log_vix < 2.8: score += 0.4
    vol_shock = feat_row.get("vol_shock_20", 0.0)
    if abs(vol_shock) > 1.5: score *= 1.2
    
    if score > neutral_band: pred = 2
    elif score < -neutral_band: pred = 0
    else: pred = 1
    
    raw_probs = np.array([
        max(0.05, 0.33 - score * 0.15),
        max(0.05, 0.34 - abs(score) * 0.10),
        max(0.05, 0.33 + score * 0.15),
    ])
    raw_probs = np.clip(raw_probs, 0.0, 1.0)
    raw_probs /= raw_probs.sum()
    return pred, raw_probs.tolist()

# ── API Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

DATA_CACHE = {}
CACHE_TTL = 300

def get_stock_data(ticker):
    now = time.time()
    if ticker in DATA_CACHE and now - DATA_CACHE[ticker]['time'] < CACHE_TTL:
        return DATA_CACHE[ticker]['data']
    raw = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    DATA_CACHE[ticker] = {'time': now, 'data': raw}
    return raw

def get_vix_data():
    ticker = "^VIX"
    now = time.time()
    if ticker in DATA_CACHE and now - DATA_CACHE[ticker]['time'] < CACHE_TTL:
        return DATA_CACHE[ticker]['data']
    vix_raw = yf.download(ticker, period="3y", auto_adjust=True, progress=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    DATA_CACHE[ticker] = {'time': now, 'data': vix_raw}
    return vix_raw

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request body."}), 400

    ticker = data.get("ticker", "^GSPC").strip().upper()
    target_date = data.get("date", None)
    model_id = data.get("model", "PROPOSED").upper()

    if not ticker or len(ticker) > 10:
        return jsonify({"error": "Invalid ticker symbol."}), 400
    import re
    if not re.match(r'^[A-Z0-9.^\-]+$', ticker):
        return jsonify({"error": "Ticker contains invalid characters."}), 400
    if target_date:
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', str(target_date)):
            return jsonify({"error": "Date must be in YYYY-MM-DD format."}), 400
    
    try:
        full_raw = get_stock_data(ticker)
        if full_raw.empty:
            return jsonify({"error": f"No data found for ticker '{ticker}'"}), 400

        # Ground Truth Reality Calculation
        actual_reality = "AWAITING DATA"
        actual_reality_color = "#888888"
        if target_date:
            try:
                # Get the integer index of the target date to look ahead 5 trading days
                date_mask = full_raw.index <= pd.Timestamp(target_date)
                if date_mask.any():
                    target_idx = np.where(date_mask)[0][-1]
                    if target_idx + HORIZON < len(full_raw):
                        target_close = full_raw['Close'].iloc[target_idx]
                        future_close = full_raw['Close'].iloc[target_idx + HORIZON]
                        future_ret = np.log(future_close / target_close)
                        
                        past_data = full_raw.iloc[:target_idx+1]
                        if len(past_data) >= VOL_WINDOW:
                            past_ret = np.log(past_data['Close'] / past_data['Close'].shift(1))
                            sigma_t = past_ret.tail(VOL_WINDOW).std()
                            epsilon = 1.0 * sigma_t
                            
                            if future_ret > epsilon:
                                actual_reality = "UP"
                                actual_reality_color = "#4a7c59"
                            elif future_ret < -epsilon:
                                actual_reality = "DOWN"
                                actual_reality_color = "#9c3b3b"
                            else:
                                actual_reality = "NEUTRAL"
                                actual_reality_color = "#8c8c8c"
            except Exception as e:
                print(f"Error calculating reality: {e}")

        # Truncate data for model input
        raw = full_raw.copy()
        if target_date:
            raw = raw.loc[:target_date]
            if raw.empty:
                return jsonify({"error": f"No data up to {target_date}"}), 400

        vix_raw = get_vix_data()
        vix_aligned = vix_raw['Close'].reindex(raw.index).ffill().bfill()
        feat = compute_features(raw['Close'], raw['High'], raw['Low'], raw['Open'], raw['Volume'], vix_aligned)
        
        if feat.empty or len(feat) < LOOKBACK:
            return jsonify({"error": "Not enough data to form a 60-day window. Try an older stock."}), 400
        
        close = raw['Close'].reindex(feat.index)
        latest = feat.iloc[-1]
        latest_date = str(latest.name.date())
        current_price = close.iloc[-1]
        prev_price = close.iloc[-2] if len(close) > 1 else current_price
        price_change = current_price - prev_price
        price_pct = (price_change / prev_price) * 100
        current_vix = vix_aligned.loc[latest.name]
        
        regime = "HIGH VOLATILITY" if current_vix > 20 else "LOW VOLATILITY"

        # ── Inference Engine ──────────────────────────────────────────────────
        pred_class = 1
        probs = [0.33, 0.34, 0.33]
        
        if kairos_scaler is not None:
            window_df = feat[BASE_FEATURE_COLUMNS].tail(LOOKBACK)
            if len(window_df) == LOOKBACK:
                X_tensor = np.expand_dims(window_df.values.T, axis=0)
                mean = kairos_scaler['mean']
                std = kairos_scaler['std']
                X_norm = (X_tensor - mean) / std
                
                if model_id == 'ENSEMBLE':
                    p1 = p2 = p3 = np.array([0.33, 0.34, 0.33])
                    if MODELS['PROPOSED']:
                        with torch.no_grad():
                            t = torch.tensor(X_norm, dtype=torch.float32).to(device)
                            p1 = torch.softmax(MODELS['PROPOSED'](t), dim=1).cpu().numpy()[0]
                    if MODELS['LSTM']:
                        with torch.no_grad():
                            t = torch.tensor(X_norm, dtype=torch.float32).to(device)
                            p2 = torch.softmax(MODELS['LSTM'](t), dim=1).cpu().numpy()[0]
                    if MODELS['XGBOOST']:
                        X_flat = X_norm.reshape(1, -1)
                        p3 = MODELS['XGBOOST'].predict_proba(X_flat)[0]
                    probs_arr = (p1 + p2 + p3) / 3.0
                    pred_class = int(np.argmax(probs_arr))
                    probs = probs_arr.tolist()

                elif model_id in ['XGBOOST', 'DUMMY'] and MODELS.get(model_id):
                    X_flat = X_norm.reshape(1, -1)
                    probs_arr = MODELS[model_id].predict_proba(X_flat)[0]
                    pred_class = int(np.argmax(probs_arr))
                    probs = probs_arr.tolist()

                elif model_id in ['PROPOSED', '1D-CNN', 'CNNPRED-2D', 'LSTM'] and MODELS.get(model_id):
                    with torch.no_grad():
                        t = torch.tensor(X_norm, dtype=torch.float32).to(device)
                        logits = MODELS[model_id](t)
                        probs_arr = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        pred_class = int(np.argmax(probs_arr))
                        probs = probs_arr.tolist()
                else:
                    pred_class, probs = rule_based_signal(latest.to_dict(), close, -1)
            else:
                pred_class, probs = rule_based_signal(latest.to_dict(), close, -1)
        else:
            pred_class, probs = rule_based_signal(latest.to_dict(), close, -1)

        # Chart Data
        raw_aligned = raw.reindex(feat.index)
        labels_series = generate_labels(raw_aligned["Close"], feat.index, lambda_val=1.0)
        recent_feat = feat.tail(90)
        recent_close = close.tail(90)
        recent_labels = labels_series.reindex(recent_feat.index)

        chart_data = {
            "dates":  [str(d.date()) for d in recent_close.index],
            "prices": [round(float(p), 2) for p in recent_close.values],
            "labels": [int(l) if not np.isnan(l) else 1 for l in recent_labels.values],
        }

        feature_display = {
            "Log Ret (Close)":    f"{float(latest['log_ret_close']):.6f}",
            "Log Ret (Open)":     f"{float(latest['log_ret_open']):.6f}",
            "Log Ret (High)":     f"{float(latest['log_ret_high']):.6f}",
            "Log Ret (Low)":      f"{float(latest['log_ret_low']):.6f}",
            "Rolling Vol (20d)":  f"{float(latest['rolling_vol_20']):.4f}",
            "RSI (14)":           f"{float(latest['rsi_14']) * 100:.2f}",
            "MACD (norm)":        f"{float(latest['macd_norm']):.6f}",
            "VIX (log)":          f"{float(latest['log_vix']):.4f}",
        }

        return jsonify({
            "ticker":        ticker,
            "date":          latest_date,
            "model_id":      model_id,
            "current_price": round(current_price, 2),
            "price_change":  round(price_change, 2),
            "price_pct":     round(price_pct, 2),
            "vix":           round(current_vix, 2),
            "regime":        regime,
            "reality":       {
                "label": actual_reality,
                "color": actual_reality_color
            },
            "prediction":    {
                "class":  pred_class,
                "label":  CLASS_NAMES[pred_class].upper(),
                "color":  CLASS_COLORS[pred_class],
                "probs":  [round(p, 4) for p in probs],
            },
            "features":   feature_display,
            "chart":      chart_data,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {str(e)}"}), 500


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, port=int(os.environ.get("PORT", 5050)))
