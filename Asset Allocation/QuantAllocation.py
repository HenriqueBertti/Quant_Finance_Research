import os 
import optuna # type: ignore
import logging
import random
import datetime

import pandas as pd 
import numpy as np 
import yfinance as yf 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as sch 
import tensorflow as tf 
from tensorflow import keras #type: ignore 
from keras import layers
from scipy.spatial.distance import pdist 
from sklearn.preprocessing import StandardScaler
from arch import arch_model 

# === Settings ===
FINAL_DATE = datetime.datetime.now().strftime('2026-06-30')
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed_value = 42 
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# === Data Extraction and Cleaning ===

etfs_dict = {
    'Tech': ['SOXX', 'QQQ'],
    'Biotech': ['IBB'],
    'Healthcare': ['XLV'],
    'Energy': ['XLE'],
    'Emergents': ['IMG'],
    'Strategic_Commodities': ['REMX'],
    'Gold': ['IAU'],
    'War': ['ITA'],
    'Small Caps': ['VBR'],
    'Hedge': ['DBMF']
}

all_etfs = [item for sublist in etfs_dict.values() for item in sublist]

raw_data = yf.download(all_etfs, start='2014-01-01', end=FINAL_DATE, auto_adjust=True) 
close_df = raw_data['Close'].copy()
close_df = close_df.dropna(how='all')
volume_df = raw_data['Volume'].copy().ffill()

# === Filtering ETFs by volume and time series length ===
valid_etfs = []
MIN_DAYS_HISTORY = 252 # Minimum number of days of historical negotiation (guarantee liquidity)
MIN_ADTV = 10_000_000 # Minimum Average Daily Trading Volume to avoid slippage and liquidity issues

for t in all_etfs:
    # Skip only if the ticker is missing from either DataFrame
    if t not in close_df.columns or t not in volume_df.columns:
        continue
        
    if close_df[t].dropna().shape[0] < MIN_DAYS_HISTORY:
        print(f'ETF {t} rejected: Insufficient Historical Data')
        continue

    # Extracting the last 21 trading days safely as series
    recent_prices = close_df[t].iloc[-21:]
    recent_volumes = volume_df[t].iloc[-21:]

    # FIX: Use safe element-wise multiplication aligning indices and avoiding NaNs
    daily_volume_financial = recent_prices.multiply(recent_volumes, fill_value=0)
    adtv_21 = daily_volume_financial.mean()

    # Apply volume filter to non-FOREX assets
    if not t.endswith('=X'): # ignore FOREX ETFs
        if pd.isna(adtv_21) or adtv_21 < MIN_ADTV:
            print(f'ETF {t} rejected: Low Average Daily Trading Volume. Calc: {adtv_21}')
            continue

    # FIX: Append to valid list after passing all filters
    valid_etfs.append(t)

print(f'Valid ETFs found: {valid_etfs}')

# Safety check: Prevent running analysis if no tickers survived the filters
if not valid_etfs:
    print("Error: No ETFs passed the volume or data length filters.")
    exit()

clean_close = close_df[valid_etfs].copy()
clean_ret_log = np.log(clean_close / clean_close.shift(1)).dropna()

# Winsorizing/clipping outliers beyond 4 standard deviations
for t in valid_etfs:
    mean = clean_ret_log[t].mean()
    std = clean_ret_log[t].std()
    superior_limit = mean + (4 * std)
    inferior_limit = mean - (4 * std)

    clean_ret_log[t] = clean_ret_log[t].clip(lower=inferior_limit, upper=superior_limit)
print('Data Cleaning Completed!')

# ===============================================================================================================
# === Statistical Analysis ===  

print('Starting Statistical Analysis...')

stats_df = pd.DataFrame({
    'Mean': clean_ret_log.mean(),
    'Variance': clean_ret_log.var(),
    'Std Dev': clean_ret_log.std(),
    'Skewness': clean_ret_log.skew(),
    'Kurtosis': clean_ret_log.kurt()
})

print(stats_df)

correlation_matrix = clean_ret_log.corr()
print(correlation_matrix)

# --- Test for Normality, Stacionarity and Heteroscedasticity ---

print('Initating Tests for Normality, Stacionarity and Heteroscedasticy...')

from scipy.stats import kstest, norm, probplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

test_results = []

for column in clean_ret_log.columns:
    serie = clean_ret_log[column]

    standard_series = (serie - serie.mean()) / serie.std()
    ks_stat, ks_p_value = kstest(standard_series, 'norm')
    adf_stat, adf_p_value, _, _, _, _ = adfuller(serie)
    arch_stat, arch_p_value, _, _ = het_arch(serie)

    test_results.append({
        'Index': column,  
        'KS p-value': ks_p_value,
        'ADF p-value': adf_p_value,
        'ARCH p-value': arch_p_value
    })

df_test_results = pd.DataFrame(test_results)
print('Test Results for Normality, Stationarity and Heteroscedasticity:')
print(df_test_results)

# Statistical Interpretation:
# KS < 0.05: Series not normal (Justifies the Extreme Value Theory).
# ADF < 0.05: Series are stationary (Safe for temporal modeling).
# ARCH-LM < 0.05: There is volatility clustering (Justifies GARCH).

# === ARIMA/SARIMA model ===

import warnings
from statsmodels.tsa.arima.model import ARIMA

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- Creating a new DataFrame to store the data ---
ar_residuals = pd.DataFrame(index=clean_ret_log.index)

for col in clean_ret_log.columns:
    ar_model = ARIMA(clean_ret_log[col].dropna(), order=(1, 0, 0))
    ar_result = ar_model.fit()
    ar_residuals[col] = ar_result.resid

print('Visualizing ARIMA/SARIMA Econometric Models')
print(ar_residuals)

# ===============================================================================================================
# --- E-GARCH Model Configuration --- 

print('Calculating E-GARCH Volatility...')
egarch_volatility = pd.DataFrame()

# Using valid_etfs and clean_ret_log for volatility calculation and prediction:
for t in valid_etfs: 
    try: 
        returns = 100 * clean_ret_log[t].dropna()
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
        res = am.fit(update_freq=5, disp='off')
        egarch_volatility[t] = res.conditional_volatility / 100
    except Exception as e:
        print(f"Error calculating E-GARCH for {t}: {e}")
        continue

egarch_volatility = egarch_volatility.dropna()
print('E-GARCH Calculation Completed!')

# ===============================================================================================================
# --- Features Engineering --- 

AVAILABLE_CAPITAL = 1_000_000

print("Downloading OHLCV data for the model's technical indicators...")
ohlcv = yf.download(valid_etfs, start='1998-12-16', end=FINAL_DATE, auto_adjust=True)
ohlcv.index = ohlcv.index.tz_localize(None)
clean_close.index = clean_close.index.tz_localize(None)

print("Downloading Global Leading Indicators (VIX and 10Y Yield)...")
macro_tickers = ['^VIX', '^TNX']
macro_data = yf.download(macro_tickers, start='2015-01-01', end=FINAL_DATE, auto_adjust=True)['Close']
macro_data = macro_data.ffill()
macro_data.index = macro_data.index.tz_localize(None) # Removing timezone

macro_data['VIX_Change'] = np.log(macro_data['^VIX'] / macro_data['^VIX'].shift(1)) 
macro_data['TNX_Change'] = macro_data['^TNX'].diff()

print("Downloading Benchmark for Relative Performance...")
benchmark = yf.download('IVV', start='1998-12-16', end=FINAL_DATE, auto_adjust=True)['Close']
benchmark = benchmark.ffill()
benchmark.index = benchmark.index.tz_localize(None)

panel = []
print("Building Neural Panel with Scoring System...")

for t in valid_etfs:
    try: 
        if t not in egarch_volatility.columns: continue

        df = clean_close[[t]].copy()
        df.columns = ['Close']
        df = df.dropna()
        df['ticker'] = t

        # --- 1. EXTRA DATA FOR INDICATORS ---
        df['ret_log'] = clean_ret_log[t]
        df['High'] = ohlcv['High'][t]
        df['Low'] = ohlcv['Low'][t]
        df['Volume'] = ohlcv['Volume'][t]

        # --- 2. TECHNICAL INDICATORS CALCULATION ---
        # RSI (14 days)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))

        # EMAs and MAs
        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Volume Ratio and Momentum
        df['Volume_ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-8)
        df['Price_momentum_1d'] = df['Close'].pct_change() 

        # Stochastic (Stoch_K)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-8))

        # --- 3. POINT SYSTEM ---
        df['points_buy'] = 0
        
        df.loc[df['RSI'] <= 30, 'points_buy'] += 20
        df.loc[(df['RSI'] > 30) & (df['RSI'] <= 40), 'points_buy'] += 10
        df.loc[df['RSI'] > 50, 'points_buy'] += 10
        
        ema_bullish = (df['EMA_9'] > df['EMA_50']) & (df['EMA_21'] > df['EMA_50'])
        df.loc[ema_bullish, 'points_buy'] += 15
        
        df.loc[df['MACD'] > 0, 'points_buy'] += 15
        
        macd_growing = df['MACD_hist'] > df['MACD_hist'].shift(1)
        df.loc[macd_growing, 'points_buy'] += 5
        
        df.loc[df['Volume_ratio'] >= 1.5, 'points_buy'] += 15
        df.loc[df['Price_momentum_1d'] > 0.02, 'points_buy'] += 5
        df.loc[df['Stoch_K'] <= 20, 'points_buy'] += 5

        # --- 4. SIMPLIFIED SELL TRIGGER ---
        ema9_above_ma50_prev = df['EMA_9'].shift(1) > df['MA_50'].shift(1)
        ema9_below_ma50_now = df['EMA_9'] <= df['MA_50']
        # Converting boolean (True/False) to int (1/0)
        df['sell_trigger'] = (ema9_above_ma50_prev & ema9_below_ma50_now).astype(int) 

        # --- 5. RISK AND MACRO ---
        df['egarch_vol'] = egarch_volatility[t]
        df = df.join(macro_data[['VIX_Change', 'TNX_Change']], how='inner')

        # --- 6. TARGET MACHINE LEARNING ---
        df['target_weekly'] = np.log(df['Close'].shift(-5) / df['Close'].shift(-1))
        df['rank_score'] = df['target_weekly'] / (df['egarch_vol'] * np.sqrt(5) + 1e-6)
        
        panel.append(df)
    except Exception as e: 
        print(f'Error processing ETF {t}: {e}')
        continue

data = pd.concat(panel).dropna()
data.index = data.index.tz_localize(None)
data['rank'] = data.groupby(data.index)['rank_score'].rank(pct=True)

# ===============================================================================================================
# --- DATA PREPARATION (Chronological Split TRAIN / VAL) ---

features = ['points_buy', 'sell_trigger', 'egarch_vol', 'VIX_Change', 'TNX_Change']

# 1. Isolating the latest date for final inference
latest_date = data.index.max()
latest_data = data[data.index == latest_date].copy()

# Data before the latest date
historical_data = data[data.index < latest_date].copy()

train_data = historical_data[historical_data.index < '2024-12-31']
val_data = historical_data[historical_data.index >= '2025-01-01']

# 2. The Scaler learns mean/std ONLY from the past (Pure Alpha)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
y_train = (train_data['rank_score'] > 0).astype(int).values

X_val = scaler.transform(val_data[features])
y_val = (val_data['rank_score'] > 0).astype(int).values

# 3. Save today's isolated data for inference
X_latest_alpha = scaler.transform(latest_data[features])

# ===============================================================================================================
# --- XGBoost Model & Optuna Optimization ---

import xgboost as xgb
from sklearn.metrics import accuracy_score

print("\nStarting XGBoost hyperparameter optimization with Optuna...")

def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"\nBest architecture found: {study.best_params}")

print("\nTraining the Final XGBoost Model with all available data...")
X_all = np.vstack((X_train, X_val))
y_all = np.concatenate((y_train, y_val))

best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['random_state'] = 42

final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_all, y_all)

print("Model successfully trained!")

# ===============================================================================================================
# --- Plotting Feature Importance ---

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

importances = final_model.feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=True)

bars = feature_imp.plot(kind='barh', color='#00ff00', ax=ax, edgecolor='none')
ax.set_title("AI Feature Importance (XGBoost)", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Importance Weight in Algorithm", fontsize=10, color='white')
ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.7)

for ax_obj in [ax]:
    ax_obj.spines['top'].set_visible(False)
    ax_obj.spines['right'].set_visible(False)
    ax_obj.spines['left'].set_color('#30363d')
    ax_obj.spines['bottom'].set_color('#30363d')

plt.tight_layout()
plt.show()

# ===============================================================================================================
# --- Allocation ---

# PARAMETERS 
ALPHA_POWER = 2.0  # AI signal weight
RISK_FLOOR = 0.05  # 5% risk floor 

# ROLLING DOWNSIDE RISK
def get_rolling_downside_risk(ticker, window=252):
    hist = clean_ret_log[ticker].tail(window)
    neg_rets = hist[hist < 0]
    if len(neg_rets) < 2: return RISK_FLOOR
    return neg_rets.std() * np.sqrt(252)

efficiency_scores = {}
print("\nGenerating Allocation:")

for ticker in valid_etfs:
    ticker_feat = X_latest_alpha[latest_data['ticker'] == ticker]
    if len(ticker_feat) == 0:
        ai_alpha = 0.50
    else:
        ai_alpha = float(final_model.predict_proba(ticker_feat)[0][1])
    
    # SIGMA: Moving downside risk with safety floor
    raw_sigma = get_rolling_downside_risk(ticker)
    adjusted_sigma = max(raw_sigma, RISK_FLOOR) 
    
    # FINAL EFFICIENCY
    score = (ai_alpha ** ALPHA_POWER) / (adjusted_sigma + 1e-6)
    efficiency_scores[ticker] = score
    
    print(f"{ticker:4} | Alpha: {ai_alpha:.4f} | Sigma: {raw_sigma*100:5.2f}% | Efficiency: {score:.2f}")

# INSTITUTIONAL NORMALIZATION (40% CLIP)
weights = pd.Series(efficiency_scores)
weights = weights / weights.sum()
weights = weights.clip(upper=0.40) # Prevents over-concentration
weights /= weights.sum()

print("\n --- FINAL PORTFOLIO ---")
for ticker, weight in weights.sort_values(ascending=False).items():
    print(f"{ticker}: {weight*100:.2f}%")