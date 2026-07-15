import os 
import logging
import random
import datetime

import optuna
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
import xgboost as xgb
from sklearn.metrics import accuracy_score

# --- Model Configuration --- 
FINAL_DATE = datetime.datetime.now().strftime('%Y-%m-%d') # Automated to current date
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed_value = 42 
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

CAPITAL_DISPONIVEL = 1_000_000

# --- Data Loading and Processing ---
etfs_dict = {
    'Emergents': ['IEMG', 'VWO'],
    'Energy': ['XLE'],
    'Defense': ['ITA'],
    'Strategic_Commodities': ['REMX'],
    'Tech': ['VGT', 'XLK', 'QQQ'],
    'Biotech': ['XBI', 'XLV'],
    'Small Caps': ['VBR'],
    'International': ['EXUS'],
    'Gold': ['IAU'],
    'Alternatives': ['DBMF']
}

all_etfs = [item for sublist in etfs_dict.values() for item in sublist]

raw_data = yf.download(all_etfs, start='2014-01-01', end=FINAL_DATE, auto_adjust=True) 

# 10-11 years timeframe to ensure high liquidity and analyze different economic cycles
close_df = raw_data['Close'].copy().ffill()
close_df = close_df.dropna(how='all')

volume_df = raw_data['Volume'].copy().ffill()

valid_etfs = []
MIN_DAYS_HISTORY = 252 

# For a correct application of HRP, a filter of at least 1 trading year is applied.
# This is used because not all companies may be listed since the initial data period.
MIN_ADTV = 10_000_000

for t in all_etfs:
    if t not in close_df.columns or t not in volume_df.columns:
        continue
    if close_df[t].dropna().shape[0] < MIN_DAYS_HISTORY:
        print(f'ETF {t} was rejected: Time history too short')
        continue

    precos_recentes = close_df[t].iloc[-21:]
    volumes_recentes = volume_df[t].iloc[-21:]

    adtv_21 = (precos_recentes * volumes_recentes).mean()
    
    if not t.endswith('=X'): # Ignores currencies, which do not have standard volume
        if pd.isna(adtv_21) or adtv_21 < MIN_ADTV:
            print(f'ETF {t} rejected: Low liquidity (Calculated: {adtv_21})')
            continue
            
    valid_etfs.append(t)

print(f"Validated ETFs: {valid_etfs}")

clean_close = close_df[valid_etfs].copy()
clean_ret_log = np.log(clean_close/clean_close.shift(1))

for t in valid_etfs:
    media = clean_ret_log[t].mean()
    desvio = clean_ret_log[t].std()
    limite_superior = media + (4 * desvio)
    limite_inferior = media - (4 * desvio)

    clean_ret_log[t] = clean_ret_log[t].clip(lower=limite_inferior, upper=limite_superior)
    
print('Portfolio Data Collection and Analysis Completed')

# --- E-GARCH Model Configurations --- 
egarch_volatility = pd.DataFrame()

# Using valid_etfs and clean_ret_log for volatility calculation and forecasting:
for t in valid_etfs: 
    try: 
        returns = 100 * clean_ret_log[t].dropna()
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
        res = am.fit(update_freq=5, disp='off')
        egarch_volatility[t] = res.conditional_volatility / 100
    except: 
        continue

egarch_volatility = egarch_volatility.dropna()

# --- Features --- 
# The .tz_localize function will remove timezone differences between data.
print("Downloading OHLCV data for model technical indicators...")

ohlcv = yf.download(valid_etfs, start='1998-12-16', end=FINAL_DATE, auto_adjust=True)
ohlcv.index = ohlcv.index.tz_localize(None)
clean_close.index = clean_close.index.tz_localize(None)

print("Downloading Global Leading Indicators (VIX and 10Y Yields)...")
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
        df['Price_momentum_1d'] = df['Close'].pct_change() # 24h momentum adapted to daily

        # Stochastic (Stoch_K)
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-8))

        # --- 3. POINTS SYSTEM (ORTHOGONALIZATION SUGGESTION) ---
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
        # Converting boolean (True/False) to number (1/0) for AI understanding
        df['sell_trigger'] = (ema9_above_ma50_prev & ema9_below_ma50_now).astype(int) 

        # --- 5. RISK AND MACRO (Fund Shielding) ---
        df['egarch_vol'] = egarch_volatility[t]
        df = df.join(macro_data[['VIX_Change', 'TNX_Change']], how='inner')

        # --- 6. MACHINE LEARNING TARGET ---
        df['target_weekly'] = np.log(df['Close'].shift(-5) / df['Close'].shift(-1))
        df['rank_score'] = df['target_weekly'] / (df['egarch_vol'] * np.sqrt(5) + 1e-6)
        
        panel.append(df)
    except Exception as e: 
        print(f'Error in ETF {t}: {e}')
        continue

data = pd.concat(panel).dropna()
data.index = data.index.tz_localize(None)
data['rank'] = data.groupby(data.index)['rank_score'].rank(pct=True)

# The AI now crosses the Points System with Macroeconomics
features = ['points_buy', 'sell_trigger', 'egarch_vol', 'VIX_Change', 'TNX_Change']

# --- DATA PREPARATION (Chronological Cut TRAIN / VAL) ---
train_data = data[data.index < '2024-12-31']
val_data = data[(data.index >= '2025-01-01') & (data.index < data.index.max())]

latest_date = data.index.max()
latest_data = data[data.index == latest_date].copy()

# The Scaler learns the mean/deviation ONLY from the past (Pure Alpha)
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
y_train = (train_data['rank_score'] > 0).astype(int).values

X_val = scaler.transform(val_data[features])
y_val = (val_data['rank_score'] > 0).astype(int).values

# Keep isolated today's data for final inference
X_latest_alpha = scaler.transform(latest_data[features])

print("Starting Hyperparameter Optimization (XGBoost) with Optuna...")

def objective(trial):
    # Search space for best market parameters
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
    
    # Instantiate and train tree model
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Accuracy evaluation on validation base (Out-of-sample)
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    return accuracy

# Runs 30 trials to find the best decision tree
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print(f"\n✅ Best architecture found:")
print(study.best_params)

print("\nTraining the Super XGBoost Model with all available data...")

# Combine Train and Validation for the final model to have the latest market "picture"
X_all = np.vstack((X_train, X_val))
y_all = np.concatenate((y_train, y_val))

# Retrieves best parameters from Optuna
best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['random_state'] = 42

# Trains definitive model
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_all, y_all)

print("✅ Model successfully trained!")

# --- PLOT: AI'S BRAIN (Feature Importance) ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Extracts weight given by AI to each indicator
importances = final_model.feature_importances_
feature_imp = pd.Series(importances, index=features).sort_values(ascending=True)

# Generates aesthetic bar chart
bars = feature_imp.plot(kind='barh', color='#00ff00', ax=ax, edgecolor='none')
ax.set_title("What is the AI looking at? (Variable Importance)", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Importance Weight in Algorithm", fontsize=10, color='white')
ax.grid(axis='x', color='#30363d', linestyle='--', alpha=0.7)

for ax_obj in [ax]:
    ax_obj.spines['top'].set_visible(False)
    ax_obj.spines['right'].set_visible(False)
    ax_obj.spines['left'].set_color('#30363d')
    ax_obj.spines['bottom'].set_color('#30363d')

plt.tight_layout()
plt.savefig("feature_importance.png") # Changed from plt.show() to savefig for automation compatibility
plt.close()

# --- Allocation ---

# PARAMETERS 
ALPHA_POWER = 2.0  # Weight of AI signal
RISK_FLOOR = 0.05  # 5% risk floor 

# 2. SIGMA ENGINE: ROLLING DOWNSIDE RISK 
def get_rolling_downside_risk(ticker, window=252):
    # Gets only the last 252 days of returns up to the current date
    hist = clean_ret_log[ticker].tail(window)
    neg_rets = hist[hist < 0]
    if len(neg_rets) < 2: return RISK_FLOOR
    return neg_rets.std() * np.sqrt(252)

# 3. ALLOCATION EXECUTION
efficiency_scores = {}
print("Generating Allocation:")

for ticker in valid_etfs:
    # Alpha: AI Prediction (using blind scaler)
    ticker_feat = X_latest_alpha[latest_data['ticker'] == ticker]
    if len(ticker_feat) == 0:
        ai_alpha = 0.50
    else:
        ai_alpha = float(final_model.predict_proba(ticker_feat)[0][1])
    
    # SIGMA: Rolling downside risk with Safety Floor
    raw_sigma = get_rolling_downside_risk(ticker)
    adjusted_sigma = max(raw_sigma, RISK_FLOOR) # Leveling for ultra-safe assets
    
    # FINAL EFFICIENCY
    score = (ai_alpha ** ALPHA_POWER) / (adjusted_sigma + 1e-6)
    efficiency_scores[ticker] = score
    
    print(f"{ticker:4} | Alpha: {ai_alpha:.4f} | Sigma: {raw_sigma*100:5.2f}% | Efficiency: {score:.2f}")

# 4. INSTITUTIONAL NORMALIZATION (CLIP 40%)
weights = pd.Series(efficiency_scores)
weights = weights / weights.sum()
weights = weights.clip(upper=0.40) # Prevents excessive concentration
weights /= weights.sum()

print("\n --- FINAL PORTFOLIO ---")
for ticker, weight in weights.sort_values(ascending=False).items():
    print(f"{ticker}: {weight*100:.2f}%")
    
# --- Plotting Correlation and Final Allocations ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 1.2]})
bg_plot_color = '#121212'
fig.patch.set_facecolor(bg_plot_color)
ax1.set_facecolor(bg_plot_color)
ax2.set_facecolor(bg_plot_color)

# Left Plot: Feature Importances
ax1.barh(feature_imp.index, feature_imp.values, color='#00aaff', edgecolor='none')
ax1.set_title("AI Brain: Variable Importance (XGBoost)", fontsize=14, fontweight='bold', color='white')
ax1.set_xlabel("Decision Weight in Algorithm", fontsize=10, color='white')
ax1.grid(axis='x', color='#30363d', linestyle='--', alpha=0.7)

# Right Plot: Financial Allocations
pesos_ordenados = weights.sort_values(ascending=True)
valores_financeiros = pesos_ordenados * CAPITAL_DISPONIVEL

bars2 = ax2.barh(pesos_ordenados.index, valores_financeiros.values, color='#00ff00', edgecolor='none')
ax2.set_title(f"Allocation (Total: R$ {CAPITAL_DISPONIVEL:,.2f})", fontsize=14, fontweight='bold', color='white')
ax2.set_xlabel("Invested Value (R$)", fontsize=10, color='white')
ax2.grid(axis='x', color='#30363d', linestyle='--', alpha=0.7)

for bar in bars2:
    width = bar.get_width()
    ax2.annotate(f'R$ {width:,.0f}',
                 xy=(width, bar.get_y() + bar.get_height() / 2),
                 xytext=(5, 0),
                 textcoords="offset points",
                 ha='left', va='center', fontsize=10, color='white', fontweight='bold')
                 
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#30363d')
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig("final_allocation.png") # Automatically saving output plots
plt.close()

print("Initiating Out-of-Sample Backtest...")