import os 
import optuna # type: ignore
import logging
import random
import datetime
import warnings

import pandas as pd 
import numpy as np 
import yfinance as yf 
import seaborn as sns
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as sch 
import tensorflow as tf 
from scipy.spatial.distance import pdist 
from sklearn.preprocessing import StandardScaler
from arch import arch_model 
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import psutil

# === Settings ===
FINAL_DATE = datetime.datetime.now().strftime('2026-06-30')
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

seed_value = 42 
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ===============================================================================================================
# === Transaction Costs Module ===
# ===============================================================================================================

def apply_transaction_costs(weights_prev, weights_new, liquidity_profile, k_impact):
    """
    Computes the total transaction cost for a single rebalance event.
    Cost_i = spread_i + k_impact × |Δw_i|       (in bps)
    Total  = Σ( |Δw_i| × Cost_i / 10,000 )      (in decimal)
    """
    etf_list = list(set(list(weights_prev.keys()) + list(weights_new.keys())))
    cost_details = {}
    total_cost = 0.0

    for etf in etf_list:
        w_prev = weights_prev.get(etf, 0.0)
        w_new = weights_new.get(etf, 0.0)
        turnover = abs(w_new - w_prev)

        if turnover == 0:
            continue

        spread_bps = liquidity_profile.get(etf, 5.0)  
        impact_bps = k_impact * turnover * 10000       
        commission_bps = 0.0                            

        total_cost_bps = spread_bps + impact_bps + commission_bps
        cost_decimal = turnover * total_cost_bps / 10000

        total_cost += cost_decimal

    return total_cost

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

print('Downloading ETF Data...')
raw_data = yf.download(all_etfs, start='2014-01-01', end=FINAL_DATE, auto_adjust=True) 
close_df = raw_data['Close'].copy().dropna(how='all')
volume_df = raw_data['Volume'].copy().ffill()

valid_etfs = []
MIN_DAYS_HISTORY = 252 
MIN_ADTV = 10_000_000 

for t in all_etfs:
    if t not in close_df.columns or t not in volume_df.columns:
        continue
    if close_df[t].dropna().shape[0] < MIN_DAYS_HISTORY:
        continue

    recent_prices = close_df[t].iloc[-21:]
    recent_volumes = volume_df[t].iloc[-21:]
    daily_volume_financial = recent_prices.multiply(recent_volumes, fill_value=0)
    adtv_21 = daily_volume_financial.mean()

    if not t.endswith('=X'): 
        if pd.isna(adtv_21) or adtv_21 < MIN_ADTV:
            continue

    valid_etfs.append(t)

clean_close = close_df[valid_etfs].copy()
clean_ret_log = np.log(clean_close / clean_close.shift(1)).dropna()

for t in valid_etfs:
    mean = clean_ret_log[t].mean()
    std = clean_ret_log[t].std()
    clean_ret_log[t] = clean_ret_log[t].clip(lower=mean - (4 * std), upper=mean + (4 * std))

# ===============================================================================================================
# --- E-GARCH Model Configuration --- 

print('Calculating E-GARCH Volatility (t-Student)...')
egarch_volatility = pd.DataFrame()

for t in valid_etfs: 
    try: 
        returns = 100 * clean_ret_log[t].dropna()
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='t')
        res = am.fit(update_freq=5, disp='off')
        egarch_volatility[t] = res.conditional_volatility / 100
    except Exception as e:
        continue

egarch_volatility = egarch_volatility.dropna()

# ===============================================================================================================
# --- Features Engineering (REMOVED DISCRETIONARY POINTS) --- 

print("Downloading OHLCV and Macro data...")
ohlcv = yf.download(valid_etfs, start='1998-12-16', end=FINAL_DATE, auto_adjust=True)
ohlcv.index = ohlcv.index.tz_localize(None)
clean_close.index = clean_close.index.tz_localize(None)

macro_tickers = ['^VIX', '^TNX']
macro_data = yf.download(macro_tickers, start='2015-01-01', end=FINAL_DATE, auto_adjust=True)['Close']
macro_data = macro_data.ffill()
macro_data.index = macro_data.index.tz_localize(None) 

macro_data['VIX_Change'] = np.log(macro_data['^VIX'] / macro_data['^VIX'].shift(1)) 
macro_data['TNX_Change'] = macro_data['^TNX'].diff()

panel = []
print("Building Feature Panel (Pure Technical & Macro Data)...")

for t in valid_etfs:
    try: 
        if t not in egarch_volatility.columns: continue

        df = clean_close[[t]].copy()
        df.columns = ['Close']
        df = df.dropna()
        df['ticker'] = t
        df['Volume'] = ohlcv['Volume'][t]
        df['High'] = ohlcv['High'][t]
        df['Low'] = ohlcv['Low'][t]

        # Raw Indicators (Let Machine Learning find the rules)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))

        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        df['Volume_ratio'] = df['Volume'] / (df['Volume'].rolling(window=20).mean() + 1e-8)
        df['Price_momentum_1d'] = df['Close'].pct_change() 

        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14 + 1e-8))

        df['egarch_vol'] = egarch_volatility[t]
        df = df.join(macro_data[['VIX_Change', 'TNX_Change']], how='inner')

        # Continuous Target Prep
        df['target_weekly'] = np.log(df['Close'].shift(-5) / df['Close'])
        df['rank_score'] = df['target_weekly'] / (df['egarch_vol'] * np.sqrt(5) + 1e-6)
        
        panel.append(df)
    except Exception as e: 
        continue

data = pd.concat(panel).dropna()
data.index = data.index.tz_localize(None)
# Cross-sectional target percentile [0, 1] - Continuous Regression Target
data['rank'] = data.groupby(data.index)['rank_score'].rank(pct=True)

# ===============================================================================================================
# --- DATA PREPARATION ---

features = ['RSI', 'MACD', 'MACD_hist', 'Volume_ratio', 'Price_momentum_1d', 
            'Stoch_K', 'egarch_vol', 'VIX_Change', 'TNX_Change']

latest_date = data.index.max()
latest_data = data[data.index == latest_date].copy()
historical_data = data[data.index < latest_date].copy()

train_data = historical_data[historical_data.index < '2024-12-31']
val_data = historical_data[historical_data.index >= '2025-01-01']

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
# Now using continuous target for robust modeling
y_train = train_data['rank'].values 

X_val = scaler.transform(val_data[features])
y_val = val_data['rank'].values

X_latest_alpha = scaler.transform(latest_data[features])

# ===============================================================================================================
# --- XGBoost Regressor & TimeSeries CV Optimization ---

print("\nStarting XGBoost Continuous hyperparameter optimization (Time Series Split)...")

def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'reg:squarederror', # Changed to Regression
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    mse_scores = []
    
    for train_idx, val_idx in tscv.split(X_train):
        X_tr, X_v = X_train[train_idx], X_train[val_idx]
        y_tr, y_v = y_train[train_idx], y_train[val_idx]
        
        model = xgb.XGBRegressor(**param)
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        
        preds = model.predict(X_v)
        mse_scores.append(mean_squared_error(y_v, preds))
        
    return np.mean(mse_scores)

# Minimizing MSE
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

best_params = study.best_params
best_params['objective'] = 'reg:squarederror'
best_params['random_state'] = 42
print(f"\nBest architecture found: {best_params}")

# ===============================================================================================================
# === WALK-FORWARD ENGINE & TRADING LOG SETUP ===
# ===============================================================================================================

print("\nStarting Walk-Forward Backtest (Expanding Window + Transaction Costs)...")

oos_dates = val_data.index.unique().sort_values()
portfolio_daily_returns = []
current_weights = pd.Series(0.0, index=valid_etfs) 
trade_log = [] 

ALPHA_POWER = 2.0  
RISK_FLOOR = 0.05  

# Mock Liquidity Parameters based on transaction_costs.py
LIQUIDITY_PROFILE = {t: 5.0 for t in valid_etfs} # Base 5bps spread
K_IMPACT = 0.05

# Dynamic Regressor
dynamic_model = xgb.XGBRegressor(**best_params)
dynamic_model.fit(X_train, y_train)

for i, current_date in enumerate(oos_dates):
    cost_decimal = 0.0
    
    if i % 21 == 0:
        past_mask = historical_data.index <= current_date
        past_window_data = historical_data.loc[past_mask]
        
        X_available = scaler.transform(past_window_data[features])
        y_available = past_window_data['rank'].values
        
        dynamic_model.fit(X_available, y_available)
        temp_efficiency = {}
        
        for ticker in valid_etfs:
            idx_mask = (val_data.index == current_date) & (val_data['ticker'] == ticker)
            row_indices = np.where(idx_mask)[0]
            ticker_specific_feat = X_val[row_indices]
            
            if len(ticker_specific_feat) > 0:
                # Predicting continuous rank and ensuring positive values for allocation
                ai_alpha = float(dynamic_model.predict(ticker_specific_feat)[0])
                ai_alpha = np.clip(ai_alpha, 0.01, 0.99)
            else:
                ai_alpha = 0.50
                
            past_returns = clean_ret_log[ticker].loc[:current_date].iloc[:-1].tail(252)
            neg_rets = past_returns[past_returns < 0]
            
            raw_sigma = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 1 else RISK_FLOOR
            adjusted_sigma = max(raw_sigma, RISK_FLOOR)
            
            temp_efficiency[ticker] = (ai_alpha ** ALPHA_POWER) / (adjusted_sigma + 1e-6)
            
        new_weights = pd.Series(temp_efficiency)
        if new_weights.sum() > 0:
            new_weights = new_weights / new_weights.sum()
            new_weights = new_weights.clip(upper=0.40) 
            new_weights /= new_weights.sum()
        else:
            new_weights = pd.Series(0.0, index=valid_etfs)
            
        # Cost application on Rebalance Execution
        cost_decimal = apply_transaction_costs(
            weights_prev=current_weights.to_dict(),
            weights_new=new_weights.to_dict(),
            liquidity_profile=LIQUIDITY_PROFILE,
            k_impact=K_IMPACT
        )
            
        for ticker in valid_etfs:
            peso_anterior = current_weights.get(ticker, 0.0)
            peso_novo = new_weights.get(ticker, 0.0)
            delta = peso_novo - peso_anterior
            
            if abs(delta) > 0.005:
                trade_log.append({
                    'Data': current_date,
                    'Ticker': ticker,
                    'Operação': 'COMPRA 🟢' if delta > 0 else 'VENDA 🔴',
                    'Ordem (% Fundo)': delta * 100,      
                    'Peso Alocado (%)': peso_novo * 100  
                })
                
        current_weights = new_weights.copy()
        
    if current_date in clean_ret_log.index:
        # Calculate daily raw return and subtract Transaction Friction
        daily_ret = clean_ret_log.loc[current_date, current_weights.index].mul(current_weights).sum()
        daily_ret -= cost_decimal 
        portfolio_daily_returns.append({'Data': current_date, 'Return': daily_ret})

# --- KPI Section Execution ---
df_port_rets = pd.DataFrame(portfolio_daily_returns)
if not df_port_rets.empty:
    df_port_rets.set_index('Data', inplace=True)
    portfolio_returns = df_port_rets['Return']
else:
    portfolio_returns = pd.Series(dtype=float)

start_dt = portfolio_returns.index.min()
end_dt = portfolio_returns.index.max()
bench_data = yf.download('IVV', start=start_dt, end=end_dt, auto_adjust=True, progress=False)['Close']
if isinstance(bench_data, pd.DataFrame): bench_data = bench_data.iloc[:, 0]
bench_rets = np.log(bench_data / bench_data.shift(1)).dropna()

# Print Summary
print("\n" + "="*80)
print(" === INSTITUTIONAL MODEL EVALUATION (Friction & Continuous Objective) ===")
print("="*80)
if not portfolio_returns.empty:
    ann_ret = (1 + float(np.exp(portfolio_returns.sum()) - 1)) ** (252 / len(portfolio_returns)) - 1
    bench_ann_ret = (1 + float(np.exp(bench_rets.sum()) - 1)) ** (252 / len(bench_rets)) - 1
    alpha = float(ann_ret - bench_ann_ret)
    print(f"Annualized Return (Net of Costs): {ann_ret*100:8.2f}%")
    print(f"Alpha (vs S&P 500 Benchmark):     {alpha*100:8.2f}%")
print("="*80 + "\n")

# ===============================================================================
# === 5. VISUALIZATION: PORTFOLIO VS BENCHMARK ===
# ===============================================================================

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt

print("\nGenerating Cumulative Performance...")

if not portfolio_returns.empty and not bench_rets.empty:
    # Same date from Portfolio and Benchmark
    bench_rets_aligned = bench_rets.reindex(portfolio_returns.index).fillna(0)
    
    # Real Cumulative Return: Log returns back into simple
    port_cum_ret = np.exp(portfolio_returns.cumsum()) - 1
    bench_cum_ret = np.exp(bench_rets_aligned.cumsum()) - 1
    
    # Graph Settings
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Line's Plot
    ax.plot(port_cum_ret.index, port_cum_ret * 100, label='AI Quant Portfolio (Net of Costs)', color='#00ff00', linewidth=2.5)
    ax.plot(bench_cum_ret.index, bench_cum_ret * 100, label='S&P 500 (IVV)', color='#8b949e', linewidth=2, linestyle='--')

    # Formatting
    ax.set_title("Cumulative Performance: AI Portfolio vs S&P 500", fontsize=15, fontweight='bold', color='white', pad=15)
    ax.set_ylabel("Liquid Performance Return (%)", fontsize=12, color='white')
    ax.set_xlabel("Data", fontsize=12, color='white')
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 6. Grid e legends
    ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', frameon=False, labelcolor='white', fontsize=11)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#30363d')
        
    plt.tight_layout()
    plt.show()
else:
    print("No real data for performance.")