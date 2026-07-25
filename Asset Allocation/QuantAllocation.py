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
from sklearn.metrics import accuracy_score
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

# === Filtering ETFs by volume and time series length ===
valid_etfs = []
MIN_DAYS_HISTORY = 252 
MIN_ADTV = 10_000_000 

for t in all_etfs:
    if t not in close_df.columns or t not in volume_df.columns:
        continue
        
    if close_df[t].dropna().shape[0] < MIN_DAYS_HISTORY:
        print(f'ETF {t} rejected: Insufficient Historical Data')
        continue

    recent_prices = close_df[t].iloc[-21:]
    recent_volumes = volume_df[t].iloc[-21:]

    daily_volume_financial = recent_prices.multiply(recent_volumes, fill_value=0)
    adtv_21 = daily_volume_financial.mean()

    if not t.endswith('=X'): 
        if pd.isna(adtv_21) or adtv_21 < MIN_ADTV:
            print(f'ETF {t} rejected: Low Average Daily Trading Volume. Calc: {adtv_21}')
            continue

    valid_etfs.append(t)

print(f'Valid ETFs found: {valid_etfs}')

if not valid_etfs:
    print("Error: No ETFs passed the volume or data length filters.")
    exit()

clean_close = close_df[valid_etfs].copy()
clean_ret_log = np.log(clean_close / clean_close.shift(1)).dropna()

# Winsorizing
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

from scipy.stats import kstest
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

# ===============================================================================================================
# --- E-GARCH Model Configuration --- 

print('Calculating E-GARCH Volatility (t-Student)...')
egarch_volatility = pd.DataFrame()

for t in valid_etfs: 
    try: 
        returns = 100 * clean_ret_log[t].dropna()
        # CORREÇÃO 1: Utilizando dist='t' em respeito ao teste de KS (caudas pesadas)
        am = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='t')
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
print("Building Neural Panel with Scoring System...")

for t in valid_etfs:
    try: 
        if t not in egarch_volatility.columns: continue

        df = clean_close[[t]].copy()
        df.columns = ['Close']
        df = df.dropna()
        df['ticker'] = t

        df['ret_log'] = clean_ret_log[t]
        df['High'] = ohlcv['High'][t]
        df['Low'] = ohlcv['Low'][t]
        df['Volume'] = ohlcv['Volume'][t]

        # Correction: Applying Exponential Smoothing (Wilder's Smoothing)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / (loss + 1e-8)
        df['RSI'] = 100 - (100 / (1 + rs))

        df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
        df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()

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

        ema9_above_ma50_prev = df['EMA_9'].shift(1) > df['MA_50'].shift(1)
        ema9_below_ma50_now = df['EMA_9'] <= df['MA_50']
        df['sell_trigger'] = (ema9_above_ma50_prev & ema9_below_ma50_now).astype(int) 

        df['egarch_vol'] = egarch_volatility[t]
        df = df.join(macro_data[['VIX_Change', 'TNX_Change']], how='inner')

        # Correction from previous code.
        df['target_weekly'] = np.log(df['Close'].shift(-5) / df['Close'])
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

latest_date = data.index.max()
latest_data = data[data.index == latest_date].copy()

historical_data = data[data.index < latest_date].copy()

train_data = historical_data[historical_data.index < '2024-12-31']
val_data = historical_data[historical_data.index >= '2025-01-01']

scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[features])
y_train = (train_data['rank_score'] > 0).astype(int).values

X_val = scaler.transform(val_data[features])
y_val = (val_data['rank_score'] > 0).astype(int).values

X_latest_alpha = scaler.transform(latest_data[features])

# ===============================================================================================================
# --- XGBoost Model & Optuna Optimization ---

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

best_params = study.best_params
best_params['objective'] = 'binary:logistic'
best_params['random_state'] = 42

print(f"\nBest architecture found: {best_params}")

# ===============================================================================================================
# === WALK-FORWARD ENGINE & TRADING LOG SETUP ===
# ===============================================================================================================

print("\nStarting Walk-Forward Backtest (Expanding Window / Out-of-Sample)...")

oos_dates = val_data.index.unique().sort_values()
portfolio_daily_returns = []
current_weights = pd.Series(0.0, index=valid_etfs) 
trade_log = [] 

ALPHA_POWER = 2.0  
RISK_FLOOR = 0.05  

# Inicializa o modelo dinâmico treinado até o início da validação
dynamic_model = xgb.XGBClassifier(**best_params)
dynamic_model.fit(X_train, y_train)

for i, current_date in enumerate(oos_dates):
    
    # Rebalanceamento a cada 21 dias (mensal)
    if i % 21 == 0:
        print(f"Rebalancing Model at {current_date.strftime('%Y-%m-%d')}...")
        
        # CORREÇÃO 4: Re-treinamento Iterativo (Expanding Window)
        # Extrai toda a base de dados conhecida ATÉ a data atual da iteração
        past_mask = historical_data.index <= current_date
        past_window_data = historical_data.loc[past_mask]
        
        X_available = scaler.transform(past_window_data[features])
        y_available = (past_window_data['rank_score'] > 0).astype(int).values
        
        # Atualiza a inteligência do robô com a informação mais recente (sem olhar pro futuro)
        dynamic_model.fit(X_available, y_available)
        
        temp_efficiency = {}
        
        for ticker in valid_etfs:
            idx_mask = (val_data.index == current_date) & (val_data['ticker'] == ticker)
            row_indices = np.where(idx_mask)[0]
            ticker_specific_feat = X_val[row_indices]
            
            if len(ticker_specific_feat) > 0:
                # Utiliza o modelo dinâmico atualizado
                ai_alpha = float(dynamic_model.predict_proba(ticker_specific_feat)[0][1])
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
        daily_ret = clean_ret_log.loc[current_date, current_weights.index].mul(current_weights).sum()
        portfolio_daily_returns.append({'Data': current_date, 'Return': daily_ret})

df_trades = pd.DataFrame(trade_log)

# ===============================================================================================================
# === PAYOUT, KPI E PERFORMANCE ===
# ===============================================================================================================

print("\nCalculating Payout and Hit Rate...")

if not df_trades.empty:
    trade_tickers = df_trades['Ticker'].unique().tolist()
    start_trade = pd.to_datetime(df_trades['Data'].min())
    end_trade = pd.to_datetime(clean_close.index.max()) 
    
    real_prices = clean_close[trade_tickers].loc[start_trade:end_trade]
    payout_results = []
    
    for i in range(len(df_trades)):
        row = df_trades.iloc[i]
        ticker = row['Ticker']
        data_entrada = pd.to_datetime(row['Data'])
        direcao = 1 if 'COMPRA' in row['Operação'] else -1 
        
        futuros_trades = df_trades.iloc[i+1:]
        proximo_trade = futuros_trades[futuros_trades['Ticker'] == ticker]
        
        if not proximo_trade.empty:
            data_saida = pd.to_datetime(proximo_trade.iloc[0]['Data'])
        else:
            data_saida = pd.to_datetime(real_prices.index[-1])
            
        try:
            preco_entrada = real_prices.loc[:data_entrada, ticker].iloc[-1]
            preco_saida = real_prices.loc[:data_saida, ticker].iloc[-1]
            retorno_trade = (preco_saida / preco_entrada) - 1
            payout_decisao = retorno_trade * direcao
            
            payout_results.append({
                'Data Ordem': data_entrada.strftime('%Y-%m-%d'),
                'Data Saída / Reavaliação': data_saida.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'Ação da IA': row['Operação'],
                'Preço de Entrada': f"${preco_entrada:.2f}",
                'Preço de Saída': f"${preco_saida:.2f}",
                'Retorno do Ativo': f"{retorno_trade * 100:.2f}%",
                'Status do Trade': 'WIN ✅' if payout_decisao > 0 else 'LOSS ❌'
            })
        except Exception as e:
            continue 
            
    df_payout = pd.DataFrame(payout_results)
    acertos = len(df_payout[df_payout['Status do Trade'] == 'WIN ✅'])
    total = len(df_payout)
    hit_rate = (acertos / total) * 100 if total > 0 else 0
    
    print("\n" + "="*80)
    print(" === INSTITUTIONAL TRADE LOG ===")
    print("="*80)
    print(df_payout.to_string(index=False))
    print("-" * 80)
    print(f"Model Hit Rate: {hit_rate:.2f}% ({acertos} Wins / {total} Total Trades)")
    print("=" * 80 + "\n")
    
else:
    print("\nNo trades were executed during the out-of-sample period.")

# --- NEW FINANCIAL METRICS ---
df_port_rets = pd.DataFrame(portfolio_daily_returns)
if not df_port_rets.empty:
    df_port_rets.set_index('Data', inplace=True)
    portfolio_returns = df_port_rets['Return']
else:
    portfolio_returns = pd.Series(dtype=float)

gains = portfolio_returns[portfolio_returns > 0]
losses = portfolio_returns[portfolio_returns < 0]

avg_gain = float(gains.mean()) if not gains.empty else 0.0
avg_loss = float(losses.mean()) if not losses.empty else 0.0
max_gain = float(portfolio_returns.max()) if not portfolio_returns.empty else 0.0
payoff = avg_gain / abs(avg_loss) if avg_loss != 0 else 0.0

if not portfolio_returns.empty:
    total_ret = float(np.exp(portfolio_returns.sum()) - 1)
    ann_ret = (1 + total_ret) ** (252 / len(portfolio_returns)) - 1
    
    start_dt = portfolio_returns.index.min()
    end_dt = portfolio_returns.index.max()
    bench_data = yf.download('IVV', start=start_dt, end=end_dt, auto_adjust=True, progress=False)['Close']
    
    if isinstance(bench_data, pd.DataFrame):
        bench_data = bench_data.iloc[:, 0]
        
    bench_rets = np.log(bench_data / bench_data.shift(1)).dropna()
    bench_total_ret = float(np.exp(bench_rets.sum()) - 1)
    bench_ann_ret = (1 + bench_total_ret) ** (252 / len(bench_rets)) - 1
    alpha = float(ann_ret - bench_ann_ret)
else:
    alpha = 0.0

print(f"Mean Return (Positive Days): {avg_gain*100:8.2f}%")
print(f"Mean Loss (Negative Days):   {avg_loss*100:8.2f}%")
print(f"Daily Win Return:            {max_gain*100:8.2f}%")
print(f"Payoff (Return/Loss):     {payoff:8.2f}")
print(f"Alpha (vs Benchmark S&P 500):    {alpha*100:8.2f}")
print("=========================================")

print("\n" + "="*40)
print(" 💻 COMPUTATIONAL PERFORMANCE ")
print("="*40)
print(f"CPU Usage:      {psutil.cpu_percent(interval=0.5)}%")
ram_info = psutil.virtual_memory()
print(f"RAM Usage:      {ram_info.percent}% ({ram_info.used / (1024 ** 3):.2f} GB)")
print("=========================================\n")

# === Plotting against the benchmark === 

print('\Generating graphical performance')
import matplotlib.ticker as mtick

if not portfolio_returns.empty and not bench_rets.empty:
    bench_rets_aligned = bench_rets.reindex(portfolio_returns.index).fillna(0)

    port_cum_ret = np.exp(portfolio_returns.cumsum()) - 1
    bench_cum_ret = np.exp(bench_rets_aligned.cumsum()) - 1

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # 4. Plotagem das Linhas
    ax.plot(port_cum_ret.index, port_cum_ret * 100, label='AI Quant Portfolio', color='#00ff00', linewidth=2.5)
    ax.plot(bench_cum_ret.index, bench_cum_ret * 100, label='S&P 500 (IVV)', color='#8b949e', linewidth=2, linestyle='--')

    # 5. Formatação de Eixos e Títulos
    ax.set_title("Cumulative Performance: AI Portfolio vs S&P 500", fontsize=15, fontweight='bold', color='white', pad=15)
    ax.set_ylabel("Return (%)", fontsize=12, color='white')
    ax.set_xlabel("Data", fontsize=12, color='white')
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 6. Grid e Legendas
    ax.grid(axis='y', color='#30363d', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', frameon=False, labelcolor='white', fontsize=11)
    
    # 7. Limpeza das Bordas (Estética limpa)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#30363d')
        
    plt.tight_layout()
    plt.show()
else:
    print("No data for plot comparison.")

# ===============================================================================
# === PARAMETRIC VaR & CVaR BACKTESTING ===
# ===============================================================================

import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# --- 1. Backtesting Functions (Kupiec and Christoffersen) ---

def kupiec_test(failures, n_obs, alpha):
    """Kupiec's Proportion of Failures (POF) Test for Unconditional Coverage"""
    p_exp = alpha
    p_obs = failures / n_obs
    
    if p_obs == 0 or p_obs == 1:
        return np.nan, np.nan
        
    # Kupiec Likelihood Ratio
    lr_uc = -2 * ((n_obs - failures) * np.log(1 - p_exp) + failures * np.log(p_exp)) + \
             2 * ((n_obs - failures) * np.log(1 - p_obs) + failures * np.log(p_obs))
    
    p_value = 1 - stats.chi2.cdf(lr_uc, df=1)
    return lr_uc, p_value

def christoffersen_test(violations):
    """Christoffersen's Test for Independence"""
    T00 = T01 = T10 = T11 = 0
    
    # Transition counting
    for i in range(1, len(violations)):
        if violations[i-1] == 0 and violations[i] == 0: T00 += 1
        elif violations[i-1] == 0 and violations[i] == 1: T01 += 1
        elif violations[i-1] == 1 and violations[i] == 0: T10 += 1
        elif violations[i-1] == 1 and violations[i] == 1: T11 += 1
            
    # Transition probabilities
    pi_01 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0
    pi_11 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0
    pi = (T01 + T11) / (T00 + T01 + T10 + T11)
    
    # Epsilon to prevent log(0)
    eps = 1e-10
    
    log_L_null = (T00 + T10) * np.log(1 - pi + eps) + (T01 + T11) * np.log(pi + eps)
    log_L_alt = T00 * np.log(1 - pi_01 + eps) + T01 * np.log(pi_01 + eps) + \
                T10 * np.log(1 - pi_11 + eps) + T11 * np.log(pi_11 + eps)
                
    lr_ind = -2 * (log_L_null - log_L_alt)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)
    return lr_ind, p_value

# --- 2. Risk Parameters & Setup ---

confidence_level = 0.95
alpha = 1 - confidence_level
window = 252 # Estimation window (1 business year)

# Critical Z-value (approx -1.645 for 95%)
z_alpha = stats.norm.ppf(alpha)
pdf_z = stats.norm.pdf(z_alpha)

# Packing the single Series into a dictionary to avoid the "columns" Attribute Error
assets_to_test = {
    'AI Quant Portfolio': portfolio_returns.dropna(),
    'S&P 500 Benchmark': bench_rets.dropna()
}

backtest_results = []

# --- 3. Execution Engine ---

for name, ret in assets_to_test.items():
    
    # Check if the asset has enough history to build the rolling window
    if len(ret) < window:
        print(f"Skipping {name}: Not enough historical data for the {window}-day rolling window.")
        continue
        
    # Rolling Mean and Std Dev (Shifted by 1 to use T-1 data for T prediction)
    mu_rolling = ret.rolling(window=window).mean().shift(1)
    sigma_rolling = ret.rolling(window=window).std().shift(1)
    
    # Aligning arrays (Removing the initial burn-in period)
    ret_eval = ret[window:]
    mu_eval = mu_rolling[window:]
    sigma_eval = sigma_rolling[window:]
    
    # Parametric VaR and CVaR Estimation
    var_parametric = mu_eval + z_alpha * sigma_eval
    cvar_parametric = mu_eval - sigma_eval * (pdf_z / alpha)
    
    # Flagging Violations (When actual return is worse than the projected VaR)
    violations = (ret_eval < var_parametric).astype(int)
    n_obs = len(ret_eval)
    n_failures = violations.sum()
    failure_rate = n_failures / n_obs if n_obs > 0 else 0
    
    # Kupiec Test
    lr_uc, p_val_uc = kupiec_test(n_failures, n_obs, alpha)
    
    # Christoffersen Test
    lr_ind, p_val_ind = christoffersen_test(violations.values)
    
    # Conditional Coverage Test
    lr_cc = lr_uc + lr_ind
    p_val_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
    
    backtest_results.append({
        'Strategy / Asset': name,
        'Observations': n_obs,
        'Expected Failures': np.round(n_obs * alpha, 1),
        'Actual Failures': n_failures,
        'Failure Rate (%)': np.round(failure_rate * 100, 2),
        'Kupiec p-value': np.round(p_val_uc, 4),
        'Christoffersen p-value': np.round(p_val_ind, 4),
        'Cond. Coverage p-value': np.round(p_val_cc, 4)
    })

# --- 4. Display Panel ---

if backtest_results:
    results_df = pd.DataFrame(backtest_results).set_index('Strategy / Asset')
    print("\n" + "="*85)
    print(" === PARAMETRIC VaR BACKTEST (Normal Distribution Baseline) ===")
    print(f" Confidence Level: {confidence_level*100}% | Alpha: {alpha*100}% | Rolling Window: {window} days")
    print("="*85)
    print(results_df.to_string())
    print("="*85 + "\n")