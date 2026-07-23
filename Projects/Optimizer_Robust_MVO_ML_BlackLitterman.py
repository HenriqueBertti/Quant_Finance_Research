import pandas as pd 
import numpy as np
import yfinance as yf  
import warnings
import datetime

warnings.filterwarnings('ignore')
FINAL_DATE = datetime.datetime.now().strftime('2026-06-30')

# === Selecting Assets for Downloading and Cleaning Data ===

tickers = {
    'ETFs': ['REMX', 'VGT', 'XLK', 'QQQ'],
    'Stocks': ['NVDA', 'MSFT', 'GOOGL', 'VALE3.SA', 'ITUB4.SA'],
    'Commodities': ['GLD', 'BZ=F', 'SI=F']
}

all_tickers = [item for sublist in tickers.values() for item in sublist]

raw_data_close = yf.download(all_tickers, start='2010-01-01', end=FINAL_DATE)['Close']
df_close = raw_data_close.copy()
df_close.dropna(inplace=True)
df_close = df_close.rename(columns={
    'BZ=F': 'Brent Oil',
    'GLD': 'Gold',
    'SI=F': 'Silver',
    'GOOGL': 'Google',
    'ITUB4.SA': 'Itaú',
    'MSFT': 'Microsoft',
    'NVDA': 'NVIDIA',
    'VALE3.SA': 'Vale'
})
print(df_close)

# === Calculating the logarithmic returns and Statistical Descriptive Analysis === 

log_returns = np.log(df_close / df_close.shift(1))
log_returns.dropna(inplace=True)
log_returns

stats_df = pd.DataFrame({
    'Mean': log_returns.mean(),
    'Variance': log_returns.var(),
    'Standard Deviation': log_returns.std(),
    'Skewness': log_returns.skew(),
    'Kurtosis': log_returns.kurt()
})

print(stats_df)
correlation_matrix = log_returns.corr()
print(correlation_matrix)

# ===== Tests for Normality, Stationarity and Heteroscedasticity ====

print('Initiating Tests for Normality, Stacionarity and Heteroscedasticity...')

from scipy.stats import kstest, norm, probplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

test_results = []
for column in log_returns.columns:
    serie = log_returns[column]

    standard_series = (serie - serie.mean()) / serie.std()
    ks_stat, ks_p_value = kstest(standard_series, 'norm')
    adf_stat, adf_p_value, _, _, _, _ = adfuller(serie)
    arch_set, arch_p_value, _, _ = het_arch(serie)

    test_results.append({
        'Index': column,
        'KS p-value': ks_p_value,
        'ADF p-value': adf_p_value,
        'ARCH p-value': arch_p_value
    })

df_test_results = pd.DataFrame(test_results)
print('Test Results for Normality, Stationarity and Heteroscedasticity...')
print(df_test_results)

# Statistical Interpretation:
# KS < 0.05: Series not normal.
# ADF < 0.05: Series are stationary.
# ARCH-LM < 0.05: There is volatility clustering.

# === Ledoit-Wolf ===
from sklearn.covariance import ledoit_wolf

def calculation_robust_covariance(log_returns: pd.DataFrame) -> pd.DataFrame:
    # Applying Ledoit-Wolf to calculate the covariance matrix 
    NUM_DAYS = 252
    clean_log_returns = log_returns.dropna()
    X = clean_log_returns.values
    # Shrunked_cov and Shrinkage Covariance
    shrunked_cov, Shrinkage_cov = ledoit_wolf(X)

    # Annualyzing:
    annualyzed = shrunked_cov * NUM_DAYS

    robust_cov_df = pd.DataFrame(
        annualyzed,
        index= clean_log_returns.columns,
        columns= clean_log_returns.columns
    )

    return robust_cov_df












