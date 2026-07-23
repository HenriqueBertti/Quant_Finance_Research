import pandas as pd 
import numpy as np 
import Optimizer_Robust_MVO_ML_BlackLitterman

def calculating_implied_returns(delta: float, robust_cov: pd.DataFrame, market_weights: pd.Series) -> pd.Series:
    # Calculating the implied returns

    #Parameters:
    #delta (float): Coefficient of aversion to the risk market.
    #robust_cov (pd.DataFrame): Robust Covariance Matrix (NxN).
    #arket_weights (pd.Series): Capitalization weights of the market (N).
    
    #Return:
    #pd.Series: Vector with the implied returns (Pi).
    
    tickers = robust_cov.index
    weights_alligned = market_weights.loc[tickers].values

    sigma = robust_cov.values

    implied_returns = delta * np.dot(sigma, weights_alligned)

    return pd.Series(implied_returns, index=tickers, name = 'Implied Returns')