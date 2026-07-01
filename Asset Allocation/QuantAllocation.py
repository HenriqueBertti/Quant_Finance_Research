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
    'Tech': ['QQQ', 'XLK'],
    'Emergent': [''],
    'Strategic_Commodities': ['REMX'],

    
}

all_etfs = [item for sublist in etfs_dict.values() for item in sublist]

raw_data = yf.download(all_etfs, start='2014-01-01', end=FINAL_DATE, auto_adjust=True) 
close_df = raw_data['Close'].copy()
close_df = close_df.dropna(how='all')
volume_df = raw_data['Volume'].copy().ffill()

