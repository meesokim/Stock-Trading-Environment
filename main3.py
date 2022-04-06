import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3 import A2C

from env.StockTradingEnv import StockTradingEnv

import pandas as pd
import FinanceDataReader as fdr
import torch

# df = pd.read_csv('./data/AAPL.csv')
from datetime import datetime as dt, timedelta
import random, os, sys

KRX_FILE = 'krx.csv'
def get_krx():
    if os.path.exists(KRX_FILE):
        krx = pd.read_csv(KRX_FILE, index_col=0)
    else:
        krx = fdr.StockListing('KRX')    
        krx.to_csv(KRX_FILE)
    return krx
    
def get_ticker(name):
    krx = get_krx()    
    result = krx.Symbol[krx.Name==name]
    if len(result) > 0:
        ticker = result.values[0]
    else:
        ticker = name
    return ticker

def get_random_ticker():
    krx = get_krx()
    return random.choice(krx.Symbol[~krx.ListingDate.isnull()].values)

if __name__ == '__main__':
    
    sdays = random.randint(365,365*3)
    start_date = dt.now() - timedelta(days=sdays)
    end_date = dt.today()
    duration = (end_date - start_date).days
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    if len(sys.argv) > 1:
        ticker = get_ticker(sys.argv[1])
    else:
        ticker = get_random_ticker()    
    print(ticker, start, end)
    df = ""
    while len(df) < duration / 2:
        df = fdr.DataReader(ticker, start, end).reset_index()
        print(df)
        import os, sys
        if len(df) < 10:
            sys.exit()
    df = df.sort_values('Date')

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df, None)])
    if os.path.exists(f'{ticker}_a2c_stock.zip'):
        model = A2C.load(f'{ticker}_a2c_stock')
    else:
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        model.save(f'{ticker}_a2c_stock')

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            break
        env.render()

    torch.save(model.policy.state_dict(), "test.pt")
