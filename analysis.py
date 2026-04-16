import pandas as pd
import numpy as np
from model import get_embedding
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV


headlines = pd.read_csv('tariff_news_headlines.csv')
first_date = headlines['date'].min()


rates = pd.read_csv('tariff_rates.csv')
df = headlines.merge(rates, on = 'date', how='outer')

print(df.dtypes)

mkt = pd.read_csv('market_reaction.csv')
mkt_filtered = mkt[mkt['date'] > first_date]
returns = mkt_filtered['date']
price_cols = ['sp500', 'shanghai_composite', 'dxy', 'usd_cny', 'crude_oil_wti', 'steel_futures', 'aluminum_futures', 'soybeans']
print(mkt_filtered.dtypes)
returns[price_cols] = mkt_filtered[price_cols].pct_change()

df = df.merge(returns, on = 'date', how = 'inner')
