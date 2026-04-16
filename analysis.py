import pandas as pd
import numpy as np
import csv
from model import get_embedding

headlines = pd.read_csv('tariff_news_headlines.csv')
first_date = headlines['date'].min()


rates = pd.read_csv('tariff_rates.csv')
df = headlines.merge(rates, how='left')


mkt = pd.read_csv('market_reaction.csv')
mkt_filtered = mkt[mkt['date'] > first_date]
returns = mkt_filtered['date']
price_cols = ['date', 'sp500', 'shanghai_composite', 'dxy', 'usd_cny', 'crude_oil_wti', 'steel_futures', 'aluminum_futures', 'soybeans']
returns[price_cols] = mkt_filtered[price_cols].pct_change()

