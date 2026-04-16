import pandas as pd
import numpy as np
import csv
from model import get_embedding
from sklearn.decomposition import PCA


headlines = pd.read_csv('tariff_news_headlines.csv')
first_date = headlines['date'].min()


rates = pd.read_csv('tariff_rates.csv')
df = headlines.merge(rates, how='left')


mkt = pd.read_csv('market_reaction.csv')
mkt_filtered = mkt[mkt['date'] > first_date]
returns = mkt_filtered['date']
price_cols = ['date', 'sp500', 'shanghai_composite', 'dxy', 'usd_cny', 'crude_oil_wti', 'steel_futures', 'aluminum_futures', 'soybeans']
returns[price_cols] = mkt_filtered[price_cols].pct_change()

print(df.dtypes)
"""
df = pd.get_dummies(df, columns=['product_category', 'tariff_level'], drop_first=True)
embeddings = np.stack([get_embedding(text) for text in df['headline']])
pca = PCA(n_components=15)
embeddings_reduced = pca.fit_transform(embeddings)

structured_cols = [c for c in df.columns if c not in ['date', 'headline', 'sp500']]
X_structured = df[structured_cols].values  # (n_events, n_structured)

X = np.hstack([embeddings_reduced, X_structured])  # (n_events, 15 + n_structured)
y = df['sp500'].values
"""