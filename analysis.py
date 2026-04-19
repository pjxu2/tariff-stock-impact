import pandas as pd
import numpy as np
from model import get_embedding
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## LOADING DATA
headlines = pd.read_csv('tariff_news_headlines.csv', parse_dates=['date'])
rates = pd.read_csv('tariff_rates.csv', parse_dates=['date'])
first_date = headlines['date'].min()
mkt = pd.read_csv('market_reaction.csv', parse_dates=['date'])
mkt = mkt.sort_values('date').drop_duplicates(subset='date')
df = headlines.merge(rates, on = 'date', how='outer')


## COMPUTE RETURNS
price_cols = ['sp500', 'shanghai_composite', 'dxy', 'usd_cny', 'crude_oil_wti', 'steel_futures', 'aluminum_futures', 'soybeans']
returns = mkt[['date'] + price_cols].copy()
returns[price_cols] = mkt[price_cols].pct_change()

for window in [1, 3, 5]:
    returns[f'sp500_fwd_{window}d'] = mkt['sp500'].shift(-window) / mkt['sp500'] - 1

mkt = mkt[mkt['date'] >= first_date].copy()

returns = returns[returns['date'] >= first_date].copy()

df = df.merge(returns, on = 'date', how = 'inner')



## Handle NaNs
target_col = 'sp500_fwd_3d'
df = df.dropna(subset=[target_col])
df['tariff_rate_pct'] = df['tariff_rate_pct'].fillna(0)
df['country'] = df['country'].fillna('unknown')
df = df.sort_values('date').reset_index(drop=True)

print(df.columns)
print(df.head(10)[['date', 'headline_x', 'headline_y']])
'''
## Handle categorical variables with one-hot encoding, as well as embeddings with finbert
df = pd.get_dummies(df, columns=['product_category', 'country'], drop_first=False)
embeddings = np.stack([get_embedding(text) for text in df['headline_x']])

pca = PCA(n_components=15)
embeddings_reduced = pca.fit_transform(embeddings)


print(df.columns)

exclude = ['date', 'headline', target_col, 'sp500_fwd_1d', 'sp500_fwd_5d']
structured_cols = [c for c in df.columns if c not in exclude]
X_structured = df[structured_cols].values 




## Final training values
X = np.hstack([embeddings_reduced, X_structured])  # (n_events, 15 + n_structured)
y = df['sp500'].values
split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))
'''