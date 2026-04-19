import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from model import get_embedding

# ── 1. LOAD & AGGREGATE HEADLINES BY DATE ────────────────────────────────────
headlines = pd.read_csv('tariff_news_headlines.csv', parse_dates=['date'])
rates     = pd.read_csv('tariff_rates.csv', parse_dates=['date'])
mkt       = pd.read_csv('market_reaction.csv', parse_dates=['date'])

# Stack both headline sources vertically, group by date into lists
all_headlines = (
    pd.concat([headlines[['date', 'headline']], rates[['date', 'headline']]], axis=0)
    .groupby('date')['headline']
    .apply(list)
    .reset_index()
)
print(all_headlines.head(24))
print(all_headlines.info())
# ── 2. COMPUTE RETURNS, FILTER TO HEADLINE DATES ONLY ────────────────────────
price_cols = ['sp500', 'shanghai_composite', 'dxy', 'usd_cny',
              'crude_oil_wti', 'steel_futures', 'aluminum_futures', 'soybeans']

mkt = mkt.sort_values('date').drop_duplicates(subset='date')
returns = mkt[['date'] + price_cols].copy()
returns[price_cols] = mkt[price_cols].pct_change(fill_method=None)
for w in [1, 3, 5]:
    returns[f'sp500_fwd_{w}d'] = mkt['sp500'].shift(-w) / mkt['sp500'] - 1

# Inner join: only keep dates that have both headlines AND market data
df = pd.merge_asof(
    all_headlines, 
    returns, 
    on='date', 
    direction='forward', # If news hits Saturday, attach it to Monday's market data
    tolerance=pd.Timedelta(days=4)
)


df = df.dropna(subset=['sp500_fwd_1d']).sort_values('date').reset_index(drop=True)
print(df.head(16))
print(df.info())

# ── 3. EMBED ──────────────────────────────────────────────────────────────────
def embed_list(texts):
    vecs = np.stack([get_embedding(t) for t in texts])
    return vecs.mean(axis=0)


embeddings = np.stack(df['headline'].apply(embed_list).values)  # (n, 768)

# ── 4. PCA + RIDGE ────────────────────────────────────────────────────────────
target_col = 'sp500_fwd_1d'
print("Data shape:", embeddings.shape)
X = PCA(n_components=15).fit_transform(embeddings)
y = df[target_col].values

split = int(0.8 * len(df))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], cv=5)
model.fit(X_train, y_train)
print("Test R²:", model.score(X_test, y_test))