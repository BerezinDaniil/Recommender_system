import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
from sklearn.linear_model import Ridge

import os
import joblib
# 
UPLOAD_FOLDER = 'data/'


# load_ridge_models

directory_linear = 'linear_models'
linear_models = {}
for i in range(1, 101):
    path = directory_linear + '/' + f'model_{i}.pkl'
    model = joblib.load(path)
    linear_models[f'model_{i}'] = model
print("Done!")

# load_catboost_models
directory_catboost = 'models'
models = {}
for i in range(1, 101):
    path = directory_catboost + '/' + f'model_{i}.cbm'
    model = CatBoostRegressor()
    model.load_model(path)
    models[f'model_{i}'] = model
print("Done!")
# Данные для моделей
df = pd.read_csv(f'{UPLOAD_FOLDER}train_joke_df.csv')
df_f = pd.read_csv(f'{UPLOAD_FOLDER}data.csv')

test = pd.read_csv(f'{UPLOAD_FOLDER}input.csv')
print('Done!')

arr = test['UID']

# Предсказание
BEST = []
TOP_10 = []
for uid in arr:
    rating = {}
    for jid in range(1, 101):
        if np.sum((df['UID'] == uid) & (df['JID'] == jid)):
            rating_jid = np.array(df[(df['UID'] == uid) & (df['JID'] == jid)]['Rating'])[0]
        else:
            X_test_f = df_f[df_f['UID'] == jid].drop(columns=[f'{jid}', 'UID'])
            X_test_f['ridge_model_pred'] = linear_models[f'model_{jid}'].predict(np.array(X_test_f))
            rating_jid = models[f'model_{jid}'].predict(X_test_f)[0]
        rating[f'{jid}'] = rating_jid
    sorted_rating = sorted(rating.items(), key=lambda kv: -kv[1])
    best = {sorted_rating[0][0]: sorted_rating[0][1]}
    top_10 = [i[0] for i in sorted_rating[1:10]]
    BEST.append(best)
    TOP_10.append(top_10)

ans = pd.DataFrame({'best': BEST, 'top_10': TOP_10})
ans.to_csv(f'{UPLOAD_FOLDER}output.csv', index=False, sep='\t')
