import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, RepeatedKFold
from scipy import sparse
from data_preprocessing import data_process
from feature_engineering import data_forest_select
from utils import myFeval

## get data
# train, test, X_train_, y_train_, X_test_, y_test_, id_test_ = data_process()
train, test, X_train_, y_train_, X_test_, y_test_, id_test_ = data_forest_select()

X_train = np.array(X_train_)
X_test  = np.array(X_test_)
y_train = np.array(y_train_)
y_test = np.array(y_test_)
id_test = np.array(id_test_)

# print(type(y_train_))

## xgb
def xgb_model():
    xgb_params = {"booster":'gbtree','eta': 0.005, 'max_depth': 5, 'subsample': 0.7, 
                'colsample_bytree': 0.8, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': True, 'nthread': 8}
    folds = KFold(n_splits=5, shuffle=True, random_state=2018)
    oof_xgb = np.zeros(len(train))
    predictions_xgb = np.zeros(len(test))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("fold n°{}".format(fold_+1))
        trn_data = xgb.DMatrix(X_train[trn_idx], y_train[trn_idx])
        val_data = xgb.DMatrix(X_train[val_idx], y_train[val_idx])
        
        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]
        clf = xgb.train(dtrain=trn_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=100, params=xgb_params, feval=myFeval)
        oof_xgb[val_idx] = clf.predict(xgb.DMatrix(X_train[val_idx]), ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
        
    print("CV score: {:<8.8f}".format(mean_squared_error(oof_xgb, y_train_)))
    print('predictions_xgb: ', predictions_xgb)
    return predictions_xgb

predictions = xgb_model()
predictions = predictions + 1
predictions_2d = np.array([predictions]).swapaxes(1, 0)
id_test_2d = np.array([id_test]).swapaxes(1, 0)
predictions_df = pd.DataFrame(predictions_2d, columns=['happiness'])
# predictions_df = predictions_df.map(lambda x:x+1)
id_test_df = pd.DataFrame(id_test_2d, columns=['id'])
df = id_test_df.join(predictions_df)

print(df.head())
print(df.columns)

# y_test = y_test.map(lambda x:x+1)
# print(y_test)
# print(y_test + 1)

# 生成csv文件
pd.DataFrame(df).to_csv('y_test.csv', index=False)

# all features：CV score: 0.45451567
# forest feature selection ：CV score: 0.48657298