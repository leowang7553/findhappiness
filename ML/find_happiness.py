import numpy as np
import pandas as pd
import xgboost as xgb
from utils import load_model
from feature_engineering import data_forest_select
from utils import hour_cut, birth_split, income_cut, house_cut, floor_area_cut

train, test, X_train_, y_train_, X_test_, id_test_ = data_forest_select()

X_train = np.array(X_train_)
X_test  = np.array(X_test_)
y_train = np.array(y_train_)
id_test = np.array(id_test_)

'''
'birth_s', 'BMI', 'health', 'marital', 'religion', 'depression', 'relax', 'class', 'status_3_before', 'class_10_after', 'income_cut',
                    'inc_exp_cut', 'inc_ability', 'house_cut', 'floor_area_cut', 'family_m', 'floor_area_avg_cut', 'family_income_cut', 'family_status', 'social_neighbor', 'social_friend', 'equity'
'''
input_data = {
    'birth': [1989],
    'height_cm': [183],
    'weight_jin': [155],
    'health': [4],
    'marital': [3],
    'religion': [0],
    'depression': [5],
    'relax': [3],
    'class': [6],
    'status_3_before': [1],
    'class_10_after': [8],
    'income': [400000],
    'inc_exp': [5000000],
    'inc_ability': [3],
    'house': [1],
    'floor_area': [60],
    'family_m': [2],
    'family_income': [1000000],
    'family_status': [6],
    'social_neighbor': [7],
    'social_friend': [5],
    'equity': [4]
}

input_data_df = pd.DataFrame(input_data)

# 增加特征
input_data_df['birth_s'] = input_data_df['birth'].map(birth_split)  # 出生的年代
input_data_df['income_cut'] = input_data_df['income'].map(income_cut)  # 收入分组
input_data_df['family_income_cut'] = input_data_df['family_income'].map(income_cut)  # 收入分组
input_data_df['inc_exp_cut'] = input_data_df['inc_exp'].map(income_cut) # 期望收入分组
input_data_df['house_cut'] = input_data_df['house'].map(house_cut)  # 房产数分组
input_data_df['floor_area_cut'] = input_data_df['floor_area'].map(floor_area_cut)  # 住房面积分组
input_data_df['BMI'] = round((input_data_df['weight_jin']/2) / ((input_data_df['height_cm']/100)*(input_data_df['height_cm']/100)))  # 体重指数BMI
input_data_df['floor_area_avg'] = round(input_data_df['floor_area'] / input_data_df['family_m']) # 人均住宅面积
input_data_df['floor_area_avg_cut'] = input_data_df['floor_area_avg'].map(floor_area_cut)  # 人均住宅面积分组

# print(input_data_df)

X_data = input_data_df[['birth_s', 'BMI', 'health', 'marital', 'religion', 'depression', 'relax', 'class', 'status_3_before', 'class_10_after', 'income_cut',
                    'inc_exp_cut', 'inc_ability', 'house_cut', 'floor_area_cut', 'family_m', 'floor_area_avg_cut', 'family_income_cut', 'family_status', 'social_neighbor', 'social_friend', 'equity']]
X_data_ndarr = np.array(X_data)

clf_xgb = load_model('clf_xgb')
predictions = clf_xgb.predict(xgb.DMatrix(X_data_ndarr), ntree_limit=clf_xgb.best_ntree_limit)
predictions = predictions + 1

print(predictions)

# xgb.plot_importance(clf_xgb, max_num_features=20)
# print(clf_xgb.get_score(fmap='', importance_type='weight'))
# print(clf_xgb.feature_importances_)

# predictions_2d = np.array([predictions]).swapaxes(1, 0)
# id_test_2d = np.array([id_test]).swapaxes(1, 0)
# predictions_df = pd.DataFrame(predictions_2d, columns=['happiness'])
# # predictions_df = predictions_df.map(lambda x:x+1)
# id_test_df = pd.DataFrame(id_test_2d, columns=['id'])
# df = id_test_df.join(predictions_df)
# df['happiness'] = df['happiness'].map(lambda x:round(x))

# print(df.head())
# print(df.columns)

# 生成csv文件
# pd.DataFrame(df).to_csv('y_test_1.csv', index=False)
