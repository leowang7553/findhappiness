import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

from data_preprocessing import data_process

train, test, X_train_, y_train_, X_test_, id_test_ = data_process()

def forest_evaluation():
    feat_labels = X_train_.columns

    # 用均值填补空缺值
    imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imr = imr.fit(X_train_.values)
    imputed_data = imr.transform(X_train_.values)
    X_train_1 = pd.DataFrame(imputed_data)
    X_train_1.columns = feat_labels

    # print(X_train_1.info(verbose=True,null_counts=True))
    # print(X_train_['income_cut'].value_counts())
    # print(type(X_train_1))

    # 用随机森林评估特征重要性
    forest = RandomForestClassifier(n_estimators=500, random_state=1)
    forest.fit(X_train_1,y_train_)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 打印特征重要性
    for f in range(X_train_.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

'''
desc: 随机森林评估特征重要性，选取前20个
'''
def data_forest_select():
    X_train_f = X_train_[['equity', 'depression', 'floor_area', 'county', 'city', 'family_income', 'weight_jin', 'class', 'class_10_after', 'survey_age',
                    'height_cm', 'birth', 'income', 'marital_1st', 'inc_exp', 'hour', 'public_service_7', 'province', 'public_service_6', 's_birth']]
    X_test_f = X_test_[['equity', 'depression', 'floor_area', 'county', 'city', 'family_income', 'weight_jin', 'class', 'class_10_after', 'survey_age',
                'height_cm', 'birth', 'income', 'marital_1st', 'inc_exp', 'hour', 'public_service_7', 'province', 'public_service_6', 's_birth']]
    return train, test, X_train_f, y_train_, X_test_f, id_test_

'''
desc: 结合算法评估的特征重要性与人工筛选
'''
def data_select():
    X_train_f = X_train_[['birth_s', 'BMI', 'health', 'marital', 'religion', 'depression', 'relax', 'class', 'status_3_before', 'class_10_after', 'income_cut',
                    'inc_exp_cut', 'inc_ability', 'house_cut', 'floor_area_cut', 'family_m', 'floor_area_avg_cut', 'family_income_cut', 'family_status', 'social_neighbor', 'social_friend', 'equity']]
    X_test_f = X_test_[['birth_s', 'BMI', 'health', 'marital', 'religion', 'depression', 'relax', 'class', 'status_3_before', 'class_10_after', 'income_cut',
                    'inc_exp_cut', 'inc_ability', 'house_cut', 'floor_area_cut', 'family_m', 'floor_area_avg_cut', 'family_income_cut', 'family_status', 'social_neighbor', 'social_friend', 'equity']]
    # print(X_train_f.info(verbose=True, null_counts=True))
    # print(X_train_f['depression'].value_counts())
    # print(X_train_f['social_friend'].min())
    # print(X_train_f.max())
    return train, test, X_train_f, y_train_, X_test_f, id_test_

# data_select()

'''
desc: 算法+人工筛选
- birth   birth_s替换
- height_cm  BMI替换
- weight_jin  BMI替换
- health
- marital
- religion
- depression
- relax
- class
- status_3_before
- class_10_after
- income  income_cut替换
- inc_exp  inc_exp_cut替换
- inc_ability
- house  house_cut替换
- floor_area floor_area_cut替换
- family_m
- family_income  family_income_cut替换
- family_status
- social_neighbor
- social_friend
- equity
'''