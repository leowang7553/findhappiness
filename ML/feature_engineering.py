import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

from data_preprocessing import data_process

train, test, X_train_, y_train_, X_test_, y_test_, id_test_ = data_process()

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

def data_forest_select():
    X_train_f = X_train_[['equity', 'depression', 'floor_area', 'county', 'city', 'family_income', 'weight_jin', 'class', 'class_10_after', 'survey_age',
                    'height_cm', 'birth', 'income', 'marital_1st', 'inc_exp', 'hour', 'public_service_7', 'province', 'public_service_6', 's_birth']]
    X_test_f = X_test_[['equity', 'depression', 'floor_area', 'county', 'city', 'family_income', 'weight_jin', 'class', 'class_10_after', 'survey_age',
                'height_cm', 'birth', 'income', 'marital_1st', 'inc_exp', 'hour', 'public_service_7', 'province', 'public_service_6', 's_birth']]
    return train, test, X_train_f, y_train_, X_test_f, y_test_, id_test_


'''
 1) equity                         0.023727
 2) depression                     0.018445
 3) floor_area                     0.017662
 4) county                         0.017301
 5) city                           0.015866
 6) family_income                  0.015782
 7) weight_jin                     0.015636
 8) class                          0.015000
 9) class_10_after                 0.014890
10) survey_age                     0.014688
11) height_cm                      0.014664
12) birth                          0.014658
13) income                         0.014310
14) marital_1st                    0.013884
15) inc_exp                        0.013140
16) hour                           0.012959
17) public_service_7               0.012826
18) province                       0.012638
19) public_service_6               0.012621
20) s_birth                        0.012369
'''