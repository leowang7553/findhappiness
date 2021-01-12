import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
from utils import hour_cut, birth_split, income_cut, house_cut, floor_area_cut
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# 文件路径
filePath = sys.path[0]

# 导入数据
train_abbr = pd.read_csv(filePath + os.sep + 'datasets' + os.sep + 'happiness_train_abbr.csv', encoding='ISO-8859-1') # 精简版训练集
train = pd.read_csv(filePath + os.sep + 'datasets' + os.sep + 'happiness_train_complete.csv', encoding='ISO-8859-1') # 完整版训练集
test_abbr = pd.read_csv(filePath + os.sep + 'datasets' + os.sep + 'happiness_test_abbr.csv', encoding='ISO-8859-1') # 精简版测试集
test = pd.read_csv(filePath + os.sep + 'datasets' + os.sep + 'happiness_test_complete.csv', encoding='ISO-8859-1') # 完整版测试集
test_sub = pd.read_csv(filePath + os.sep + 'datasets' + os.sep + 'happiness_submit.csv', encoding='ISO-8859-1') # 测试集（结果）

# 观察数据大小
# print('test shape: ', test.shape)
# print('test_abbr shape: ', test_abbr.shape)
# print('test_sub shape: ', test_sub.head())
# print('train shape: ', train.shape)
# print('train_abbr shape: ', train_abbr.shape)

# 简单查看数据
# train.head()

# 查看数据是否缺失
# train.info(verbose=True,null_counts=True)

def data_process():
    # 查看label分布
    y_train_ = train['happiness']
    id_test_ = test_sub['id']

    # 处理y_train中的异常值：将-8换成3
    y_train_ = y_train_.map(lambda x:3 if x==-8 else x)

    # 处理y_train中的数据：让label从0开始, happiness:0~4
    y_train_ = y_train_.map(lambda x:x-1)

    # 数据拼接：train和test连在一起
    data = pd.concat([train,test],axis=0,ignore_index=True, sort=True)

    # 特征异常值处理
    data['join_party'] = data['join_party'].map(lambda x:0 if pd.isnull(x)  else 1) # 是否入党
    data['health'] = data['health'].map(lambda x:4 if x==-8 else x)  # 特征health异常值处理，取众数4
    data['depression'] = data['depression'].map(lambda x:4 if x==-8 else x)  # 特征depression异常值处理，取众数4
    data['relax'] = data['relax'].map(lambda x:4 if x==-8 else x)  # 特征relax异常值处理，取众数4
    data['class'] = data['class'].map(lambda x:5 if x==-8 else x)  # 特征class异常值处理，取众数5
    data['status_3_before'] = data['status_3_before'].map(lambda x:2 if x==-8 else x)  # 特征status_3_before异常值处理，取众数2
    data['class_10_after'] = data['class_10_after'].map(lambda x:5 if x==-8 else x)  # 特征class_10_after异常值处理，取众数5
    data['inc_ability'] = data['inc_ability'].map(lambda x:2 if x==-8 else x)  # 特征class_10_after异常值处理，取众数2
    data['family_status'] = data['family_status'].map(lambda x:3 if x==-8 else x)  # 特征family_status异常值处理，取众数3
    data['equity'] = data['equity'].map(lambda x:4 if x==-8 else x)  # 特征family_status异常值处理，取众数4
    data['social_neighbor'] = data['social_neighbor'].map(lambda x:2 if x==-8 else x)  # 特征family_status异常值处理，取众数2
    data['social_friend'] = data['social_friend'].map(lambda x:3 if x==-8 else x)  # 特征family_status异常值处理，取众数3
    data['family_m'] = data['family_m'].map(lambda x:1 if x <= 0 else x)  # 特征family_status异常值处理

    # data数据处理：处理调查时间的时间特征
    data['survey_time'] = pd.to_datetime(data['survey_time'],format='%Y-%m-%d %H:%M:%S')
    data['weekday'] = data['survey_time'].dt.weekday
    data['year'] = data['survey_time'].dt.year
    data['quarter'] = data['survey_time'].dt.quarter
    data['hour'] = data['survey_time'].dt.hour
    data['month'] = data['survey_time'].dt.month
    data['hour_cut'] = data['hour'].map(hour_cut)

    # 增加特征
    data['survey_age'] = data['year']-data['birth']  # 做问卷时候的年龄
    data['birth_s'] = data['birth'].map(birth_split)  # 出生的年代
    data['income_cut'] = data['income'].map(income_cut)  # 收入分组
    data['family_income_cut'] = data['family_income'].map(income_cut)  # 收入分组
    data['inc_exp_cut'] = data['inc_exp'].map(income_cut) # 期望收入分组
    data['house_cut'] = data['house'].map(house_cut)  # 房产数分组
    data['floor_area_cut'] = data['floor_area'].map(floor_area_cut)  # 住房面积分组
    data['BMI'] = round((data['weight_jin']/2) / ((data['height_cm']/100)*(data['height_cm']/100)))  # 体重指数BMI
    data['floor_area_avg'] = round(data['floor_area'] / data['family_m']) # 人均住宅面积
    data['floor_area_avg_cut'] = data['floor_area_avg'].map(floor_area_cut)  # 人均住宅面积分组

    #填充数据
    data['edu_status'] = data['edu_status'].fillna(5)
    data['edu_yr'] = data['edu_yr'].fillna(-2)
    data['property_other'] = data['property_other'].map(lambda x:0 if pd.isnull(x)  else 1)
    data['hukou_loc'] = data['hukou_loc'].fillna(1)
    data['social_neighbor'] = data['social_neighbor'].fillna(8)
    data['social_friend'] = data['social_friend'].fillna(8)
    data['work_status'] = data['work_status'].fillna(0)
    data['work_yr'] = data['work_yr'].fillna(0)
    data['work_type'] = data['work_type'].fillna(0)
    data['work_manage'] = data['work_manage'].fillna(0)
    data['family_income'] = data['family_income'].fillna(-2)
    data['invest_other'] = data['invest_other'].map(lambda x:0 if pd.isnull(x)  else 1)

    #填充数据
    data['minor_child'] = data['minor_child'].fillna(0)
    data['marital_1st'] = data['marital_1st'].fillna(0)
    data['s_birth'] = data['s_birth'].fillna(0)
    data['marital_now'] = data['marital_now'].fillna(0)
    data['s_edu'] = data['s_edu'].fillna(0)
    data['s_political'] = data['s_political'].fillna(0)
    data['s_hukou'] = data['s_hukou'].fillna(0)
    data['s_income'] = data['s_income'].fillna(0)
    data['s_work_exper'] = data['s_work_exper'].fillna(0)
    data['s_work_status'] = data['s_work_status'].fillna(0)
    data['s_work_type'] = data['s_work_type'].fillna(0)

    # 让label从0开始
    data['happiness'] = data['happiness'].map(lambda x:x-1)

    # 去掉三个缺失值很多的
    data = data.drop(['edu_other'], axis=1)
    data = data.drop(['happiness'], axis=1)
    data = data.drop(['survey_time'], axis=1)

    # 删除id标识
    data = data.drop(['id'], axis=1)

    # 划分训练集与测试集
    X_train_ = data[:train.shape[0]]
    X_test_  = data[train.shape[0]:]

    # target_column = 'happiness'
    # feature_columns = list(X_test_.columns) 

    # X_train = np.array(X_train_)
    # X_test  = np.array(X_test_)
    # y_train = np.array(y_train_)
    
    return train, test, X_train_, y_train_, X_test_, id_test_



# 生成csv文件
# pd.DataFrame(X_train).reset_index().to_csv('X_train.csv', index=False)