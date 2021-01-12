'''
公共方法
'''
from sklearn.metrics import mean_squared_error
import joblib
import os

'''
desc: 把一天的时间分段
param: x@number
return: @number
'''
def hour_cut(x):
    if 0<=x<6:
        return 0
    elif  6<=x<8:
        return 1
    elif  8<=x<12:
        return 2
    elif  12<=x<14:
        return 3
    elif  14<=x<18:
        return 4
    elif  18<=x<21:
        return 5
    elif  21<=x<24:
        return 6

'''
desc: 划分出生的年代
param: x@number
return: @number
'''
def birth_split(x):
    if 1920 <= x < 1930:
        return 0
    elif 1930 <= x < 1940:
        return 1
    elif 1940 <= x < 1950:
        return 2
    elif 1950 <= x < 1960:
        return 3
    elif 1960 <= x < 1970:
        return 4
    elif 1970 <= x < 1980:
        return 5
    elif 1980 <= x < 1985:
        return 6
    elif 1985 <= x < 1990:
        return 7
    elif 1990 <= x < 1995:
        return 8
    elif 1995 <= x < 2000:
        return 9

'''
desc: 收入分组
param: x@number
return: @number
'''
def income_cut(x):
    if x <= 0:
        return 0
    elif 0 < x <= 2500:
        return 1
    elif 2500 < x <= 5000:
        return 2
    elif 5000 < x <= 10000:
        return 3
    elif 10000 < x <= 25000:
        return 4
    elif 25000 < x <= 50000:
        return 5
    elif 50000 < x <= 75000:
        return 6
    elif 75000 < x <= 100000:
        return 7
    elif 100000 < x <= 150000:
        return 8
    elif 150000 < x <= 200000:
        return 9
    elif 200000 < x <= 300000:
        return 10
    elif 300000 < x <= 400000:
        return 11
    elif 400000 < x <= 500000:
        return 12
    elif 500000 < x <= 1000000:
        return 13
    elif x > 1000000:
        return 14

'''
desc: 房产数分组
param: x@number
return: @number
'''
def house_cut(x):
    if x <= 0:
        return 0
    elif 0 < x <= 5:
        return x
    elif x > 5:
        return 8

'''
desc: 住房面积分组
param: x@number
return: @number
'''
def floor_area_cut(x):
    if x <= 0:
        return 0
    elif 0 < x <= 10:
        return 1
    elif 10 < x <= 20:
        return 2
    elif 20 < x <= 30:
        return 3
    elif 30 < x <= 40:
        return 4
    elif 40 < x <= 50:
        return 5
    elif 50 < x <= 70:
        return 6
    elif 70 < x <= 100:
        return 7
    elif 100 < x <= 130:
        return 8
    elif 130 < x <= 150:
        return 9
    elif 150 < x <= 175:
        return 10
    elif 175 < x <= 200:
        return 11
    elif 200 < x <= 250:
        return 12
    elif 250 < x <= 300:
        return 13
    elif 300 < x <= 500:
        return 14
    elif 500 < x <= 1000:
        return 15
    elif x > 1000:
        return 16  

'''
desc: 自定义评价函数
'''
def myFeval(preds, xgbtrain):
    label = xgbtrain.get_label()
    score = mean_squared_error(label,preds)
    return 'myFeval',score

'''
desc: 存储模型
'''
def save_model(clf, model_name):
    dirsName = 'Model_Saved'
    curPath = os.path.dirname(os.path.realpath(__file__))
    dirs = curPath[:-2] + os.sep + dirsName

    if not os.path.exists(dirs):
        os.makedirs(dirs)     
    
    joblib.dump(clf, dirs + os.sep + model_name + '.pkl')  # 保存模型

'''
desc: 读取模型
'''
def load_model(model_name):
    dirsName = 'Model_Saved'
    curPath = os.path.dirname(os.path.realpath(__file__))
    dirs = curPath[:-2] + os.sep + dirsName
    clf = joblib.load(dirs + os.sep + model_name + '.pkl') 
    return clf

