## 一起挖掘幸福感

### 运行环境
- anaconda 3（python 3.7.3）
- web.py
- xgboost

### 关键特征说明
> 随机森林特征重要性评估
- *equity：认为社会是否公平（1 = 完全不公平; 2 = 比较不公平; 3 = 说不上公平但也不能说不公平; 4 = 比较公平; 5 = 完全公平; ）
- *depression：在过去的四周中您感到心情抑郁或沮丧的频繁程度（1 = 总是; 2 = 经常; 3 = 有时; 4 = 很少; 5 = 从不;）
- *floor_area：您现在住的这座住房的套内建筑面积
- county：采访地点-县/区编码
- city：采访地点-地级市编码
- *family_income：您家去年全年家庭总收入
- *weight_jin：您目前的体重是（斤）
- *class：您认为自己目前处于哪个等级上（1 = 1(最底层); 10 = 10(最顶层); ）
- *class_10_after：您认为您10年前处于哪个等级上（1 = 1(最底层); 10 = 10(最顶层); ）
- *survey_age
- *height_cm：您目前的身高是（厘米）
- *birth：您的出生日期-年
- *income：您个人去年全年的总收入
- marital_1st：您第一次结婚的时间
- *inc_exp：您认为您的年收入达到多少元，您才会比较满意
- public_service_7：您对下列公共服务其他各领域的满意度-低保，灾害，流浪乞讨，残疾，孤儿救助等
- public_service_6：您对下列公共服务其他各领域的满意度-社会保障
- province：采访地点-省/自治区/直辖市编码（1 = 上海市; 2 = 云南省; 3 = 内蒙古自治区; 4 = 北京市; 5 = 吉林省; 6 = 四川省; 7 = 天津市; 8 = 宁夏回族自治区; 9 = 安徽省; 10 = 山东省; 11 = 山西省; 12 = 广东省; 13 = 广西壮族自治区; 14 = 新疆维吾尔自治区; 15 = 江苏省; 16 = 江西省; 17 = 河北省; 18 = 河南省; 19 = 浙江省; 20 = 海南省; 21 = 湖北省; 22 = 湖南省; 23 = 甘肃省; 24 = 福建省; 25 = 西藏自治区; 26 = 贵州省; 27 = 辽宁省; 28 = 重庆市; 29 = 陕西省; 30 = 青海省; 31 = 黑龙江省; ）
- s_birth：您目前的配偶或同居伴侣是哪一年出生的

```
desc: forest evaluation
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
```

> xgbBoost特征重要性评估
- *equity
- *health：您觉得您目前的身体健康状况（1 = 很不健康; 2 = 比较不健康; 3 = 一般; 4 = 比较健康; 5 = 很健康;）
- *age：年龄
- *class
- *floor_area
- *income
- *status_3_before：与三年前相比，您的社会经济地位发生了什么变化（1 = 上升了; 2 = 差不多; 3 = 下降了; ）
- *class_10_after：
- *relax：在过去一年中，您是否经常在您的空闲时间做下面的事情-休息放松（1 = 从不; 2 = 很少; 3 = 有时; 4 = 经常; 5 = 非常频繁; ）
- *family_status：您家的家庭经济状况在所在地属于哪一档（1 = 远低于平均水平; 2 = 低于平均水平; 3 = 平均水平; 4 = 高于平均水平; 5 = 远高于平均水平; ）
- class_14：您认为在您14岁时，您的家庭处在哪个等级上（1 = 1(最底层); 10 = 10(最顶层); ）
- social_neighbor：您与邻居进行社交娱乐活动的频繁程度（1 = 几乎每天; 2 = 一周1到2次; 3 = 一个月几次; 4 = 大约一个月1次; 5 = 一年几次; 6 = 一年1次或更少; 7 = 从来不; ）
- class_10_before
- *religion：您的宗教信仰-不信仰宗教（0 = 否; 1 = 是; ）
- *marital：您目前的婚姻状况（1 = 未婚; 2 = 同居; 3 = 初婚有配偶; 4 = 再婚有配偶; 5 = 分居未离婚; 6 = 离婚; 7 = 丧偶; ）
- *family_m：您家目前住在一起的通常有几人（包括您本人）
- neighbor_familiarity：您和邻居，街坊/同村其他居民互相之间的熟悉程度（1 = 非常不熟悉; 2 = 不太熟悉; 3 = 一般; 4 = 比较熟悉; 5 = 非常熟悉; ）
- view：根据您的一般印象您对一些重要事情所持的观点和看法与社会大众一致的时候有多少（1 = 一致的时候非常少; 2 = 一致的时候比较少; 3 = 一般; 4 = 一致的时候比较多; 5 = 一致的时候非常多; ）
- *house：您家现拥有几处房产
- *inc_ability：考虑到您的能力和工作状况，您目前的收入是否合理（1 = 非常合理; 2 = 合理; 3 = 不合理; 4 = 非常不合理; ）

> 算法筛选+人工筛选
- birth 您的出生日期-年
- height_cm 您目前的身高是（厘米）
- weight_jin 您目前的体重是（斤）
- health 您觉得您目前的身体健康状况（1 = 很不健康; 2 = 比较不健康; 3 = 一般; 4 = 比较健康; 5 = 很健康;）
- marital 您目前的婚姻状况（1 = 未婚; 2 = 同居; 3 = 初婚有配偶; 4 = 再婚有配偶; 5 = 分居未离婚; 6 = 离婚; 7 = 丧偶; ）
- religion 您的宗教信仰-不信仰宗教（0 = 否; 1 = 是; ）
- depression 在过去的四周中您感到心情抑郁或沮丧的频繁程度（1 = 总是; 2 = 经常; 3 = 有时; 4 = 很少; 5 = 从不;）
- relax 在过去一年中，您是否经常在您的空闲时间做下面的事情-休息放松（1 = 从不; 2 = 很少; 3 = 有时; 4 = 经常; 5 = 非常频繁; ）
- class 您认为自己目前处于哪个等级上（1 = 1(最底层); 10 = 10(最顶层); ）
- status_3_before 与三年前相比，您的社会经济地位发生了什么变化（1 = 上升了; 2 = 差不多; 3 = 下降了; ）
- class_10_after 您认为您10年后将会在哪个等级上（1 = 1(最底层); 10 = 10(最顶层); ）
- income 您个人去年全年的总收入
- inc_exp 您认为您的年收入达到多少元，您才会比较满意
- inc_ability 考虑到您的能力和工作状况，您目前的收入是否合理（1 = 非常合理; 2 = 合理; 3 = 不合理; 4 = 非常不合理; ）
- house 您家现拥有几处房产
- floor_area 您现在住的这座住房的套内建筑面积
- family_m 您家目前住在一起的通常有几人（包括您本人）
- family_income 您家去年全年家庭总收入
- family_status 您家的家庭经济状况在所在地属于哪一档（1 = 远低于平均水平; 2 = 低于平均水平; 3 = 平均水平; 4 = 高于平均水平; 5 = 远高于平均水平; ）
- social_neighbor 您与邻居进行社交娱乐活动的频繁程度（1 = 几乎每天; 2 = 一周1到2次; 3 = 一个月几次; 4 = 大约一个月1次; 5 = 一年几次; 6 = 一年1次或更少; 7 = 从来不; ）
- social_friend 您与其他朋友进行社交娱乐活动的频繁程度（1 = 几乎每天; 2 = 一周1到2次; 3 = 一个月几次; 4 = 大约一个月1次; 5 = 一年几次; 6 = 一年1次或更少; 7 = 从来不; ）
- equity 认为社会是否公平（1 = 完全不公平; 2 = 比较不公平; 3 = 说不上公平但也不能说不公平; 4 = 比较公平; 5 = 完全公平; ）