"""
特征工程——特征转换
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 归一化
x1 = [[1, 2], [0, 5], [3, 1], [7, 8]]
mms = MinMaxScaler(feature_range=(0, 1))
x1 = mms.fit_transform(x1)
print(x1)
print('\n···································\n')

# 标准化
x2 = [[1, 2], [0.5, 6], [0, 10], [1, 18]]
sds = StandardScaler()
x2 = sds.fit_transform(x2)
print(x2)
