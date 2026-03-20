"""
特征工程——特征选择
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# 低方差过滤
#   构造数据，一列标准差为1，一列为0.1
a = np.random.randn(1000)
b = np.random.normal(5, 0.1, 1000)
x = np.vstack((a, b)).T
#   过滤方差小于0.2的
vt = VarianceThreshold(0.2)
x_filtered = vt.fit_transform(x)
print(x_filtered)
print('\n···································\n')

# pearson相关系数
#   读取数据
adv = pd.read_csv(r'..\data\advertising.csv')
#   清洗数据：去掉id列
adv.drop(adv.columns[0], axis=1, inplace=True)
#   分离特征和标签
x = adv.drop('Sales', axis=1, inplace=False)
y = adv['Sales']
#   计算pearson相关系数
corr = x.corrwith(y, method='pearson')
print(corr)
print('\n···································\n')

# PCA
#   定义主成分
a = np.random.normal(0, 1, 1000)
b = np.random.normal(0, 0.5, 1000)
#   定义噪声
c = np.random.normal(0, 0.1, 1000)
#   构造特征
f1 = a + b
f2 = a + c
f3 = b + c
x = np.vstack((a, b, c)).T
#   主成分分析
x_pca = PCA(n_components=2).fit_transform(x)  # n_components浮点数：表示保留多少比例的信息，整数：表示保留的维度数
print(x_pca.shape)
