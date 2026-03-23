"""
拟合——欠拟合、恰好拟合、过拟合
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分训练集/测试集
from sklearn.linear_model import LinearRegression  # 线性回归模型
from sklearn.metrics import mean_squared_error  # 均方误差损失函数
from sklearn.preprocessing import PolynomialFeatures  # 构建多项式特征

# 生成数据
x = np.linspace(-np.pi, np.pi, 300).reshape(-1, 1)
y = np.sin(x) + np.random.uniform(-0.8, 0.8, 300).reshape(-1, 1)

# 构建散点图
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].scatter(x, y)
axs[1].scatter(x, y)
axs[2].scatter(x, y)

# 划分数据集：训练集/测试集
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# 定义模型
model = LinearRegression()

# 欠拟合
#   训练模型
model.fit(train_x, train_y)
#   模型评估
train_loss = mean_squared_error(train_y, model.predict(train_x))
test_loss = mean_squared_error(test_y, model.predict(test_x))
print(f"欠拟合：\n训练误差：{train_loss}，测试误差：{test_loss}\n模型评估分数：{model.score(test_x, test_y)}")
#   绘制曲线
axs[0].plot(x, model.predict(x), c='r')
print('\n···································\n')

# 恰好拟合
#   特征工程
poly5 = PolynomialFeatures(degree=5)  # 构建多项式特征
train_x2 = poly5.fit_transform(train_x)
test_x2 = poly5.transform(test_x)
#   训练模型
model.fit(train_x2, train_y)
#   模型评估
train_loss = mean_squared_error(train_y, model.predict(train_x2))
test_loss = mean_squared_error(test_y, model.predict(test_x2))
print(f"恰好拟合：\n训练误差：{train_loss}，测试误差：{test_loss}\n模型评估分数：{model.score(test_x2, test_y)}")
#   绘制曲线
axs[1].plot(x, model.predict(poly5.transform(x)), c='r')
print('\n···································\n')

# 过拟合
#   特征工程
poly22 = PolynomialFeatures(degree=22)  # 构建多项式特征
train_x3 = poly22.fit_transform(train_x)
test_x3 = poly22.transform(test_x)
#   训练模型
model.fit(train_x3, train_y)
#   模型评估
train_loss = mean_squared_error(train_y, model.predict(train_x3))
test_loss = mean_squared_error(test_y, model.predict(test_x3))
print(f"过拟合：\n训练误差：{train_loss}，测试误差：{test_loss}\n模型评估分数：{model.score(test_x3, test_y)}")
#   绘制曲线
axs[2].plot(x, model.predict(poly22.transform(x)), c='r')

plt.show()
