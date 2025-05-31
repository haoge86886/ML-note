"""""""""

平方误差成本函数: J(w,b) = 1/2m ∑ (f(xi) - yi)²
交叉熵成本函数 : L(w,b) = -1/m ∑ [yi*ln(f(xi))+(1-y)*ln(1-f(xi))]

线性回归中更多使用平方误差成本函数

将j最小化,即说明模型的拟合程度好,以找到最好的w,b
"""""""""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test, y_test, color='blue', label='真实值')
plt.plot(X_test, y_pred, color='red', label='预测值')
plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归示例")
plt.legend()
plt.show()

def cost_function(y_test, y_pred):
    m =  len(y)
    return 1/2*m*(np.sum(pow((y_test-y_pred),2)))

cost = cost_function(y_test, y_pred)
print('成本:',cost)