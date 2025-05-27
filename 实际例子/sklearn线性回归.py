import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. 生成样本数据
# 假设 x 和 y 存在线性关系：y = 2x + 1，加上一些噪声
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)                 #人为制造的线性关系,并加上噪声

# 2. 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 4. 输出回归系数和截距
print("回归系数（斜率）：", model.coef_)
print("截距：", model.intercept_)

# 5. 使用测试集进行预测
y_pred = model.predict(X_test)

# 6. 可视化结果
plt.scatter(X_test, y_test, color='blue', label='真实值')
plt.plot(X_test, y_pred, color='red', label='预测值')
plt.xlabel("X")
plt.ylabel("y")
plt.title("线性回归示例")
plt.legend()
plt.show()

