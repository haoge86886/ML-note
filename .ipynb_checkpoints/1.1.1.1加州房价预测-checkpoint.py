import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

housing = fetch_california_housing()
X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(X_test)

plt.scatter(X_test[:,2].reshape(-1,1), y_test, color='blue', label='真实值')
plt.plot(np.sort(X_test[:,2].reshape(-1,1),axis=0), np.sort(y_pred,axis=0), color='red', label='预测值')
plt.xlabel("X")
plt.ylabel("y")
plt.title("每户平均房间数")
plt.legend()
plt.show()