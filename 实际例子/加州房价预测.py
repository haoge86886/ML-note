from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

housing = fetch_california_housing()
scaler = StandardScaler()
X = scaler.fit_transform(housing.data)
X = torch.from_numpy(X).float()
y = torch.from_numpy(housing.target).float().view(-1,1)

w = torch.randn((8,1), requires_grad=True)
b = torch.randn(1, requires_grad=True)
lr = 0.01                        #过大会梯度爆炸
epochs = 500
l = []
for i in range(epochs):
    y_pred = torch.mm(X,w) + b      # 不要设置 requires_grad=False
    loss = ((y_pred - y)**2).mean()  # 会自动构建计算图

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()
    b.grad.zero_()

    l.append(loss.item())

    if i % 20 == 0:
            j = 5
            X_print = X[:, j].detach().numpy()
            y_print = y_pred[:, 0].detach().numpy()
            y_true = y[:, 0].detach().numpy()

            plt.figure(figsize=(10,8))
            plt.scatter(X_print, y_true, color='blue', alpha=0.3, label='真实值')
            plt.scatter(X_print, y_print, color='red', alpha=0.5, label='预测值')
            plt.title(f"特征: {housing.feature_names[j]} 损失: {loss.item()}")
            plt.xlabel(housing.feature_names[j])
            plt.ylabel("Target")
            plt.legend()
            plt.show()

plt.plot([i for i in range(epochs)],l)
plt.show()