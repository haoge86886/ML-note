from sklearn.datasets import load_iris                      #数据集
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

iris = load_iris()
X = iris.data
y = iris.target
binary_filter = y < 2
X_binary = X[binary_filter]                             #二分类任务,0:setosa,1:versicolor
y_binary = y[binary_filter]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_binary)               #数据标准化
X_t = torch.from_numpy(X_scaled).float()
y_t = torch.from_numpy(y_binary).float()

w = torch.randn(4,1,requires_grad=True)                 #权重
b = torch.randn(1,1,requires_grad=True)                 #偏置

lr = 0.5
epochs = 200

m = []                                                  #损失记录

for epoch in range(epochs):
    z = torch.mm(X_t,w)+ b                                       #加权计算
    y_pred = torch.sigmoid(z)                           #激活函数

    eps  = 1e-7
    loss = - (y_t * torch.log(y_pred + eps) + (1 - y_t) * torch.log(1 - y_pred + eps)).mean()
    m.append(loss.item())

    loss.backward()                                     #反向传播

    with torch.no_grad():                               #更新参数
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()                                      #清空梯度
    b.grad.zero_()

    if epoch % 10 == 0:
        x = X_scaled[:,0]
        y = X_scaled[:,1]

        x_plot = torch.from_numpy(X_scaled).float()
        y_pred_plot = torch.sigmoid(x_plot @ w + b).detach().numpy()
        y_pred_label = (y_pred_plot > 0.5).astype(int).flatten()

        plt.scatter(x[y_binary == 0], y[y_binary == 0], c='blue', label='山鸢尾')
        plt.scatter(x[y_binary == 1], y[y_binary == 1], c='green', label='变色鸢尾')

        plt.title(f"损失: {loss.item():.4f}")
        plt.legend()
        plt.pause(0.1)
        plt.show()

plt.plot([i for i in range(len(m))],m)
plt.show()