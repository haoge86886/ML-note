#多项式回归
#特征的关系可能是多次的,但因符合可加性所以仍可以算是线性回归
#通过引入特征的高次项，拟合更复杂的非线性关系。
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

#特别注意标准化问题,因为高次项会将数据量级增长许多,不标准化极有可能梯度爆炸
torch.manual_seed(0)
x = torch.linspace(0, 5,100).view(-1, 1)
x_std = (x - x.mean()) / x.std()                            #注意标准化,防止梯度爆炸
y_true = 1 + 2 * x + 3 * x**2 + 0.5 * torch.randn(x.size())
x_poly = torch.cat([x_std, x_std**2], dim=1)         #拼接x与x的二次项


w = torch.randn((2,1), requires_grad=True)
b = torch.randn((1,), requires_grad=True)

lr = 0.01
epochs = 1000
l = []

for epoch in range(epochs):
    y_pred = x_poly@w + b
    loss = (y_pred - y_true).pow(2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    l.append(loss.item())

    if epoch % 50 == 0:
        plt.scatter(x.numpy(), y_true.numpy(),label='真实值',color='r')
        plt.plot(x.numpy(),y_pred.detach().numpy(),label='预估值')
        plt.title(f'loss: {loss.item()}')
        plt.legend()
        plt.show()

plt.plot([i for i in range(epochs)],l)
plt.title(f"损失函数收敛曲线")
plt.show()
