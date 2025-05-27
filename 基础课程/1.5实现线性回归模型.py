import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output

torch.manual_seed(0)
x = torch.linspace(0, 5, 100).reshape(-1, 1)
y_true = 3 * x + 2 + torch.randn_like(x) * 0.5

w = torch.randn(1, requires_grad=True)          #会自动计算梯度并保存在.grad中
b = torch.randn(1, requires_grad=True)

lr = 0.1
epochs = 100

plt.figure(figsize=(6, 4))                      #设置画布

for epoch in range(epochs):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()      #向前传播,计算损失

    loss.backward()                             #反向传播,既是∂(loss)/∂w,把带有requires_grad=True加上梯度值

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    clear_output(wait=True)                     #清除上一次图像
    plt.cla()

    n = torch.linspace(0, 5, 100).reshape(-1, 1)
    plt.scatter(x.numpy(), y_true.numpy(), label="真实数据", s=10)
    plt.plot(n.numpy(), (w * n + b).detach().numpy(), color='red', label="拟合直线")
    plt.title(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w: {w} b: {b}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.pause(0.1)

plt.show()


