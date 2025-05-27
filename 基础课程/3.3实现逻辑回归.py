import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import matplotlib.pyplot as plt

n = 100
x_1 = torch.randn(n,2)+torch.tensor([2,2])
x_2 = torch.randn(n,2)+torch.tensor([0,0])
x = torch.cat((x_1,x_2),dim=0)
y_ture =torch.cat((torch.ones(n),torch.zeros(n)),dim=0).unsqueeze(1)

w = torch.randn(1,2,requires_grad=True)
b = torch.randn(1,1,requires_grad=True)

epochs = 1000
lr = 0.08

m = []
for epoch in range(epochs):
    z = x@w.T + b
    y_pred = torch.sigmoid(z)

    eps = 1e-7
    loss =  -(y_ture * torch.log(y_pred + eps) + (1 - y_ture) * torch.log(1 - y_pred + eps)).mean()
    m.append(loss.item())

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 20 == 0:
        X = torch.linspace(-1,5, 6)
        w1, w2 = w[0,0].item(), w[0,1].item()
        b_val = b[0,0].item()
        Y = -(w1 * X + b_val) / w2
        plt.plot(X,Y, 'k--')

        plt.scatter(x_1[:,0],x_1[:,1],color='r')
        plt.scatter(x_2[:,0],x_2[:,1],color='b')

        plt.title(f'loss:{loss.item():.8f}')
        plt.show()

plt.plot([i for i in range(len(m))],m)
plt.show()