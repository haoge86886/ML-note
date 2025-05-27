"""""""""
决策边界
当把数据作为横轴,标签作为纵轴,在分类问题中可以用一条线作为不同标签的分界

逻辑分布,sigmoid函数
F(z) = 1/1+e^(-z)
概率密度图像类似于正态分布,但有更高的峰部与尾部
分布函数类似arctan(x)的图像,关于(0,0.5)中心对称

逻辑回归
分类的结果输出只有两种:True或False,用于解决二元分类问题
构造一个函数,使输出是一个在(0,1)间的概率值,
令       z = w▪x+b   f(x) = g(z) = 1/1+e^(-z) = 1/1+e^(w▪x+b)
同时,y只取到0或1,当f(xi) = 0.7 , 代表xi有70%的概率属于y=1的情况,有30%的概率是y=0
即                 f(xi) = P( y=1 | xi ) = 0.7
将会以一个阈值作为决策边界x = a,当f(xi)大于该值,则认为y=1,小于认为y=0
                 y = { 1     f(xi) > a
                     { 0     f(xi) < a

多特征
有两个特征x1,x2,则f(x) = g(z) = g(w1x1 + w2x2 + b)
在坐标系上,w1x1 + w2x2 + b = 0 就是决策边界,当z > 0,点在直线右边,f(x)>0,y=1
边界可以更复杂,引入高次项等等,即z = w1x1 + w2x1^2 + w3x1^3 ........
"""""""""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 生成可分数据（两个特征，两个类别）
num_samples = 100
torch.manual_seed(0)
x_class0 = torch.randn(num_samples, 2) + torch.tensor([-2, -2])  # 类别0
x_class1 = torch.randn(num_samples, 2) + torch.tensor([2, 2])    # 类别1
X = torch.cat([x_class0, x_class1], dim=0)
y = torch.cat([torch.zeros(num_samples), torch.ones(num_samples)]).unsqueeze(1)


# 2. 定义逻辑回归模型（线性 + Sigmoid）
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)   #线性回归设定z

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression()

# 3. 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 训练模型
for epoch in range(1000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. 可视化数据点 + 决策边界
w, b = model.linear.weight.data[0], model.linear.bias.data[0]
x_vals = torch.linspace(-6, 6, 100)
y_vals = -(w[0] * x_vals + b) / w[1]

plt.figure(figsize=(6, 6))
plt.scatter(x_class0[:, 0], x_class0[:, 1], color='blue', label='Class 0')
plt.scatter(x_class1[:, 0], x_class1[:, 1], color='red', label='Class 1')
plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
plt.legend()
plt.title('Logistic Regression Decision Boundary')
plt.grid(True)
plt.show()