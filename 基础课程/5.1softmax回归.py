"""""
可以类比逻辑回归,同样是将数据汇总后投入激活函数中得到(0,1)间的值,代表其取每一个类别的概率

不同点在于:
          1.为了处理多类别,w和b的维度要拓展到与类别数相同,以进行学习
          2.激活函数换为softmax函数,ai = softmax(zi) = e^(zi) / Σ e^(zi)
            结果分别为P{y=i|X},ai为预测的类别
            
仍然使用交叉熵损失函数：L = 1/n Σ -ln(ai)  if y=i
"""""
import torch
import torch.nn.functional as F                         #函数模块
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 生成数据：1000个样本，20个特征，3个类别
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, n_redundant=0)
X = StandardScaler().fit_transform(X)  # 标准化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 转换为 tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# 模型参数初始化
num_features = X_train.shape[1]
num_classes = 3

W = torch.randn(num_features, num_classes, requires_grad=True)
b = torch.zeros(num_classes, requires_grad=True)

lr = 0.1
epochs = 1000

for epoch in range(epochs):
    # 前向传播：计算 logits 和 softmax 概率
    logits = X_train @ W + b
    probs = F.softmax(logits, dim=1)

    log_probs = torch.log(probs + 1e-9)                     # 防止 log(0)
    loss = -log_probs[range(len(y_train)), y_train].mean()

    # 反向传播
    loss.backward()

    # 参数更新
    with torch.no_grad():
        W -= lr * W.grad
        b -= lr * b.grad
        W.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 100 == 0:
        pred = torch.argmax(probs, dim=1)
        acc = (pred == y_train).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")
