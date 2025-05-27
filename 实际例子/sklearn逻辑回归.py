from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. 加载数据集
iris = load_iris()
X = iris.data       # 特征矩阵，形状 (150, 4)
y = iris.target     # 标签向量，值为 0, 1, 2 对应三种花

# 2. 拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 建立逻辑回归分类器并训练
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# 4. 在测试集上进行预测
y_pred = clf.predict(X_test)

# 5. 输出评估指标
print("分类准确率：", accuracy_score(y_test, y_pred))
print("\n详细分类报告：\n", classification_report(y_test, y_pred, target_names=iris.target_names))


