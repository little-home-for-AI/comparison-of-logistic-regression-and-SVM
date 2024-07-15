import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.decomposition import PCA
import numpy as np

import matplotlib
matplotlib.use('TkAgg')  # 或者尝试 'Qt5Agg'，取决于你的系统和偏好
import matplotlib.pyplot as plt

# 定义列名，因为原始数据没有列头
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                'Proanthocyanins', 'Color intensity', 'Hue',
                'OD280/OD315 of diluted wines', 'Proline']

# 加载数据，指定分隔符为逗号，跳过第一行（如果没有列头）
df = pd.read_csv('wine/wine.data', names=column_names, sep=',', header=None)

X = df.drop('Class', axis=1)  # 特征
y = df['Class']  # 目标变量

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
"""
# 构建逻辑回归模型
model = LogisticRegression(max_iter=1000)  # 设置迭代次数防止收敛警告
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
"""


class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls


class OneVsRestClassifier:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.models = []

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        for i in range(self.n_classes):
            model = MyLogisticRegression(self.learning_rate, self.n_iters)
            y_binary = (y == i).astype(int)
            model.fit(X, y_binary)
            self.models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.argmax(predictions, axis=0)


# 确保X_train, X_test, y_train, y_test 已经定义好
# 创建并训练OvR分类器
ovr_classifier = OneVsRestClassifier(learning_rate=0.001, n_iters=1000)
ovr_classifier.fit(X_train, y_train)

# 进行预测
y_pred = ovr_classifier.predict(X_test)

# 输出分类报告
print(classification_report(y_test, y_pred, zero_division=1))

# 输出混淆矩阵
print(confusion_matrix(y_test, y_pred))

# 输出准确率
print("Accuracy:", accuracy_score(y_test, y_pred))

# 获取所有子模型的系数
all_coefficients = [model.weights for model in ovr_classifier.models]

# 如果是多分类问题，all_coefficients 将是一个二维数组，每一行对应一个类别的系数
# 我们可以计算每个特征的平均绝对系数来衡量其总体重要性
feature_importances = np.mean(np.abs(all_coefficients), axis=0)

# 将特征名称和重要性配对
feature_importance_pairs = list(zip(column_names[1:], feature_importances))

# 按重要性排序
sorted_pairs = sorted(feature_importance_pairs, key=lambda x: x[1], reverse=True)

# 提取排序后的特征名称和重要性
sorted_features, sorted_importances = zip(*sorted_pairs)

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_features)), sorted_importances, tick_label=sorted_features)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Logistic Regression Model')
plt.show()


# 使用PCA降维至2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 将降维后的数据分为训练和测试集
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 训练模型（使用降维后的数据）
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)

# 预测
y_pred = model.predict(X_test_pca)

# 创建一个网格来绘制决策边界
x_min, x_max = X_pca[:, 0].min() - .5, X_pca[:, 0].max() + .5
y_min, y_max = X_pca[:, 1].min() - .5, X_pca[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.8)

# 绘制测试集样本点
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolors='k', marker='o')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Logistic Regression Decision Boundary on PCA-reduced Wine Dataset')

# 显示图形
plt.show()
