from  preprocess import X_train, y_train, X_test, y_test
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

# 逻辑回归模型训练
log_reg = LogisticRegression(multi_class='ovr', max_iter=200)
log_reg.fit(X_train, y_train)

# 预测
y_pred_log_reg = log_reg.predict(X_test)

# 模型评估
print("逻辑回归分类报告：")
print(classification_report(y_test, y_pred_log_reg))

# 混淆矩阵可视化
plot_confusion_matrix(log_reg, X_test, y_test, display_labels=["Class 1", "Class 2", "Class 3"], cmap=plt.cm.Blues)
plt.title("逻辑回归 - 混淆矩阵")
plt.show()
