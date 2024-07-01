from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def logistic_regression(X_train, X_test, y_train, y_test):
    # 逻辑回归模型训练
    log_reg = LogisticRegression(multi_class='ovr', max_iter=200)
    log_reg.fit(X_train, y_train)

    # 预测
    y_pred_log_reg = log_reg.predict(X_test)

    # 模型评估
    print("逻辑回归分类报告：")
    print(classification_report(y_test, y_pred_log_reg))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred_log_reg, labels=log_reg.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("logistic regression - confusion_matrix")
    plt.show()
