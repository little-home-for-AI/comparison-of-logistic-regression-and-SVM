from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

def svm_model(X_train, X_test, y_train, y_test):
    # 支持向量机模型训练
    svc = SVC(kernel='linear', probability=True)
    svc.fit(X_train, y_train)

    # 预测
    y_pred_svc = svc.predict(X_test)

    # 模型评估
    print("支持向量机分类报告：")
    print(classification_report(y_test, y_pred_svc))

    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred_svc, labels=svc.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("support SVM-confusion matrix")
    plt.show()
