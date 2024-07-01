from sklearn.preprocessing import StandardScaler
from data_preprocessing import preprocess_data
from logistic_regression_model import logistic_regression
from SVM_model import svm_model
from visualization import visualize
import pandas as pd
# 文件路径
file_path = 'wine/wine.data'

# 数据预处理
X_train, X_test, y_train, y_test = preprocess_data(file_path)

# 逻辑回归模型
logistic_regression(X_train, X_test, y_train, y_test)

# 支持向量机模型
svm_model(X_train, X_test, y_train, y_test)

# 可视化
# 将所有数据传递给可视化函数
X_scaled = StandardScaler().fit_transform(pd.read_csv(file_path, header=None).drop(0, axis=1))
y = pd.read_csv(file_path, header=None)[0]
visualize(X_scaled, y)
