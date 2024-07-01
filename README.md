
# 实验 2：基于葡萄酒数据建模分析

## 目录结构

```
.
├── data_preprocessing.py
├── logistic_regression_model.py
├── svm_model.py
├── visualization.py
└── main.py
```

## 实验简介

本实验旨在使用葡萄酒数据集进行逻辑回归和支持向量机的算法建模，对比模型性能，选取参数，将分类效果进行可视化（混淆矩阵和测试集数据的二维特征的决策边界）。作图时可适当选用 PCA 和 TSNE 降维。

## 依赖环境

需要安装以下 Python 库：

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

可以使用以下命令安装所需库：

```sh
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 文件说明

1. **data_preprocessing.py**

   包含数据预处理的函数，包括数据读取、标准化和训练集与测试集的拆分。

2. **logistic_regression_model.py**

   包含逻辑回归模型的训练、预测和评估函数，并生成混淆矩阵的可视化。

3. **svm_model.py**

   包含支持向量机模型的训练、预测和评估函数，并生成混淆矩阵的可视化。

4. **visualization.py**

   包含使用 PCA 和 TSNE 降维并可视化数据的函数。

5. **main.py**

   主文件，用于调用其他模块的函数，完成数据预处理、逻辑回归建模、支持向量机建模和结果可视化。

## 使用方法

1. 确保数据文件 `wine.data` 位于合适的位置（例如，`/mnt/data/wine.data`）。

2. 将所有 Python 文件保存到同一目录下。

3. 运行 `main.py` 文件：

   ```sh
   python main.py
   ```

## 输出结果

运行 `main.py` 后，程序将完成以下任务：

1. 数据预处理：标准化数据，并拆分为训练集和测试集。
2. 逻辑回归建模：训练逻辑回归模型，进行预测，并输出分类报告和混淆矩阵。
3. 支持向量机建模：训练支持向量机模型，进行预测，并输出分类报告和混淆矩阵。
4. 可视化：使用 PCA 和 TSNE 降维并可视化数据，展示不同类别数据在二维空间中的分布情况。

## 结果示例

- 逻辑回归分类报告和混淆矩阵
- 支持向量机分类报告和混淆矩阵
- PCA 和 TSNE 降维后的数据分布图

## 注意事项

- 确保数据文件路径正确。
- 调整模型参数和数据预处理方式，可以进一步优化模型性能。

## 参考资料

- scikit-learn 文档：https://scikit-learn.org/stable/documentation.html
- matplotlib 文档：https://matplotlib.org/stable/contents.html
- seaborn 文档：https://seaborn.pydata.org/
