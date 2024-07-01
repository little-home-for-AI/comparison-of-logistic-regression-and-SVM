import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize(X_scaled, y):
    # PCA dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # TSNE dimensionality reduction
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_scaled)

    # Visualization of decision boundaries
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=50)
    plt.title("PCA - Logistic Regression Decision Boundary")

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='viridis', s=50)
    plt.title("TSNE - Logistic Regression Decision Boundary")

    plt.show()
