import torch
import matplotlib
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import numpy as np
from umap import UMAP

from loader import load
from config import batch_size
from model import model


def get_embeddings(signals):
    feature_vectors = []
    model.eval()
    for i, signal in enumerate(load(np.array(signals), batch_size)):
        feature_vectors.extend(model.forward_once(torch.Tensor(signal)))

    for i, _ in enumerate(feature_vectors):
        feature_vectors[i] = feature_vectors[i].detach().numpy()
    feature_vectors = np.array(feature_vectors)

    return feature_vectors


def plot_embeddings(embeddings, targets, fig_name):
    plot.figure(figsize=(12, 12))

    pca = PCA(2)
    pca.fit(embeddings)
    compressed_features = pca.transform(embeddings)

    colors = matplotlib.colors.ListedColormap(['green', 'red'])

    plot.scatter(compressed_features[:, 0], compressed_features[:, 1],
                 c=targets, cmap=colors)

    plot.savefig(f'result/PCA_2d_{fig_name}.png')
    plot.close()

    pca = PCA(3)
    pca.fit(embeddings)
    compressed_features = pca.transform(embeddings)

    axes = plot.axes(projection='3d')
    axes.scatter3D(compressed_features[:, 0], compressed_features[:, 1], compressed_features[:, 2],
                   c=targets, cmap=colors)
    plot.savefig(f'result/PCA_3d_{fig_name}.png')
    plot.close()

    reducer = UMAP()
    compressed_features = reducer.fit_transform(embeddings)

    plot.scatter(compressed_features[:, 0], compressed_features[:, 1],
                 c=targets, cmap=colors)

    plot.savefig(f'result/UMAP_2d_{fig_name}.png')
    plot.close()

    reducer = UMAP(n_components=3)
    compressed_features = reducer.fit_transform(embeddings)

    axes = plot.axes(projection='3d')
    axes.scatter(compressed_features[:, 0], compressed_features[:, 1], compressed_features[:, 2],
                 c=targets, cmap=colors)

    plot.savefig(f'result/UMAP_3d_{fig_name}.png')
    plot.close()
