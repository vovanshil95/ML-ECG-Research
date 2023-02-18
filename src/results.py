import torch
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import numpy as np

from loader import load
from config import data_path, batch_size
from preprocessing import prepare_data
from model import model

signals = prepare_data(data_path)

model.load_state_dict(torch.load('result/trained/trained1'))

feature_vectors = []

model.eval()
for i, signal in enumerate(load(signals, batch_size)):

    feature_vectors.extend(model.forward_once(torch.Tensor(signal)))
    print(f"processed {(i+1) * batch_size} signals of 720")

for i, _ in enumerate(feature_vectors):
    feature_vectors[i] = feature_vectors[i].detach().numpy()
feature_vectors = np.array(feature_vectors)

torch.save(feature_vectors, 'result/feature_vectors')


plot.figure(figsize=(12, 12))

pca = PCA(2)
pca.fit(feature_vectors)
compressed_features = pca.transform(feature_vectors)

plot.scatter(compressed_features[:, 0], compressed_features[:, 1])
plot.savefig('result/2d_p_val.png')

pca = PCA(3)
pca.fit(feature_vectors)
compressed_features = pca.transform(feature_vectors)

axes = plot.axes(projection='3d')
axes.scatter3D(compressed_features[:, 0], compressed_features[:, 1], compressed_features[:, 2])
plot.savefig('result/3d_p_val.png')
