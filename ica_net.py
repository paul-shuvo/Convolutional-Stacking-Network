import numpy as np
from sklearn.decomposition import PCA


class ICANet:
    def __init__(self, cfg=None, data=None):
        self.cfg = cfg
        self.data = data

    def get_feature_maps(self, n_feature_maps, data=None, layers = 1):

        if data is None:
            data = self.data

        # dim = data.shape

        pca_layers = {}
        for layer in range(layers):

            # key = str(layer + 1) + " : " + str(n_feature_maps[layer])
            key = layer + 1
            pca_layers[key] = self.get_PCA(n_feature_maps[layer], data)

        return pca_layers


    def get_PCA(self, feature_maps, data=None):
        pca = PCA(feature_maps)
        s = pca.fit(data)
        return pca.fit(data)

