from StackingConvNet import StackingConvNet
from sklearn.decomposition import PCA
import numpy as np


class PCA_Net(StackingConvNet):
    # **********
    def __init__(self, cfg_name, configSection="DEFAULT"):
        """
        Constructor function.
        """

        # Explicitly call the parent class' constructor
        super().__init__(cfg_name, configSection)

        # Set the feature map generator function to PCA
        self.get_featureMaps = self.get_PCA

    # **********
    def get_PCA(self, n_components, feature_patches):
        """
        Function to compute Principal Component Analysis (PCA) of input patches.
        The output components array has the shape [height, width, n_components].
        """

        # Define an instance of PCA dimensionality reduction
        pca = PCA(n_components)

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches, (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Fit the PCA model
        pca.fit(feature_patches)

        # Normalize components (remove the component mean)
        components = np.subtract(pca.components_, np.expand_dims(np.mean(pca.components_, axis=1), axis=1))

        # Reshape components
        components = np.swapaxes(components, 0, 1)
        components = np.reshape(components, (feature_patches_shape[1], feature_patches_shape[2], components.shape[1]))

        return components
