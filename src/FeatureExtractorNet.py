from StackingConvNet import StackingConvNet
from sklearn.decomposition import PCA, KernelPCA, FastICA, FactorAnalysis
import numpy as np


# *******************
class FeatureExtractorNet(StackingConvNet):
    # **********
    def __init__(self, cfg_name, config_section="DEFAULT"):
        """
        Constructor function.
        :param cfg_name: Configuration file string
        :param config_section: Configuration section string
        """

        # Explicitly call the parent class' constructor
        super().__init__(cfg_name, config_section)

        # Set the feature map generator functions
        self.train_feature_extractors = {'PCA': self.train_PCA, 'Kernel_PCA': self.train_Kernel_PCA, 'ICA': self.train_ICA, 'FA': self.train_FA}
        self.extract_features = {'PCA': self.extract_PCA_features, 'Kernel_PCA': self.extract_KPCA_features, 'ICA': self.extract_ICA_features, 'FA': self.extract_FA_features}

    # **********
    def extract_PCA_features(self, model, feature_patches, components, zero_pad, feature_map_shape, stride):
        """
        Function to extract features using Principal Component Analysis (PCA).
        :param model: A trained scikit-learn PCA object
        :param feature_patches: Input feature patches to compute PCA on them [patches, height, width]
        :param components: The requested PCA components (list)
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Network stride
        :return: Extracted feature map [batch, height, width, channel]
        """

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Extract the new feature maps via PCA
        extracted_features = model.transform(feature_patches)

        # Reshape the extracted feature map from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return extracted_features

    # **********
    def train_PCA(self, components, feature_patches, kernel_mode, zero_pad, feature_map_shape=None, stride=None):
        """
        Function to compute Principal Component Analysis (PCA) of input patches.
        :param components: The requested PCA components (list)
        :param feature_patches: Input feature patches to compute PCA on them [patches, height, width]
        Warning: This function modifies the feature_patches argument!
        :param kernel_mode: If enabled PCA components are returned instead of the trained extractors themselves
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Stride used in generation of patches
        :return: If kernel mode disabled, PCA feature extractor (a single object) and extracted features [batch, height, width, channel].
        If kernel mode enabled, PCA components (eigenvectors) array [height, width, n_components].
        """

        # Define an instance of PCA dimensionality reduction
        pca = PCA(n_components=max(components) + 1, copy=False)

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        if kernel_mode:
            # Fit the PCA model
            pca.fit(feature_patches)

            # Get and reshape components
            pca_components = pca.components_
            pca_components = np.swapaxes(pca_components, 0, 1)
            pca_components = np.reshape(pca_components,
                                        (feature_patches_shape[1], feature_patches_shape[2], pca_components.shape[1]))

            return pca_components[:, :, components]
        else:
            # Fit the PCA model and extract the new feature maps
            extracted_features = pca.fit_transform(feature_patches)

            # Reshape the extracted patch from [batch * height * width, channel] to [batch, height, width, channel]
            extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                        feature_map_shape, stride)

            # Extract the requested components
            extracted_features = extracted_features[:, :, :, components]

            return pca, extracted_features

    # **********
    def extract_KPCA_features(self, model, feature_patches, components, zero_pad, feature_map_shape, stride):
        """
        Function to extract features using Kernel Principal Component Analysis (KPCA).
        :param model: A trained scikit-learn kernel PCA object
        :param feature_patches: Input feature patches to compute kernel PCA on them [patches, height, width]
        :param components: The requested Kernel PCA components (list)
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Network stride
        :return: Extracted feature map [batch, height, width, channel]
        """

        # Check if kernel mode is not enabled while kernel PCA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for Kernel PCA.'

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                       (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Extract the new feature maps via PCA
        extracted_features = model.transform(feature_patches)

        # Reshape the extracted feature map from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return extracted_features

    # **********
    def train_Kernel_PCA(self, components, feature_patches, zero_pad, feature_map_shape, stride, ** kwargs):
        """
        Function to compute Kernel Principal Component Analysis (KPCA) of input patches.
        :param components: The requested Kernel PCA components (list)
        :param feature_patches: Input feature patches to compute PCA on them [patches, height, width].
        Warning: This function modifies the feature_patches argument!
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Stride used in generation of patches
        :return: KPCA feature extractor (a single object).
        """

        # Check if kernel mode is not enabled while kernel PCA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for Kernel PCA.'

        # Define an instance of Kernel PCA dimensionality reduction
        kpca = KernelPCA(n_components=max(components) + 1, copy_X=False, n_jobs=-1, kernel=self.cfg["kernelPCA_kernel_type"])

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Fit the KPCA model and extract the new feature maps
        extracted_features = kpca.fit_transform(feature_patches)

        # Reshape the extracted patch from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return kpca, extracted_features

    # **********
    def extract_ICA_features(self, model, feature_patches, components, zero_pad, feature_map_shape, stride):
        """
        Function to extract features using Independent Component Analysis (ICA).
        :param model: A trained scikit-learn ICA object
        :param feature_patches: Input feature patches to compute ICA on them [patches, height, width]
        :param components: The requested ICA components (list)
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Network stride
        :return: Extracted feature map [batch, height, width, channel]
        """

        # Check if kernel mode is not enabled while ICA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for ICA.'

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Extract the new feature maps via ICA
        extracted_features = model.transform(feature_patches)

        # Reshape the extracted feature map from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return extracted_features

    # **********
    def train_ICA(self, components, feature_patches, zero_pad, feature_map_shape, stride, **kwargs):
        """
        Function to compute Independent Component Analysis (ICA) of input patches.
        :param components: The requested ICA components (list)
        :param feature_patches: Input feature patches to compute ICA on them [patches, height, width]
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Stride used in generation of patches
        :return: ICA feature extractor (a single object) and extracted features [batch, height, width, channel].
        """

        # Check if kernel mode is not enabled while ICA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for ICA.'

        # Define an instance of ICA dimensionality reduction
        ica = FastICA(n_components=max(components) + 1, max_iter=self.cfg["max_iteration_ICA"])

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Fit the ICA model and extract the new feature maps
        extracted_features = ica.fit_transform(feature_patches)

        # Reshape the extracted patch from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return ica, extracted_features

    # **********
    def extract_FA_features(self, model, feature_patches, components, zero_pad, feature_map_shape, stride):
        """
        Function to extract features using Factor Analysis (FA).
        :param model: A trained scikit-learn FA object
        :param feature_patches: Input feature patches to compute FA on them [patches, height, width]
        :param components: The requested components (list)
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Network stride
        :return: Extracted feature map [batch, height, width, channel]
        """

        # Check if kernel mode is not enabled while FA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for FA.'

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Extract the new feature maps via FA
        extracted_features = model.transform(feature_patches)

        # Reshape the extracted feature map from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return extracted_features

    # **********
    def train_FA(self, components, feature_patches, zero_pad, feature_map_shape, stride, **kwargs):
        """
        Function to compute Factor Analysis (FA) of input patches.
        :param components: The requested components (list)
        :param feature_patches: Input feature patches to compute FA on them [patches, height, width]
        Warning: This function modifies the feature_patches argument!
        :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
        :param feature_map_shape: Shape of the feature map [batch, height, width]
        :param stride: Stride used in generation of patches
        :return: FA feature extractor (a single object) and extracted features [batch, height, width, channel].
        """

        # Check if kernel mode is not enabled while FA is the function to extract features.
        assert not self.cfg["kernel_mode"], 'Kernel mode is not supported for FA.'

        # Define an instance of FA dimensionality reduction
        fa = FactorAnalysis(n_components=max(components) + 1, copy=False, max_iter=self.cfg["max_iteration_FA"])

        # Reshape the the input data to a 2D array (an array of 1D input data)
        feature_patches_shape = feature_patches.shape
        feature_patches = np.reshape(feature_patches,
                                     (feature_patches.shape[0], feature_patches.shape[1] * feature_patches.shape[2]))

        # Fit the Factor Analysis model and extract the new feature maps
        extracted_features = fa.fit_transform(feature_patches)

        # Reshape the extracted patch from [batch * height * width, channel] to [batch, height, width, channel]
        extracted_features = reshape_feature_vector(extracted_features, feature_patches_shape, zero_pad,
                                                    feature_map_shape, stride)

        # Extract the requested components
        extracted_features = extracted_features[:, :, :, components]

        return fa, extracted_features

# *******************
def reshape_feature_vector(feature_vector, feature_patches_shape, zero_pad, feature_map_shape, stride):
    """
    Function to reshape the feature vector of shape [batch * height * width, channel] to [batch, height, width, channel]
    :param feature_vector: Input feature vector to be reshaped (shape: [batch * height * width, channel])
    :param feature_patches_shape: Shape of input feature patches to compute ICA on them [patches, height, width]
    :param zero_pad: Boolean parameter to indicate if feature maps are zero-padded
    :param feature_map_shape: Shape of the feature map [batch, height, width]
    :param stride: Stride used in generation of patches
    :return: Output reshaped feature vector (shape: [batch, height, width, channel])
    """

    # Reshape the extracted patch from [batch * height * width, channel] to [batch, height, width, channel]
    if zero_pad:
        dim1 = int(feature_map_shape[1] / stride)
        dim2 = int(feature_map_shape[2] / stride)
    else:
        dim1 = int((feature_map_shape[1] - feature_patches_shape[1] + 1) / stride)
        dim2 = int((feature_map_shape[2] - feature_patches_shape[2] + 1) / stride)
    reshaped_feature_vector = np.reshape(feature_vector,
                                    (dim2, dim1, feature_map_shape[0], feature_vector.shape[1]))
    reshaped_feature_vector = np.swapaxes(reshaped_feature_vector, 0, 2)

    return reshaped_feature_vector
