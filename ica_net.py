import numpy as np
import os, pickle, math
import cv2 as cv
import tensorflow as tf
from sklearn.decomposition import PCA


class ICANet:
    #**********
    def __init__(self, dataset_name, cfg_name=None):
        '''
        Constructor function.
        '''
        
        # Load configurations
#         load_config(cfg_name)
        
        # Load the input dataset
        load_dataset(dataset_name)

    #**********
    def train_kernels(self, n_feature_maps, input_features, kernel_sizes):
        '''
        Function to get kernels computed form input feature maps/images.
        '''
        
        # Set current input features to the function input
        current_input = input_features
        
        # Iterate over the layers of the network
        kernels = []
        for layer, n_current_featureMaps in enumerate(n_feature_maps):
            # Get patches of the previous layer
            patches = get_patches(kernel_sizes[layer], current_input)
            
            # Compute the new feature maps from the current feature maps
            current_kernels = []
            current_layer = np.zeros((0,) + current_input.shape[1:])
            for patches_of_kernel in patches:
                # Get kernels first
                kernels_temp = self.get_PCA(n_feature_maps[layer], patches_of_kernel)
                current_kernels.append(kernels_temp)
                
                # Convolve the input feature maps with the kernels to obtain the new feature maps
                current_input = np.expand_dims(current_input, axis=-1)
                filter = np.expand_dims(kernels_temp, axis=2)
                current_layer.append(np.squeeze(np.array(tf.nn.conv2d(input=current_input, filters=filter, strides=1, padding='SAME'))))
                
            # Set the current layer as the next input
            current_input = current_layer
            
            # Save kernels of this layer
            kernels.append(current_kernels)

        return kernels

    #**********
    def get_PCA(self, n_components, feature_patches):
        '''
        Function to compute Principal Component Analysis (PCA) of input patches.
        The output components array has the shape [height, width, n_components].
        '''
        
        # Define an instance of PCA dimensionality reduction
        pca = PCA(n_components)
        
        # Fit the PCA model
        pca.fit(feature_patches)
        
        # Normalize components (remove the component mean)
        components = pca.components_ - np.mean(pca.components_, axis=1)
        
        # Reshape components
        components = np.swapaxes(components, 0, 1)
        components = np.reshape(components, (feature_patches.shape[1], feature_patches.shape[2], components.shape[1]))
        
        return components

    #**********
    def get_patches(self, kernel_sizes, features_maps):
        '''
        Function to gather patches by scanning kernels over input image/feature maps for all training samples.
        '''
        
        # If the input feature maps are each one 1D, convert them to 2D 
        dims = features_maps.shape
        if len(dims) == 2:
            shape_ = (dims[0], int(np.sqrt(dims[1])), -1)
            features_maps = np.reshape(features_maps, shape_)

        # Iterate over all kernel sizes
        pool_of_patches = []
        for kernel in kernel_sizes:
            # Extract the size in each dimension
            kernel_dim1, kernel_dim2 = kernel
            
            # Scan all the possible patch positions and add them to a numpy array
            featureMap_patches = np.zeros((0, kernel_dim1, kernel_dim2))
            range_dim1 = features_maps.shape[1] - kernel_dim1 + 1
            range_dim2 = features_maps.shape[2] - kernel_dim2 + 1
            for dim1 in range(range_dim1):
                for dim2 in range(range_dim2):
                    current_patch = features_maps[:, dim1 : dim1 + kernel_dim1, dim2 : dim2 + kernel_dim2]
                    featureMap_patches = np.append(featureMap_patches, current_patch, axis=0)
                    
                print(dim1)
                print(featureMap_patches.shape)

            # Add the patches for the current kernel size to a list containing all patches
            pool_of_patches.append(featureMap_patches)
            
            # Write the current patches on a file
            np.save(os.path.join(os.getcwd(), '/Patches/', str(kernel_dim1) + "x" + str(kernel_dim2)), featureMap_patches)
            
            # Delete the patches for the current kernel size
            del featureMap_patches

        return pool_of_patches
    
    #**********
    def load_dataset(self, datsetName):
        '''
        Function to load a dataset from a file.
        '''
        
        # Create the address to the dataset file
        filepath = os.path.join(os.getcwd(), '/Datasets/', datsetName)
        
        # Check the file path
        if not os.path.exists(filepath):
            raise Exception('\nThe dataset file "' + filepath + '" does not exist.\n')
        
        # Open the file
        with open(filepath, 'rb') as dataset_file:          
            # Load the dataset's pickle file
            dataset = pickle.load(dataset_file)
        
        # Separate input data and output labels
        self.Data = dataset[0]
        self.Labels = dataset[1]
        
    #**********
    def load_config(self, cfg_name=None):
        '''
        Function to load the configurations.
        '''
        
        pass
    

