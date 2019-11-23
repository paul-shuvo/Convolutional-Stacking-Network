import numpy as np
import os, configparser, ast, math
import _pickle as pickle
import cv2 as cv
import tensorflow as tf


class StackingConvNet:
    # **********
    def __init__(self, cfg_name, configSection="DEFAULT"):
        """
        Constructor function.
        """

        # Load configurations
        self.load_config(cfg_name, configSection)

        # Load the input dataset
        self.load_dataset(self.cfg["dataset_name"])

        # Set some initial values
        self.get_featureMaps = None
        self.patchLoadFailed = False

    # **********
    def train(self):
        """
        Function to train the Stacking Convolutional Network.
        """

        # Train kernels
        kernels = self.train_kernels(self.cfg["n_feature_maps"], self.Data, self.cfg["kernel_sizes"],
                                     self.cfg["stride"])

        # Save the kernels to a file
        with open('./Model/Kernels.pckl', 'wb') as kernel_file:
            pickle.dump(kernels, kernel_file)

    # **********
    def train_kernels(self, n_feature_maps, input_features, kernel_sizes, stride):
        """
        Function to get kernels computed form input feature maps/images.
        """

        # Check if the feature map function is decided
        assert self.get_featureMaps is not None

        # Add an extra batch dimension if there is no batch dimension in the input data
        if len(input_features.shape) < 3:
            input_features = np.expand_dims(input_features, axis=0)

        # Set current input features to the function input
        current_input = input_features

        # Iterate over the layers of the network
        kernels = []
        for layer, n_current_featureMaps in enumerate(n_feature_maps):
            # Print a message
            print('\n*****\nStarting layer ' + str(layer + 1))

            # Get patches of the previous layer
            patches = self.get_patches(kernel_sizes[layer], stride, current_input, layer)
            print('Patches generated')

            # Compute the new feature maps from the current feature maps
            current_kernels = []
            current_layer = np.zeros((0,) + current_input.shape[1:])
            for kernelSize, patches_of_kernel in enumerate(patches):
                # Get kernels first
                kernels_temp = self.get_featureMaps(n_feature_maps[layer], patches_of_kernel)
                current_kernels.append(kernels_temp)
                print('Feature maps of kernel size #' + str(kernelSize + 1) + ' computed')

                # Convolve the input feature maps with the kernels to obtain the new feature maps
                current_input = np.expand_dims(current_input, axis=3)
                filter = np.expand_dims(kernels_temp, axis=2)
                layer_temp = tf.nn.conv2d(input=current_input, filters=filter, strides=stride, padding='SAME')
                print('Convolution done')

                # Do 2D max pooling
                layer_temp = tf.nn.max_pool(input=layer_temp, ksize=kernel_sizes[layer][kernelSize], strides=stride,
                                            padding='SAME')
                print('Max pooling done')

                # Reshape the 4D array to the desired 3D ([batch, height, width, feature_channels] to [batch * feature_channels, height, width])
                layer_temp = np.swapaxes(np.array(layer_temp), 2, 3)
                layer_temp = np.swapaxes(layer_temp, 1, 2)
                layer_temp = np.reshape(layer_temp, (layer_temp.shape[0] * layer_temp.shape[1], layer_temp.shape[2], layer_temp.shape[3]))

                # Add the layer obtained for the current kernel sizes to the list of current layers
                current_layer = np.append(current_layer, layer_temp, axis=0)

            # Set the current layer as the next input
            current_input = current_layer

            # Save kernels of this layer
            kernels.append(current_kernels)

        return kernels

    # **********
    def get_patches(self, kernel_sizes, stride, features_maps, layer_no):
        """
        Function to gather patches by scanning kernels over input image/feature maps for all training samples.
        """

        # If the input feature maps are each one 1D, convert them to 2D 
        dims = features_maps.shape
        if len(dims) == 2:
            shape_ = (dims[0], int(np.sqrt(dims[1])), -1)
            features_maps = np.reshape(features_maps, shape_)

        # Iterate over all kernel sizes
        generate_patches = False
        pool_of_patches = []
        for kernel in kernel_sizes:
            # Extract the size in each dimension
            kernel_dim1, kernel_dim2 = kernel

            # Load extracted patches if they exist and enabled (without any failure in loading so far)
            if self.cfg["use_extracted_patches"] and (not self.patchLoadFailed):
                address = './Patches/layer' + str(layer_no) + '_' + str(kernel_dim1) + "x" + str(kernel_dim2) + '.npy'
                if os.path.exists(address):
                    featureMap_patches = np.load(address)
                else:
                    self.patchLoadFailed = True
                    generate_patches = True
            else:
                generate_patches = True

            if generate_patches:
                # Scan all the possible patch positions and add them to a numpy array
                featureMap_patches = np.zeros((0, kernel_dim1, kernel_dim2))
                range_dim1 = features_maps.shape[1] - kernel_dim1 + 1
                range_dim2 = features_maps.shape[2] - kernel_dim2 + 1
                for dim1 in range(0, range_dim1, stride):
                    for dim2 in range(0, range_dim2, stride):
                        current_patch = features_maps[:, dim1: dim1 + kernel_dim1, dim2: dim2 + kernel_dim2]
                        featureMap_patches = np.append(featureMap_patches, current_patch, axis=0)

                # Write the current patches on a file
                np.save('./Patches/layer' + str(layer_no) + '_' + str(kernel_dim1) + "x" + str(kernel_dim2),
                        featureMap_patches)

            # Add the patches for the current kernel size to a list containing all patches
            pool_of_patches.append(featureMap_patches)

            # Delete the patches for the current kernel size
            del featureMap_patches

        return pool_of_patches

    # **********
    def load_dataset(self, datsetName):
        """
        Function to load a dataset from a file.
        """

        # Create the address to the dataset file
        filepath = os.path.join('./Datasets/', datsetName)

        # Check the file path
        if not os.path.exists(filepath):
            raise Exception('\nThe dataset file "' + filepath + '" does not exist.\n')

        # Open the file
        with open(filepath, 'rb') as dataset_file:
            # Load the dataset's pickle file
            dataset = pickle.load(dataset_file)

        # Separate input data and output labels
        dataset[0] = np.reshape(dataset[0], (dataset[0].shape[0], int(math.sqrt(dataset[0].shape[1])), int(math.sqrt(dataset[0].shape[1]))))
        self.Data = dataset[0]
        self.Labels = dataset[1]

    # **********
    def load_config(self, configFile, configSection="DEFAULT"):
        """
        Function to load the configurations from a file.
        """

        # Create a configuration dictionary
        self.cfg = dict()

        # Load config parser
        config_parser = configparser.ConfigParser()

        # Read the config file
        listOfFilesRead = config_parser.read(os.path.join('./Config/', configFile))

        # Make sure at least a configuration file was read
        if len(listOfFilesRead) <= 0:
            raise Exception("\nFatal Error: No configuration file " + configFile + " found in ./Config/.\n")

        # Load all the necessary configurations
        self.cfg["stride"] = config_parser.getint(configSection, "stride")
        self.cfg["kernel_sizes"] = ast.literal_eval(config_parser.get(configSection, "kernel_sizes"))
        self.cfg["dataset_name"] = config_parser.get(configSection, "dataset_name")
        self.cfg["n_feature_maps"] = ast.literal_eval(config_parser.get(configSection, "n_feature_maps"))
        self.cfg["use_extracted_patches"] = config_parser.getboolean(configSection, "use_extracted_patches")
