import numpy as np
import os, configparser, ast, math
import _pickle as pickle
import cv2 as cv
import tensorflow as tf
import logging


# *******************
def conv_max(input_batch, filter_, stride, pooling_stride):
    """
    Function to convolve images with kernels and perform max pooling on them.
    :param input_batch: Batch of input feature maps [batch, height, width, channels]
    :param filter_: A filter to convolve images with [height, width, in_channels, out_channels]
    :param stride: Convolution stride (integer)
    :param pooling_stride: Pooling stride (integer)
    :return: Tensor of size [batch, height, width, channels]
    """
    processed_batch = tf.nn.conv2d(input=input_batch, filters=filter_, strides=stride, padding='SAME')
    print('Convolution done. Shape: ' + str(processed_batch.shape))

    # Do non-overlapping 2D max pooling
    processed_batch = tf.nn.max_pool(input=processed_batch, ksize=pooling_stride, strides=pooling_stride, padding='SAME')
    print('Max pooling done. Shape: ' + str(processed_batch.shape))

    return processed_batch

# *******************
class StackingConvNet:
    # **********
    def __init__(self, cfg_name, configSection="DEFAULT"):
        """
        Constructor function.
        :param cfg_name: Configuration file string
        :param configSection: Configuration section string
        """

        # Load configurations
        self.load_config(cfg_name, configSection)

        # Load the input dataset
        self.load_dataset(self.cfg["dataset_name"])

        # Set some initial values
        self.get_featureMaps = None
        self.patchLoadFailed = False
        self.featureMap_patches = None

        # Set Tensorflow's verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # WARN
        logging.getLogger('tensorflow').setLevel(logging.WARN)

    # **********
    def train(self):
        """
        Function to train the Stacking Convolutional Network.
        """

        # Train kernels
        kernels = self.train_kernels(self.cfg["n_feature_maps"], self.Data, self.cfg["kernel_sizes"],
                                     self.cfg["stride"], self.cfg["pooling_stride"])

        # Save the kernels to a file
        with open('./Model/Kernels.pckl', 'wb') as kernel_file:
            pickle.dump(kernels, kernel_file)

    # **********
    def train_kernels(self, n_feature_maps, input_features, kernel_sizes, stride, pooling_stride, n_samples=None):
        """
        Function to get kernels computed form input feature maps/images.
        :param n_feature_maps: [number of features in each layer]
        :param input_features: [batch size, height, width]
        :param kernel_sizes: List of list of lists (e.g. [[[kernel size #1 for x and y in layer 1], [Kernel sizes #2 in layer 1]], [[Kernel sizes #1 in layer 2]] , [Kernel sizes #2 in layer 2]]]])
        :param stride: List of convolution strides for each layer
        :param pooling_stride: List of pooling strides
        :param n_samples: Integer to specify number of samples used in the training
        :return: List of lists of kernels (e.g. [[kernels of layer 1], [kernels of layer 2]])
        """

        # Check if the feature map function is decided
        assert self.get_featureMaps is not None

        # Check number of samples for the training
        if n_samples is None:
            n_samples = input_features.shape[0]

        # Add an extra batch dimension if there is no batch dimension in the input data
        if len(input_features.shape) < 3:
            input_features = np.expand_dims(input_features, axis=0)

        # Add a channels dimension to the input feature matrix
        input_features = np.expand_dims(input_features, axis=0)

        # Set current input features to the function input
        # Current input has the shape [channel, training_batch, height, width]
        current_input = input_features[:, : n_samples, :, :]

        # Iterate over the layers of the network
        kernels = []
        for layer, n_current_featureMaps in enumerate(n_feature_maps):
            # Print a message
            print('\n*****\nStarting layer ' + str(layer + 1) + '. Input shape: ' + str(current_input.shape))

            # Compute the new feature maps from the current feature maps
            current_kernels = []
            first_channel = True
            for kernelSize_no, kernelSize in enumerate(kernel_sizes[layer]):
                for channel in range(current_input.shape[0]):
                    # Get patches of the previous layer
                    patches_of_kernel = self.get_patches(kernelSize, stride[layer], current_input[channel], layer)[:(input_features.shape[1] - kernelSize[0] + 1) * (input_features.shape[2] - kernelSize[1] + 1) * n_samples, :,:]
                    print('Patches generated. Shape: ' + str(patches_of_kernel.shape))

                    # Get kernels first
                    kernels_temp = self.get_featureMaps(n_feature_maps[layer], patches_of_kernel)
                    current_kernels.append(kernels_temp)
                    print('Feature maps of kernel size #' + str(kernelSize_no + 1) + ' computed. Shape: ' + str(kernels_temp.shape))

                    # Delete the patches to free up memory
                    del patches_of_kernel

                    # Convolve the input feature maps with the kernels to obtain the new feature maps
                    current_input_expanded = np.expand_dims(current_input[channel], axis=4)
                    filter_ = np.expand_dims(kernels_temp, axis=2)

                    # Divide the batch to make it feasible to run with Tensorflow (prevent GPU memory overflow)
                    tf_batch_size = 20000
                    layer_split = []
                    batch_left = current_input_expanded.shape[0]
                    while batch_left > 0:
                        first_index = current_input_expanded.shape[0] - batch_left
                        layer_split.append(conv_max(current_input_expanded[first_index: first_index + tf_batch_size], filter_, stride[layer], pooling_stride[layer]))
                        batch_left -= tf_batch_size
                    layer_temp = np.concatenate(layer_split, axis=0)
                    del layer_split

                    # Reshape the 4D array to the desired 3D
                    # ([batch, height, width, feature_channels] to [feature_channels, batch, height, width])
                    for dim_counter in range(len(layer_temp.shape) - 1, 0, -1):
                        layer_temp = np.swapaxes(np.array(layer_temp), dim_counter - 1, dim_counter)

                    # Add the layer obtained for the current kernel sizes to the list of current layers
                    if first_channel:
                        current_layer = np.zeros((0,) + layer_temp.shape[1:])
                        first_channel = False
                    current_layer = np.append(current_layer, layer_temp, axis=0)

                    print('New channels created. Shape: ', str(layer_temp.shape))
                    del layer_temp

            # Set the current layer as the next input
            current_input = current_layer
            print('Layer generation done. Shape: ', str(current_layer.shape))

            # Save kernels of this layer
            kernels.append(current_kernels)

        return kernels

    # **********
    def get_patches(self, kernel_size, stride, features_maps, layer_no):
        """
        Function to gather patches by scanning kernels over input image/feature maps for all training samples.
        :param kernel_size: Size of kernel [x, y]
        :param stride: An integer to represent stride in both x and y directions
        :param features_maps: Batch of feature maps  to get their patches [batch, height, width]
        :param layer_no: Layer number (integer)
        :return: stack of patches of feature maps [patches_per_featureMap * batch, height, width]
        """

        # Iterate over all kernel sizes
        generate_patches = False

        # Extract the size in each dimension
        kernel_dim1, kernel_dim2 = kernel_size

        # Load extracted patches if they exist and enabled (without any failure in loading so far)
        if self.cfg["use_extracted_patches"] and (not self.patchLoadFailed):
            address = './Patches/layer' + str(layer_no) + '_' + str(kernel_dim1) + "x" + str(kernel_dim2) + '.npy'
            if os.path.exists(address):
                self.featureMap_patches = np.load(address)
            else:
                self.patchLoadFailed = True
                generate_patches = True
        else:
            generate_patches = True

        if generate_patches:
            # Scan all the possible patch positions and add them to a numpy array
            self.featureMap_patches = np.zeros((0, kernel_dim1, kernel_dim2))
            range_dim1 = features_maps.shape[1] - kernel_dim1 + 1
            range_dim2 = features_maps.shape[2] - kernel_dim2 + 1
            for dim1 in range(0, range_dim1, stride):
                for dim2 in range(0, range_dim2, stride):
                    current_patch = features_maps[:, dim1: dim1 + kernel_dim1, dim2: dim2 + kernel_dim2]
                    self.featureMap_patches = np.append(self.featureMap_patches, current_patch, axis=0)

            # Write the current patches on a file
            np.save('./Patches/layer' + str(layer_no) + '_' + str(kernel_dim1) + "x" + str(kernel_dim2),
                    self.featureMap_patches)

        return self.featureMap_patches

    # **********
    def load_dataset(self, datsetName):
        """
        Function to load a dataset from a file.
        :param datsetName: Dataset name string
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
        dataset[0] = np.reshape(dataset[0], (
        dataset[0].shape[0], int(math.sqrt(dataset[0].shape[1])), int(math.sqrt(dataset[0].shape[1]))))
        self.Data = dataset[0]
        self.Labels = dataset[1]

    # **********
    def load_config(self, configFile, configSection="DEFAULT"):
        """
        Function to load the configurations from a file.
        :param configFile: Configuration file string
        :param configSection: Configuration section string
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
        self.cfg["stride"] = ast.literal_eval(config_parser.get(configSection, "stride"))
        self.cfg["kernel_sizes"] = ast.literal_eval(config_parser.get(configSection, "kernel_sizes"))
        self.cfg["dataset_name"] = config_parser.get(configSection, "dataset_name")
        self.cfg["n_feature_maps"] = ast.literal_eval(config_parser.get(configSection, "n_feature_maps"))
        self.cfg["use_extracted_patches"] = config_parser.getboolean(configSection, "use_extracted_patches")
        self.cfg["pooling_stride"] = ast.literal_eval(config_parser.get(configSection, "pooling_stride"))
