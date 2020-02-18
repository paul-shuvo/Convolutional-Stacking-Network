import numpy as np
import os, configparser, ast, math
import _pickle as pickle
import tensorflow as tf
import logging


# *******************
class StackingConvNet:
    # **********
    def __init__(self, cfg_name, config_section="DEFAULT"):
        """
        Constructor function.
        :param cfg_name: Configuration file string
        :param config_section: Configuration section string
        """

        # Load configurations
        self.load_config(cfg_name, config_section)

        #Check

        # Set some initial values
        self.train_feature_extractors = None
        self.extract_features = None
        self.patchLoadFailed = False
        self.featureMap_patches = None

        # Decide on what components to use for the relevant feature extractors
        if len(self.cfg["components"]) == 0:
            self.components_in_layers = [range(i) for i in self.cfg["n_feature_maps"]]
        elif self.cfg["n_feature_maps"] == [len(i) for i in self.cfg["components"]]:
            self.components_in_layers = self.cfg["components"]
            self.components_in_layers = [[elem - 1 for elem in i] for i in self.components_in_layers]
        else:
            raise ValueError('The input components in the config file does not match the input n_feature_maps.')

        # Try to load feature extractors
        model_address = os.path.join('./Model/', self.cfg["convolutional_model_filename"] + '.pckl')
        if os.path.exists(model_address):
            with open(model_address, 'rb') as model_file:
                self.feature_extractors = pickle.load(model_file)
        else:
            self.feature_extractors = None

        # Try to load network settings
        settings_address = os.path.join('./Model/', self.cfg["convolutional_network_settings_filename"] + '.pckl')
        if os.path.exists(settings_address):
            with open(settings_address, 'rb') as settings_file:
                self.network_settings = pickle.load(settings_file)
        else:
            self.network_settings = None

        # Set Tensorflow's verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # WARN
        logging.getLogger('tensorflow').setLevel(logging.WARN)

    # **********
    def train(self):
        """
        Function to train the Stacking Convolutional Network.
        """

        # Load the input dataset
        self.train_samples, self.train_labels, self.test_samples, self.test_labels = self.load_dataset(
            dataset_name=self.cfg["dataset_name"],
            n_samples=self.cfg["n_samples"],
            test_set_size=self.cfg["test_set_size"])

        # Train kernels
        self.feature_extractors = self.train_conv_net(n_feature_maps=self.cfg["n_feature_maps"],
                                                      input_features=self.train_samples,
                                                      kernel_sizes=self.cfg["kernel_sizes"],
                                                      stride=self.cfg["stride"],
                                                      pooling_stride=self.cfg["pooling_stride"],
                                                      batch_size=self.cfg["batch_size"],
                                                      kernel_mode=self.cfg["kernel_mode"],
                                                      zero_pad=self.cfg["zero_pad"],
                                                      components=self.components_in_layers,
                                                      feature_extractor_types=self.cfg["feature_extractor_types"],
                                                      save_patches=self.cfg["save_patches"])

        # Construct the network settings dictionary
        self.network_settings = {'stride': self.cfg["stride"],
                                 'pooling_stride': self.cfg["pooling_stride"],
                                 'kernel_sizes': self.cfg["kernel_sizes"],
                                 'zero_pad': self.cfg["zero_pad"],
                                 'feature_extractor_types': self.cfg["feature_extractor_types"]}

        # Save the kernels to a file
        with open('./Model/' + self.cfg["convolutional_model_filename"] + '.pckl', 'wb') as feature_extractor_file:
            pickle.dump(self.feature_extractors, feature_extractor_file)

        # Save the network settings to a file
        with open('./Model/' + self.cfg["convolutional_network_settings_filename"] + '.pckl',
                  'wb') as network_settings_file:
            pickle.dump(self.network_settings, network_settings_file)

        # Save the raw test data
        self.save_dataset(self.test_samples, self.test_labels, './Datasets/Test_Data/Test_Set')

        # Get the output feature maps for the training and test data
        training_features = self.infer(images=self.train_samples,
                                       feature_extractors=self.feature_extractors,
                                       network_settings=self.network_settings,
                                       kernel_mode=self.cfg["kernel_mode"],
                                       components=self.components_in_layers)
        test_features = self.infer(images=self.test_samples,
                                   feature_extractors=self.feature_extractors,
                                   network_settings=self.network_settings,
                                   kernel_mode=self.cfg["kernel_mode"],
                                   components=self.components_in_layers)

        # Save datasets with labels as output and obtained features as input
        self.save_dataset(training_features, self.train_labels, './Datasets/Converted_Datasets/Converted_Training_Set')
        self.save_dataset(test_features, self.test_labels, './Datasets/Converted_Datasets/Converted_Test_Set')

    # **********
    def infer(self, images, feature_extractors, network_settings, kernel_mode, components):
        """
        Function to compute features of an image.
        :param images: Input images with the same height and width as in the training [batch, height, width]
        :param feature_extractors: List of lists of feature extractors/kernels
        (e.g. [[feature extractors/kernels of layer 1], [feature extractors/kernels of layer 2]])
        :param network_settings: Settings of the network architecture
        :param kernel_mode: If enabled kernels are used instead of feature extractors
        :param components: List of lists of components to be used in each layer [ [components in each layer] for number of layers]
        :return: Computed feature vector (1D numpy array)
        """

        # Check if the feature map function is decided
        assert self.extract_features is not None, 'Feature extractor function is not defined.'

        # Check feature extractors and network settings are loaded
        assert feature_extractors is not None, 'Model file is not loaded or network is not trained.'
        assert network_settings is not None, 'Network settings are not loaded or network is not trained.'

        # Add an extra batch dimension if there is no batch dimension in the input image
        if len(images.shape) < 3:
            images = np.expand_dims(images, axis=0)

        # Add a channels dimension to the input image
        # Current image shape is [channel, batch, height, width]
        images = np.expand_dims(images, axis=0)

        # Set current input features to the function input
        current_input = images

        # Print a message to indicate start of the inference operation
        print('\n___\nStarting inference in stacking convolutional network.')

        # Iterate over the layers of the network
        for layer, layer_feature_extractors in enumerate(feature_extractors):
            # Print a message
            print('***\nStarting layer ' + str(layer) + '. Input shape: ' + str(current_input.shape))

            # Iterate over the channels in the current layer
            first_channel = True
            kernel_size_no = 0
            for channel, channel_feature_extractors in enumerate(layer_feature_extractors):
                # Get next layer's extracted features if not in kernel mode
                if not kernel_mode:
                    # Update the index of kernel sizes list
                    if channel >= current_input.shape[0] * (kernel_size_no + 1):
                        kernel_size_no += 1

                    # Get patches of the previous layer
                    patches_of_kernel = self.get_patches(
                        kernel_size=network_settings['kernel_sizes'][layer][kernel_size_no],
                        stride=network_settings['stride'][layer],
                        features_maps=current_input[channel],
                        zero_pad=network_settings['zero_pad'],
                        training_mode=False)

                    extracted_features = self.extract_features[network_settings['feature_extractor_types'][layer]](
                        model=channel_feature_extractors,
                        feature_patches=patches_of_kernel,
                        components=components[layer],
                        zero_pad=network_settings['zero_pad'],
                        feature_map_shape=current_input[channel].shape,
                        stride=network_settings['stride'][layer])

                # Prepare data shapes for processing with Tensorflow
                if kernel_mode:
                    current_feature_maps_expanded = np.expand_dims(current_input[channel], axis=3)
                    filter_ = np.expand_dims(channel_feature_extractors, axis=2)
                else:
                    current_feature_maps_expanded = extracted_features
                    filter_ = None

                # Obtain the new feature maps by convolution (if kernel mode enabled) and pooling
                layer_temp = self.conv_max(input_batch=current_feature_maps_expanded,
                                           kernel_mode=kernel_mode,
                                           stride=network_settings['stride'][layer],
                                           pooling_stride=network_settings['pooling_stride'][layer],
                                           zero_pad=network_settings['zero_pad'],
                                           filter_=filter_,
                                           print_messages=False)

                # Reshape the 4D array to the desired 3D
                # ([batch, height, width, feature_channels] to [feature_channels, batch, height, width])
                for dim_counter in range(len(layer_temp.shape) - 1, 0, -1):
                    layer_temp = np.swapaxes(np.array(layer_temp), dim_counter - 1, dim_counter)

                # Add the layer obtained for the current kernel sizes to the list of current layers
                if first_channel:
                    current_layer = np.zeros((0,) + layer_temp.shape[1:])
                    first_channel = False
                current_layer = np.append(current_layer, layer_temp, axis=0)
                del layer_temp

            # Set the current layer as the next input
            current_input = current_layer

        # Make a 2D array out of the computed feature maps [batch, 1D feature vector]
        print('\nInference in stacking convolutional network done.\nOutput shape before flattening: ' + str(
            current_input.shape))
        current_input = np.swapaxes(current_input, 0, 1)
        feature_vector = np.reshape(current_input, (
        current_input.shape[0], current_input.shape[1] * current_input.shape[2] * current_input.shape[3]))

        # Print a message to report the output shape
        print('Output shape: ' + str(feature_vector.shape))

        return feature_vector

    # **********
    def train_conv_net(self, n_feature_maps, input_features, kernel_sizes, stride, pooling_stride, batch_size,
                       kernel_mode, zero_pad, components, feature_extractor_types, save_patches):
        """
        Function to get feature extractors trained from input feature maps/images.
        :param n_feature_maps: [number of features in each layer]
        :param input_features: [batch size, height, width]
        :param kernel_sizes: List of list of lists
        (e.g. [[[kernel size #1 for x and y in layer 1], [Kernel sizes #2 in layer 1]], [[Kernel sizes #1 in layer 2]] , [Kernel sizes #2 in layer 2]]]])
        :param stride: List of convolution strides for each layer
        :param pooling_stride: List of pooling strides
        :param batch_size: Size of the batch in operations that divide the whole training set (like pooling)
        :param kernel_mode: If enabled kernels are saved and processed instead of feature extractors
        :param zero_pad: Boolean parameter to indicate if feature maps should be zero-padded
        :param components: List of lists of components to be used in each layer [ [components in each layer] for number of layers]
        :param feature_extractor_types: List of string that defines the type of feature extractors (PCA, ICA, etc.) in each layer
        :param save_patches: Boolean variable to enable saving patches on disk
        :return: List of lists of feature extractors/kernels
        (e.g. [[feature extractors/kernels of layer 1], [feature extractors/kernels of layer 2]])
        """

        # Check if the feature map function is decided
        assert self.train_feature_extractors is not None, 'Feature extractor function is not defined.'

        # Add an extra batch dimension if there is no batch dimension in the input data
        if len(input_features.shape) < 3:
            input_features = np.expand_dims(input_features, axis=0)

        # Add a channels dimension to the input feature matrix
        input_features = np.expand_dims(input_features, axis=0)

        # Set current input features to the function input
        # Current input has the shape [channel, batch, height, width]
        current_input = input_features

        # Iterate over the layers of the network
        feature_extractors = []
        for layer, n_current_featureMaps in enumerate(n_feature_maps):
            # Print a message
            print('\n*****\nStarting layer ' + str(layer + 1) + '. Input shape: ' + str(current_input.shape))

            # Compute the new feature maps from the current feature maps
            current_feature_extractors = []
            first_channel = True
            for kernelSize_no, kernelSize in enumerate(kernel_sizes[layer]):
                for channel in range(current_input.shape[0]):
                    # Get patches of the previous layer
                    patches_of_kernel = self.get_patches(kernel_size=kernelSize,
                                                         stride=stride[layer],
                                                         features_maps=current_input[channel],
                                                         zero_pad=zero_pad,
                                                         training_mode=True,
                                                         save_patches=self.cfg['save_patches'],
                                                         layer_no=layer)
                    print('++\nPatches generated. Shape: ' + str(patches_of_kernel.shape))

                    # Train feature reduction
                    if kernel_mode:  # Get kernels
                        kernels_temp = self.train_feature_extractors[feature_extractor_types[layer]](
                            components=components[layer],
                            feature_patches=patches_of_kernel,
                            kernel_mode=kernel_mode,
                            zero_pad=zero_pad)
                        current_feature_extractors.append(kernels_temp)
                        print('Feature maps of kernel size #' + str(kernelSize_no + 1) + ' computed. Shape: ' +
                              str(kernels_temp.shape))

                    else:  # Get feature extractors and extracted features
                        feature_extractors_temp, extracted_features = self.train_feature_extractors[feature_extractor_types[layer]](
                            components=components[layer],
                            feature_patches=patches_of_kernel,
                            kernel_mode=kernel_mode,
                            zero_pad=zero_pad,
                            feature_map_shape=current_input[channel].shape,
                            stride=stride[layer])
                        current_feature_extractors.append(feature_extractors_temp)
                        print('Features extracted. Shape: ', str(extracted_features.shape))

                    # Delete the patches to free up memory
                    del patches_of_kernel

                    # Obtain the new feature maps by convolution (if kernel mode enabled) and pooling
                    # Prepare data shapes for processing with Tensorflow
                    if kernel_mode:
                        current_feature_maps_expanded = np.expand_dims(current_input[channel], axis=3)
                        filter_ = np.expand_dims(kernels_temp, axis=2)
                    else:
                        current_feature_maps_expanded = extracted_features
                        filter_ = None

                    # Divide the batch to make it feasible to run with Tensorflow (prevent GPU memory overflow)
                    tf_batch_size = batch_size
                    layer_split = []
                    batch_left = current_feature_maps_expanded.shape[0]
                    while batch_left > 0:
                        first_index = current_feature_maps_expanded.shape[0] - batch_left
                        layer_split.append(
                            self.conv_max(
                                input_batch=current_feature_maps_expanded[first_index: first_index + tf_batch_size],
                                kernel_mode=kernel_mode,
                                stride=stride[layer],
                                pooling_stride=pooling_stride[layer],
                                zero_pad=zero_pad,
                                filter_=filter_))
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
            feature_extractors.append(current_feature_extractors)

        return feature_extractors

    # **********
    def get_patches(self, kernel_size, stride, features_maps, zero_pad, training_mode, save_patches=False, layer_no=None):
        """
        Function to gather patches by scanning kernels over input image/feature maps for all training samples.
        :param kernel_size: Size of kernel [x, y]
        :param stride: An integer to represent stride in both x and y directions
        :param features_maps: Batch of feature maps  to get their patches [batch, height, width]
        :param zero_pad: Boolean parameter to indicate if feature maps should be zero-padded before patch extraction
        :param training_mode: Boolean variable to enable training time actions
        :param save_patches: Boolean variable to enable saving patches on disk
        :param layer_no: Layer number (integer)
        :return: stack of patches of feature maps [patches_per_featureMap * batch, height, width]
        """

        # Extract the size in each dimension
        kernel_dim1, kernel_dim2 = kernel_size

        if training_mode:
            # Iterate over all kernel sizes
            generate_patches = False

            # Set the address to load/save patches
            if zero_pad:
                padding_extension = '_padded'
            else:
                padding_extension = ''
            address = './Patches/layer' + str(layer_no) + '_' + str(kernel_dim1) + "x" + str(kernel_dim2) + '_input-' \
                      + str(features_maps.shape[0]) + '-' + str(features_maps.shape[1]) + '-' + \
                      str(features_maps.shape[2]) + padding_extension + '.npy'

            # Load extracted patches if they exist and enabled (without any failure in loading so far)
            if self.cfg["use_extracted_patches"] and (not self.patchLoadFailed):
                if os.path.exists(address):
                    self.featureMap_patches = np.load(address)
                else:
                    self.patchLoadFailed = True
                    generate_patches = True
            else:
                generate_patches = True
        else:
            # If not in training mode, always generate patches (do not look for saved patches on files)
            generate_patches = True

        if generate_patches:
            # Zero pad if enabled
            if zero_pad:
                features_maps = np.pad(array=features_maps,
                                       pad_width=((0, 0), (int(kernel_dim1 / 2), int(kernel_dim1 / 2)),
                                                  (int(kernel_dim2 / 2), int(kernel_dim2 / 2))),
                                       constant_values=0)

            # Scan all the possible patch positions and add them to a numpy array
            self.featureMap_patches = np.zeros((0, kernel_dim1, kernel_dim2))
            range_dim1 = features_maps.shape[1] - kernel_dim1 + 1
            range_dim2 = features_maps.shape[2] - kernel_dim2 + 1
            for dim1 in range(0, range_dim1, stride):
                for dim2 in range(0, range_dim2, stride):
                    current_patch = features_maps[:, dim1: dim1 + kernel_dim1, dim2: dim2 + kernel_dim2]
                    self.featureMap_patches = np.append(self.featureMap_patches, current_patch, axis=0)

            # Write the current patches on a file
            if training_mode and save_patches:
                np.save(address, self.featureMap_patches)

        return self.featureMap_patches

    # *******************
    def conv_max(self, input_batch, kernel_mode, stride, pooling_stride, zero_pad, filter_=None, print_messages=True):
        """
        Function to convolve images with kernels and perform max pooling on them.
        :param input_batch: Batch of input feature maps [batch, height, width, channels]
        :param kernel_mode: If enabled input filters are used to convolve with the input feature maps
        :param stride: Convolution stride (integer)
        :param pooling_stride: Pooling stride (integer)
        :param zero_pad: Boolean parameter to indicate if feature maps should be zero-padded during convolution
        :param filter_: A filter to convolve images with [height, width, in_channels, out_channels]
        :param print_messages: Boolean variable to enable printing messages after each step
        :return: Tensor of size [batch, height, width, channels]
        """

        # Determine padding type
        if zero_pad:
            padding = 'SAME'
        else:
            padding = 'VALID'

        # Convolve input feature maps with the input filter to create new feature maps
        if kernel_mode:
            processed_batch = tf.nn.conv2d(input=input_batch, filters=filter_, strides=stride, padding=padding)
            if print_messages:
                print('Convolution done. Shape: ' + str(processed_batch.shape))
        else:
            processed_batch = input_batch

        # Do non-overlapping 2D max pooling
        if pooling_stride != 0:
            processed_batch = tf.nn.max_pool(input=processed_batch, ksize=pooling_stride, strides=pooling_stride,
                                             padding='SAME')
            if print_messages:
                print('Max pooling done. Shape: ' + str(processed_batch.shape))

        return processed_batch

    # **********
    def save_dataset(self, samples, labels, path):
        """
        Function to save the input samples and corresponding labels in a file
        :param samples: Input samples (numpy array)
        :param labels: Labels (numpy array)
        :param path: The complete path, including the filename, but without any extension, of the file to be created
        """

        # Create the list structure of a dataset
        dataset = [samples, labels]

        # Save the dataset file in a pickle file
        with open(path + '.pckl', 'wb') as dataset_file:
            pickle.dump(dataset, dataset_file)

    # **********
    def load_dataset(self, dataset_name, n_samples, test_set_size):
        """
        Function to load a dataset from a file.
        :param dataset_name: Dataset name string
        :param n_samples: Number of samples to be used for training and testing
        :param test_set_size: Number of test samples in the dataset.
        :return: Returns training samples, training labels, test samples, test labels
        """

        # Create the address to the dataset file
        filepath = os.path.join('./Datasets/', dataset_name + '.pckl')

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

        # Check if dataset contains more than just one image
        if (len(dataset[0].shape) < 3) or (dataset[0].shape[0] <= 1) or (dataset[1].shape[0] <= 1):
            raise Exception('Dataset contains only one image.')

        # Check if the number of samples is equal to the number of labels
        assert dataset[0].shape[0] == dataset[1].shape[0], \
            'Mismatch in the number of samples and labels in the dataset.'

        # Check if the requested test set size is valid
        assert 0 <= test_set_size < dataset[1].shape[0], "Test set size of " + test_set_size + " not valid"

        # Separate training and test data
        dataset[0] = dataset[0][: n_samples]
        dataset[1] = dataset[1][: n_samples]
        training_data_no = int(dataset[1].shape[0] - test_set_size)
        training_samples = dataset[0][: training_data_no]
        training_labels = dataset[1][: training_data_no]
        test_samples = dataset[0][training_data_no:]
        test_labels = dataset[1][training_data_no:]

        return training_samples, training_labels, test_samples, test_labels

    # **********
    def check_matching_config(self):
        """
        Function to check if tyhe input configuration match together.
        """

        # Check if the number of requested components is equal to the list of components for each layer
        assert (len(self.cfg['components']) == 0) or (self.cfg['n_feature_maps'] == [len(layer_components) for layer_components in self.cfg['components']]), \
            'Function train_conv_net requires the input options n_feature_maps and components match.'

        # Check the number of layers are the same in the input configurations
        assert len(self.cfg['n_feature_maps']) == len(self.cfg['kernel_sizes']) == len(self.cfg['stride']) == len(self.cfg['pooling_stride']) == len(self.cfg['feature_extractor_types']), \
            'The following input configurations must have the equal length:\n- n_feature_maps\n- kernel_sizes\n- stride\n- pooling_stride\n- feature_extractor_types'

    # **********
    def load_config(self, config_file, config_section="DEFAULT"):
        """
        Function to load the configurations from a file.
        :param config_file: Configuration file string
        :param config_section: Configuration section string
        """

        # Create a configuration dictionary
        self.cfg = dict()

        # Load config parser
        config_parser = configparser.ConfigParser()

        # Read the config file
        list_of_files_read = config_parser.read(os.path.join('./Config/', config_file))

        # Make sure at least a configuration file was read
        if len(list_of_files_read) <= 0:
            raise Exception("\nFatal Error: No configuration file " + config_file + " found in ./Config/.\n")

        # Load all the necessary configurations
        self.cfg["stride"] = ast.literal_eval(config_parser.get(config_section, "stride"))
        self.cfg["kernel_sizes"] = ast.literal_eval(config_parser.get(config_section, "kernel_sizes"))
        self.cfg["dataset_name"] = config_parser.get(config_section, "dataset_name")
        self.cfg["n_feature_maps"] = ast.literal_eval(config_parser.get(config_section, "n_feature_maps"))
        self.cfg["use_extracted_patches"] = config_parser.getboolean(config_section, "use_extracted_patches")
        self.cfg["pooling_stride"] = ast.literal_eval(config_parser.get(config_section, "pooling_stride"))
        self.cfg["batch_size"] = config_parser.getint(config_section, "batch_size")
        self.cfg["n_samples"] = config_parser.getint(config_section, "n_samples")
        self.cfg["kernel_mode"] = config_parser.getboolean(config_section, "kernel_mode")
        self.cfg["kernelPCA_kernel_type"] = config_parser.get(config_section, "kernelPCA_kernel_type")
        self.cfg["max_iteration_ICA"] = config_parser.getint(config_section, "max_iteration_ICA")
        self.cfg["max_iteration_FA"] = config_parser.getint(config_section, "max_iteration_FA")
        self.cfg["zero_pad"] = config_parser.getboolean(config_section, "zero_pad")
        self.cfg["convolutional_model_filename"] = config_parser.get(config_section, "convolutional_model_filename")
        self.cfg["convolutional_network_settings_filename"] = config_parser.get(config_section,
                                                                                "convolutional_network_settings_filename")
        self.cfg["test_set_size"] = config_parser.getfloat(config_section, "test_set_size")
        self.cfg["components"] = ast.literal_eval(config_parser.get(config_section, "components"))
        self.cfg["feature_extractor_types"] = ast.literal_eval(config_parser.get(config_section, "feature_extractor_types"))
        self.cfg["save_patches"] = config_parser.getboolean(config_section, "save_patches")

        # Check if the input configuration match together
        self.check_matching_config()
