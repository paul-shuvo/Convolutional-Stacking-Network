
# Convolutional Stacking Network (CSN)
A deep convolutional network that consists of layers of stacked feature extractors. The feature extractors are not necessarily neural network.

## Description
This network functions similar to a convolutional neural network (CNN) with the difference that the layers are not just made of neural network and all the layers are not trained in repetitions of holistic backward propagation of errors (backpropagation). Instead, feature extractors are trained layer by layer from the input toward the later layers. The convolution operation, however, is similar to the conventional neural networks, with the exception that other types of feature extractors, such as independent component analysis (ICA) and principal component analysis (PCA), are used instead. The convolutional network is followed by a classifier or a regressor as is common. Currently, we are using a fully connected neural network for that.

More detailed descriptions are added after the pending papers are published.

## The code
The Convolutional Stacking Network is written in Python, by utilizing the Tensorflow, PyTorch, and Scikit-learn libraries.

The source code is placed under the *src* folder. To train and subsequently test a network, run the *Run.py* file. The configuration files are located under the *Config* directory. To set the properties of the network, training, and testing, modify the configuration files accordingly. The datasets can be placed under the *Datasets* folder. The extracted test data from the original dataset are saved under the *Datasets/Test_Data* directory. The trained models are saved in the *Model* folder. The *Patches* folder can be used to store the image patches during the internal sliding of the kernels over the feature maps.

# Developers
[Pourya Hoseini](https://github.com/pouryahoseini) and [Shuvo Paul](https://github.com/paul-shuvo) (equal contribution)

# License
Copyright 2019 - 2020, Pourya Hoseini, Shuvo Paul, and the Convolutional Stacking Network (CSN) contributors. Any usage must be with the permission of the authors.

# Contact
We can be reached at the following email addresses:
- Pourya Hoseini: [hoseini@nevada.unr.edu](mailto:hoseini@nevada.unr.edu)
- Shuvo Paul: [shuvo.k.paul@nevada.unr.edu](mailto:shuvo.k.paul@nevada.unr.edu)
