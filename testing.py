import numpy as np
import os
from data_processing import DataProcessing
from ica_net import ICANet


cwd = os.getcwd()
folder = "Dataset"
file_name = 'Fashion-MNIST.pckl'
# f = open(os.path.join(cwd, folder, file_name), 'wb')
# pickle.dump([X, y], f)
# f.close()
# dp = DataProcessing()
# Data = dp.load_data_from_directory(os.path.join(cwd, folder, file_name))
# dp.Data = Data[0][0:60000, :]
# f = open(, 'rb')
# obj = pickle.load(f)
# f.close()
# X, y = Data
# windows = [[7,7]]
# s = dp.get_image_patches(windows, Data[0][0:40000, :])
# dp.save_data(s)
filename = "5x5.npy"
datum = np.load(os.path.join(cwd,filename))
sh = datum.shape
datum = np.reshape(datum,(sh[0], -1) )
ica_n = ICANet(datum)
pca_layers = ica_n.get_feature_maps([5], datum)

l = 1