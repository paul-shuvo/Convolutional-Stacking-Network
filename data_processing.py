import numpy as np
import _pickle as pickle
import gc
import os
import datetime

class DataProcessing:
    def __init__(self, data=None):
        self.Data = data

    def load_data_from_directory(self, filepath):
        f = open(filepath, 'rb')
        # disable garbage collector
        gc.disable()
        self.Data = pickle.load(f)
        # enable garbage collector again
        gc.enable()
        f.close()
        return self.Data


    def save_data(self, data=None, filepath=None):
        if data is None:
            data = self.Data
        if filepath is None:
            filename = "data_" + str(datetime.datetime.now()) + ".pckl"
            filename = filename.replace(":", "-")
            filename = filename.replace(" ", "_")
            filepath = os.path.join(os.getcwd(), filename)

        f = open(filepath, 'wb')
        pickle.dump(data, f)
        f.close()

    def get_image_patches(self, windows, x = None, shape_= None):

        if x is None:
            x = self.Data[0]

        dim = x.shape
        if len(dim) == 2:
            shape_ = (dim[0], int(np.sqrt(dim[1])), -1)
            x = np.reshape(x, shape_)

        pool_of_patches = {}
        for window in windows:
            m,n = window
            key = str(m) + "x" + str(n)
            patches = np.zeros((0, m, n))
            range_column = x.shape[2] - m + 1
            range_row = x.shape[1] - n + 1
            for row in range(0, range_row):
                for column in range(0, range_column):
                    patch = x[:, column : column + m, row : row + n]
                    patches = np.append(patches, patch, axis=0)
                print(row)
                print(patches.shape)

            # pool_of_patches[key] = patches
            # self.save_data(patches, os.path.join(os.getcwd(), key))
            np.save(os.path.join(os.getcwd(), key), patches)
            del patch

        return pool_of_patches

    def convert_to_2d_array(self, data = None):
        if data is None:
            data = self.Data

