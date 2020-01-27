# TODO: add random split to the data_transform class
#       URL: https://discuss.pytorch.org/t/issues-with-torch-utils-data-random-split/22298
# TODO: Scale the original the data
# TODO: Add loading from checkpoint
# TODO: Add optimizer, loss, metrics to model class
# TODO: Add more comments
# FIXME: Fix some naming of files and classes
# FIXME: Fix the warning on the python file import i.e. 'train,py'

# import libraries
import torch
import torch.nn as nn
import numpy as np
import yaml
import _pickle as pickle
from torch.autograd import Variable
from torch.utils import data
from torch import optim
import os
from torchsummary import summary

# import from python files
import models
from train import Train

# set flags / seeds
# to keep experiments reproducible
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Following statement allows the cuda backend to optimize
# the graph during its first execution
torch.backends.cudnn.benchmark = True


# Start with main code
if __name__ == '__main__':
    # make sure we use gpu if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration of feature extractors
    # of PCA/ICA from 'config.yml' file
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        cfg = cfg['default']

    # Load the parameters from 'cfg'
    hyper_parameters = cfg['hyper_parameters']
    feature_extractor_net_cfg = cfg['feature_extractor_net_config']
    dataset_cfg = cfg['dataset']
    dataset_params = dataset_cfg['parameters']
    dataset_path = dataset_cfg['path']
    sampler_ratio = dataset_cfg['sampler_ratio']
    max_epochs = hyper_parameters['max_epoch']

    # Convert the data to a general structure before
    # feeding to the model. If 'train_transformed_data'
    # is true then use the transformed PCA/ICA data
    # else use the original dataset
    if cfg['train_transformed_data']:
        with open(dataset_path['transformed'][0], 'rb') as f:
            X, y = pickle.load(f)
            y = np.array([v.replace(',', '') for v in y], dtype=np.float32)
        with open(dataset_path['transformed'][1], 'rb') as f:
            X_, y_ = pickle.load(f)
            y_ = np.array([v.replace(',', '') for v in y_], dtype=np.float32)
        X = np.concatenate((X, X_), axis=0)
        y = np.concatenate((y, y_), axis=0)
        model = models.FullyConnected5()
    else:
        with open(dataset_path['original'][0], 'rb') as f:
            X, y = pickle.load(f)
            y = np.array([v.replace(',', '') for v in y], dtype=np.float32)
        ext_model = models.FullyConnected5()
        model = models.FeatureExtractorConvNet(feature_extractor_net_cfg, ext_model)



    X = Variable(torch.from_numpy(X)).float()
    y = Variable(torch.from_numpy(y)).int()

    total_samples = X.shape[0]
    num_training_sample = int(total_samples * sampler_ratio['train'])
    num_validation_sample = int(total_samples * sampler_ratio['validate'])
    num_testing_sample = int(total_samples * sampler_ratio['test'])

    train_idx, validate_idx, test_idx = data.random_split(X, [num_training_sample, num_validation_sample, num_testing_sample])
    train = [X[train_idx.indices], y[train_idx.indices]]
    validate = [X[validate_idx.indices], y[validate_idx.indices]]
    test = [X[test_idx.indices], y[test_idx.indices]]

    if cfg['train_transformed_data']:
        train[0] = train[0].view(train[0].shape[0], -1)
        validate[0] = validate[0].view(validate[0].shape[0], -1)
        test[0] = test[0].view(test[0].shape[0], -1)
    else:
        train[0] = train[0].view(train[0].shape[0], 1, 28, -1)
        validate[0] = validate[0].view(validate[0].shape[0], 1, 28, -1)
        test[0] = test[0].view(test[0].shape[0], 1, 28, -1)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyper_parameters['lr'])

    train = Train(data=[train, validate],
                  data_params=dataset_params,
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  hyper_params=hyper_parameters,
                  validate=True,
                  device=device,
                  use_tensorboard=cfg['use_tensorboard'])
    train.fit()

