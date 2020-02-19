import torch.nn.functional as F
from torch import nn
from torch import unsqueeze
from torchsummary import summary
import numpy as np

# TODO: use sequential function to make blocks and remove repetitions


class FeatureExtractorConvNet(nn.Module):
    def __init__(self, config, fcn_config, device):
        super().__init__()
        self.number_of_conv_layer = len(config['n_feature_maps'])
        self.number_of_feature_maps_per_layer = config['n_feature_maps']
        self.kernel_sizes = [x[0][0] for x in config['kernel_sizes']]
        self.strides = config['stride']
        self.pooling_strides = config['pooling_stride']
        self.zero_pad = config['zero_pad']
        self.device = device
        conv_blocks = []

        for i in range(0, self.number_of_conv_layer):
            input_channels = 1 if i == 0 else output_channels
            output_channels = self.number_of_feature_maps_per_layer[i] if i == 0 \
                else output_channels * self.number_of_feature_maps_per_layer[i]

            conv_blocks.append(self.conv_block(input_channels, output_channels, self.kernel_sizes[i], self.strides[i], self.pooling_strides[i]))

        self.net = nn.Sequential(*conv_blocks)
        self.net.to(device)
        final_layer_params = summary(self.net, (1, 28, 28), print=False)
        fc_input_shape = np.prod(final_layer_params['output_shape'][1:])
        self.ext_net = FCNCustom(fc_input_shape, fcn_config)

    def conv_block(self, input_channels, output_channels, kernel_size, stride, pooling_stride):
        if pooling_stride == 0:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size-2),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size-2),
                nn.MaxPool2d(kernel_size=pooling_stride, stride=pooling_stride, ceil_mode=True),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.net(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)  # flat
        # print(x.size())
        x = self.ext_net(x)
        x = F.log_softmax(x, dim=1)

        return x

class FCNCustom(nn.Module):
    def __init__(self, input_size, fcn_config):
        fcn_config.insert(0, input_size)
        super(FCNCustom, self).__init__()
        fcn = []
        for in_connections, out_connections in list(zip(fcn_config[:-1], fcn_config[1:])):
            if out_connections == fcn_config[-1]:
                fcn.append(nn.Sequential(
                    nn.Linear(in_connections, out_connections)))
            else:
                fcn.append(nn.Sequential(
                    nn.Linear(in_connections, out_connections),
                    nn.ReLU()))
        self.net = nn.Sequential(*fcn)

    def forward(self, x):
        x = self.net(x)
        x = F.log_softmax(x, dim=1)

        return x

class FullyConnected5(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Linear(3136, 1568)
        # self.fc1 = nn.Linear(2048, 768)
        self.fc1_ = nn.Linear(1568, 768)
        self.fc2 = nn.Linear(768, 384)
        self.fc3 = nn.Linear(384, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc1_(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)

        return x


class FullyConnected3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x


class Conv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.conv1 = nn.Conv1d(1, 16, 5)
        self.conv2 = nn.Conv1d(16, 32, 5)
        self.fc3 = nn.Linear(32 * 125, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)
        self.pool = nn.MaxPool1d(2, 2)

    def forward(self, x):
        x = unsqueeze(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 125)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)

        return x

class Conv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc3 = nn.Linear(32 * 25, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = unsqueeze(x, 1)
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 32, 32)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 25)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.log_softmax(x, dim=1)

        return x