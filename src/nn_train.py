# TODO: Make a function for Tensorboard summary writer
# FIXME: Fix the warning on the python file import i.e. 'nn_utils.py'

import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
from nn_utils import DataTransform
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter


class NNTrain:
    def __init__(self,
                 data,
                 data_params,
                 model,
                 criterion,
                 optimizer,
                 hyper_params,
                 fcn_config,
                 validate=False,
                 device='cpu',
                 use_tensorboard=True):
        """
        This class takes all the training parameters and trains the model
        :param data: Dataset for training, and validation.
        :param data_params: Parameters for data generator e.g. batch size, shuffle, etc.
        :param model: The model that the data will be fed into.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :param hyper_params: Hyper parameters for the model and training.
        :param validate: If True then during training, the model is validated
                         using validation dataset.
                         Default: False
        :param device: Device to use for computation.
                       Default: 'cpu'
        :param use_tensorboard: If True then uses Tensorboard.
                                Default: True.
        """

        self.data = data
        self.data_params = data_params
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.hyper_params = hyper_params
        self.fcn_config = fcn_config
        self.validate = validate
        self.device = device
        self.use_tensorboard = use_tensorboard
        self.training_generator = DataLoader(DataTransform(*self.data[0]), **self.data_params)
        self.validation_generator = DataLoader(DataTransform(*self.data[1]), **self.data_params)


    def fit(self):
        self.model.to(self.device)
        print(self.model)
        if self.use_tensorboard:
            writer = SummaryWriter(comment='_gr_3')
            writer.add_text('Fully Connected Layer Config', '_'.join(str(x) for x in self.fcn_config))

        max_accuracy_train = 0
        max_accuracy_validation = 0
        correct = 0
        total = 0
        graph_flag = True
        for i in range(0, self.hyper_params['max_epoch']):
            cum_loss = 0
            for samples, labels in self.training_generator:
                self.optimizer.zero_grad()
                samples = samples.to(self.device)
                if self.use_tensorboard and graph_flag:
                    writer.add_graph(model=self.model, input_to_model=samples, verbose=False)
                    graph_flag = False
                labels = labels.to(self.device)
                output = self.model(samples)
                labels = Variable(labels).long()
                loss = self.criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.05)
                self.optimizer.step()
                cum_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_accuracy = 100 * correct / total
            print(f"Training loss for epoch {i}: {cum_loss / len(self.training_generator)}")
            print(f"Training acuracy for epoch {i}: {train_accuracy}")
            validation_accuracy, validation_loss = self.validation(i)
            if self.use_tensorboard:
                writer.add_scalar('Loss/train', cum_loss, i)
                writer.add_scalar('Loss/validation', validation_loss, i)
                writer.add_scalar('Accuracy/train', train_accuracy, i)
                writer.add_scalar('Accuracy/validation', validation_accuracy, i)

            if max_accuracy_train < train_accuracy:
                max_accuracy_train = train_accuracy
                if self.use_tensorboard:
                    writer.add_text('Max_accuracy/train', str(max_accuracy_train))

            if max_accuracy_validation < validation_accuracy:
                max_accuracy_validation = validation_accuracy
                if self.use_tensorboard:
                    writer.add_text('Max_accuracy/validation', str(max_accuracy_validation))
            print(f"Max accuracy for validation is: {max_accuracy_validation}")
            if np.mod(i + 1, self.hyper_params['checkpoint_step']) == 0:
                torch.save(self.model.state_dict(), os.path.join(os.getcwd(), 'model_fmnist_' + str(i) + '.pth'))
            print("")

        writer.close()

    def validation(self, epoch):
        # model.load_state_dict(torch.load(os.path.join(os.getcwd(),'model_mnist_'+ str(99)+'.pth')))
        # self.model.to(self.device)
        correct = 0
        total = 0
        with torch.no_grad():
            cum_loss = 0
            for data in self.validation_generator:
                samples, labels = data
                samples = samples.to(self.device)  # missing line from original code
                labels = labels.to(self.device)  #
                outputs = self.model(samples.float())
                _, predicted = torch.max(outputs.data, 1)
                labels = Variable(labels).long()
                loss = self.criterion(outputs, labels)
                cum_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Validation accuracy of the network for epoch {epoch}: {accuracy} %")
        return accuracy, cum_loss
