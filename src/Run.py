from FeatureExtractorNet import *
from nn_pipeline import nn_pipeline
import os

stacking_enabled = False

# Define an instance of stacking convolutional network
def main(stacking_enabled):

    # Change the current working directory to project's parent directory
    os.chdir("..")

    if stacking_enabled:
        convNet = FeatureExtractorNet('config_stacking.cfg', config_section="DEFAULT")

        # Train the stacking convolutional network
        convNet.fit()

    # Train the neural network classifier
    nn_pipeline("config_nn.yml", stacking_enabled)


if __name__ == "__main__":
    main(stacking_enabled)
