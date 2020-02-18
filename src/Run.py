from src.FeatureExtractorNet import *

# Define an instance of stacking convolutional network
convNet = FeatureExtractorNet('Config.cfg', config_section="DEFAULT")

# Train the stacking convolutional network
convNet.train()
