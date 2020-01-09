from FeatureExtractorNet import *

# Define an instance of stacking convolutional network
# convNet = PCA_Net('Config.cfg', config_section="DEFAULT")
# convNet = Kernel_PCA_Net('Config.cfg', config_section="DEFAULT")
convNet = ICA_Net('Config.cfg', config_section="DEFAULT")

# Train the stacking convolutional network
convNet.train()
