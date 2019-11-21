from PCA_Net import PCA_Net

# Define an instance of stacking PCA convolutional network
pcaNet = PCA_Net('Config.cfg', configSection="DEFAULT")

# Train the stacking convolutional network
pcaNet.train()
