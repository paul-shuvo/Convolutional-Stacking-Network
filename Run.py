from PCA_Net import PCA_Net, Kernel_PCA_Net

# Define an instance of stacking PCA convolutional network
pcaNet = PCA_Net('Config.cfg', config_section="DEFAULT")

# Train the stacking convolutional network
pcaNet.train()

b=pcaNet.infer(pcaNet.test_samples, pcaNet.feature_extractors, pcaNet.network_settings, False)
