import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## Convolutional layers, where weights represent conv kernels
        # 1 input image channel, 6 output channels, 3x3 square convolution
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        # 6 input channels (the output of the last layer), 16 output channels, 3x3 square convolution
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        ## Linear layer: MLP, i.e. fully-connected layer.
        self.fc1 = nn.Linear(in_features = 16 * 6 * 6, out_features = 120)  # 6*6 from the image dimension, and 16 for the number of channels
        self.fc2 = nn.Linear(in_features = 120, out_features = 84) # 120 is output of the previous layer.
        self.fc3 = nn.Linear(in_features = 84, out_features = 10) # 84 is the output of the previous layer, 10 is the number of classes.

    def forward(self, x):
        # Conv1, then max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Conv2, then max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x)) # Reshape each image, processed by conv, into a vector (required for linear layers)
        # 1st Linear layer
        x = F.relu(self.fc1(x))
        # 2nd Linear layer
        x = F.relu(self.fc2(x))
        # 3rd Linear layer
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # Computes the number of flat (*"vectorized"*) features from a 2D conv.
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight