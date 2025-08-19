import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_hidden_1, n_hidden_2, convkernel, poolkernel, num_of_layers, num_of_neurons, img_size, padding_mode='zeros'):
        super(CNN, self).__init__()
        self.size = img_size
        self.convkernel = convkernel
        self.poolkernel = poolkernel
        self.padding_mode = padding_mode
        self.padding = (convkernel - 1) // 2

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_hidden_1, kernel_size=convkernel, padding=self.padding, padding_mode=self.padding_mode)
        self.conv2 = nn.Conv2d(in_channels=n_hidden_1, out_channels=n_hidden_2, kernel_size=convkernel, padding=self.padding, padding_mode=self.padding_mode)
        self.conv3 = nn.Conv2d(in_channels=n_hidden_2, out_channels=n_hidden_2, kernel_size=convkernel, padding=self.padding, padding_mode=self.padding_mode)

        self.pool = nn.MaxPool2d(kernel_size=poolkernel)
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        for i in range(num_of_layers):
            in_features = num_of_neurons if i != 0 else self._calculate_flatten_size(n_hidden_2)
            self.fc_layers.append(nn.Linear(in_features, num_of_neurons))

        self.output_layer = nn.Linear(num_of_neurons, 2)

    def _calculate_flatten_size(self, last_conv_channels):
        """ Computes the size of the flattened feature map after conv layers """
        self.eval()
        with torch.no_grad():
            x = torch.rand(1, 3, self.size, self.size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            flatten_size = x.view(-1).shape[0]
        self.train()
        return flatten_size

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)

        for fc in self.fc_layers:
            x = F.relu(fc(x))

        out = torch.sigmoid(self.output_layer(x))
        return out