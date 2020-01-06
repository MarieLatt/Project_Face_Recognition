import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 25, 
                               kernel_size = 3, stride = 1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 25, out_channels = 50, 
                               kernel_size = 3, stride = 1, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 50, out_channels = 200, 
                               kernel_size = 3, stride = 1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 200, out_channels = 400, 
                               kernel_size = 3, stride = 1, padding=1)
        self.fc1 = nn.Linear(in_features = 10*10*400, out_features = 2000)
        self.do5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features = 2000, out_features = 500)
        self.do6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features = 500, out_features = 7)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        # 2D to 1D vector
        x = x.view(-1, 10*10*400)
        x = F.relu(self.fc1(x))
        x = self.do5(x)
        x = F.relu(self.fc2(x))
        x = self.do6(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)