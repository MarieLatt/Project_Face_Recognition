import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        
        self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)        
        
        self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels = 64, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        
        self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        

        self.fc1 = nn.Linear(in_features = 10*10*512, out_features = 2000)
        self.do5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features = 2000, out_features = 500)
        self.do6 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features = 500, out_features = 6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        # 2D to 1D vector
        x = x.view(-1, 10*10*512)
        x = F.relu(self.fc1(x))
        x = self.do5(x)
        x = F.relu(self.fc2(x))
        x = self.do6(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)