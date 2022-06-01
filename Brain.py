import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class Brain(nn.Module):
    def __init__(self, arg):
        super(Brain, self).__init__()
        self.conv1 = ConvBlock(kernel_size=4, in_channels=1, out_channels=64)
        self.conv2 = ConvBlock(kernel_size=4, in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(kernel_size=3, in_channels=128, out_channels=256)
        self.conv4 = ConvBlock(kernel_size=3, in_channels=256, out_channels=256)
        self.conv5 = ConvBlock(kernel_size=2, in_channels=256, out_channels=512)
        self.conv6 = ConvBlock(kernel_size=2, in_channels=512, out_channels=512)
        self.fc1 = nn.Linear(21504, 128)
        self.fc2 = nn.Linear(128, 7)

        self.optimizer = optim.Adam(self.parameters(), lr=arg.lr)
        # self.loss = nn.MSELoss()
        self.loss = nn.SmoothL1Loss()

        self.device = torch.device(f'cuda:{arg.gpu}' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s):
        s = torch.Tensor(s).to(self.device)
        s = s.reshape(-1, 1, 6, 7)
        s = self.conv1(s)
        s = self.conv2(s)
        s = self.conv3(s)
        s = self.conv4(s)
        s = self.conv5(s)
        s = self.conv6(s).flatten(1)
        s = F.relu(self.fc1(s))
        s = torch.tanh(self.fc2(s))

        return s


if __name__ == '__main__':
    pass