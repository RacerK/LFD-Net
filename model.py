import torch
import torch.nn as nn


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class LFD_Net(nn.Module):

    def __init__(self):
        super(LFD_Net, self).__init__()

        # mainNet Architecture
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv_layer1 = nn.Conv2d(3,32,3,1,1,bias=True)
        self.conv_layer2 = nn.Conv2d(32,32,5,1,2,bias=True)
        self.conv_layer3 = nn.Conv2d(32,32,7,1,3,bias=True)
        # self.conv_layer4 = nn.Conv2d(64,32,3,1,1,bias=True)
        self.conv_layer5 = nn.Conv2d(64,16,3,1,1,bias=True)
        self.conv_layer6 = nn.Conv2d(16,3,1,1,0,bias=True)

        self.calayer = CALayer(64)
        self.palayer = PALayer(64)
        
        self.gate = nn.Conv2d(32 * 3, 3, 3, 1, 1, bias=True)

    def forward(self, img):
        x1 = self.relu(self.conv_layer1(img))
        x2 = self.relu(self.conv_layer2(x1))
        x3 = self.relu(self.conv_layer3(x2))
        x4 = x1 + x3
        gates = self.gate(torch.cat((x1,x2,x4),1))
        x6 = x1 * gates[:, [0], :, :] + x2 * gates[:, [1], :, :] + x4 * gates[:, [2], :, :]
        x7 = torch.cat((x6,x3),1)
        x8 = self.calayer(x7)
        x9 = self.palayer(x8)
        x10 = self.relu(self.conv_layer5(x9))
        x11 = self.conv_layer6(x10)

        dehaze_image = self.relu((x11 * img) - x11 + 1)

        return dehaze_image
