from torch import nn
from torchsummary import summary

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        print(y.shape)
        y = self.fc(y).view(b, c, 1, 1)
        print(y.shape)
        print(x.shape)
        print((x*y.expand_as(x)).shape)
        return x*y.expand_as(x)

if __name__ == "__main__":
    se = SELayer(32, 4)
    summary(se.cuda(), (32, 256, 256))

