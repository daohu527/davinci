import torch
import torch.nn as nn

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()

class DarkNet(nn.Module):
  def __init__(self, conv_only=False, init_weight=True) -> None:
    super(DarkNet, self).__init__()

    self.conv = self._make_conv_layers()

    if not conv_only:
      self.fc = nn.Sequential(
        nn.AvgPool2d(7),
        Squeeze(),
        nn.Linear(1024, 1000),
      )

    self.conv_only = conv_only

    if init_weight:
      self._initialize_weights()

  def forward(self, x):
    x = self.conv(x)
    if not self.conv_only:
      x = self.fc(x)
    return x

  def _make_conv_layers(self) -> nn.Sequential:
    net = nn.Sequential(
      # 1
      nn.Conv2d(3, 64, 7, stride=2, padding=3),
      nn.LeakyReLU(0.1, inplace=True),
      nn.MaxPool2d(2),
      # 2
      nn.Conv2d(64, 192, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.MaxPool2d(2),
      # 3
      nn.Conv2d(192, 128, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(128, 256, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 256, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.MaxPool2d(2),
      # 4
      nn.Conv2d(512, 256, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 256, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 256, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 256, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(256, 512, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 512, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.MaxPool2d(2),
      # 5
      nn.Conv2d(1024, 512, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 512, 1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(512, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
    )
    return net

  def _initialize_weights(self) -> None:
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
          nn.init.normal_(m.weight, 0, 0.01)
          nn.init.constant_(m.bias, 0)
