import torch
import torch.nn as nn

from darknet import DarkNet


class YOLOv1(nn.Module):
  def __init__(self, detect, num_bboxes=2, num_classes=20) -> None:
    super(YOLOv1, self).__init__()

    self.feature_size = 7
    self.num_bboxes = num_bboxes
    self.num_classes = num_classes

    self.detect = detect
    self.conv = nn.Sequential(
      nn.Conv2d(1024, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      #
      nn.Conv2d(1024, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Conv2d(1024, 1024, 3, padding=1),
      nn.LeakyReLU(0.1, inplace=True),
    )

    self.fc = self._make_fc_layers()

  def forward(self, x):
    S, B, C = self.feature_size, self.num_bboxes, self.num_classes

    x = self.detect(x)
    x = self.conv(x)
    x = self.fc(x)

    x = x.view(-1, S, S, B * 5 + C)
    return x

  def _make_fc_layers(self):
    S, B, C = self.feature_size, self.num_bboxes, self.num_classes
    net = nn.Sequential(
      nn.Flatten(),
      nn.Linear(7 * 7 * 1024, 4096),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Dropout(0.5),
      nn.Linear(4096, S * S * (B * 5 + C)),
    )
    return net


def test():
  from torch.autograd import Variable

  darknet = DarkNet(conv_only=True, init_weight=True)
  yolo = YOLOv1(darknet.conv)

  img = torch.rand(10, 3, 448, 448)
  img = Variable(img)

  output = yolo(img)
  print(output.size())


if __name__ == '__main__':
  test()
