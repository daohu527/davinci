import torch
import torch.nn as nn


class Loss(nn.Module):
  def __init__(self) -> None:
    super(Loss, self).__init__()
