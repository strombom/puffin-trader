
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torch

tensor = torch.load('features.tensor')
tensor = tensor.transpose(0, 1).transpose(1, 2)

save_image(tensor, "features.png")
