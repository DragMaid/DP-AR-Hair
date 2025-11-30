import torch.nn as nn
from torchvision.models import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 14, 21]):
        super(PerceptualLoss, self).__init__()
        self.layers = layers
        self.vgg = vgg16(pretrained=True).features

        # Freeze the weight of the VGG
        for param in self.vgg.parameers():
            param.requires_grad = False

    def forward(self, generated, target):
        loss = 0
        for layer in self.layers:
            sub_vgg = self.vgg[:layer + 1]
            gen_features = sub_vgg(generated)
            target_features = sub_vgg(target)
            loss += nn.function.mse_loss(gen_features, target_features)
        return loss
