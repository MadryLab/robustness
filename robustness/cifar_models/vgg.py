'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        out = self.features(x)
        latent = out.view(out.size(0), -1)
        out = self.classifier(latent)
        if with_latent:
            return out, latent
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def VGG11(**kwargs):
    return VGG('VGG11', **kwargs)

def VGG13(**kwargs):
    return VGG('VGG13', **kwargs)

def VGG16(**kwargs):
    return VGG('VGG16', **kwargs)

def VGG19(**kwargs):
    return VGG('VGG19', **kwargs)

vgg11 = VGG11
vgg13 = VGG13
vgg16 = VGG16
vgg19 = VGG19
