import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet_IBN', 'resnet18_ibn_a', 'resnet34_ibn_a', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
           'resnet18_ibn_b', 'resnet34_ibn_b', 'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b']


model_urls = {
    'resnet18_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'resnet34_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet18_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_b-bc2f3c11.pth',
    'resnet34_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_b-04134c37.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}

class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, norm_layer, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = norm_layer(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)

class SimpleBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm2d, self).forward(x)
        return result


class SimpleBatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        if self.train:
            self.running_mean = self.running_mean.clone()
            self.running_var = self.running_var.clone()
            result = super(SimpleBatchNorm1d, self).forward(x)
        return result


class MomentumBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0, 2, 3])
        var = torch.var(x, dim=[0, 2, 3])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :, None, None].detach()) / (
            torch.sqrt(tmp_running_var[None, :, None, None].detach() + self.eps)
        )
        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch) * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x


class MomentumBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=1.0, affine=True, track_running_stats=True, total_iters=100):
        super(MomentumBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.total_iters = total_iters
        self.cur_iter = 0
        self.mean_last_batch = None
        self.var_last_batch = None

    def momentum_cosine_decay(self):
        self.cur_iter += 1
        self.momentum = (math.cos(math.pi * (self.cur_iter / self.total_iters)) + 1) * 0.5

    def forward(self, x):
        # if not self.training:
        #     return super().forward(x)

        mean = torch.mean(x, dim=[0])
        var = torch.var(x, dim=[0])
        n = x.numel() / x.size(1)

        with torch.no_grad():
            tmp_running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
            # update running_var with unbiased var
            tmp_running_var = self.momentum * var * n / (n - 1) + (1 - self.momentum) * self.running_var

        x = (x - tmp_running_mean[None, :].detach()) / (torch.sqrt(tmp_running_var[None, :].detach() + self.eps))
        if self.affine:
            x = x * self.weight[None, :] + self.bias[None, :]

        # update the parameters
        if self.mean_last_batch is None and self.var_last_batch is None:
            self.mean_last_batch = mean
            self.var_last_batch = var
        else:
            self.running_mean = (
                self.momentum * ((mean + self.mean_last_batch) * 0.5) + (1 - self.momentum) * self.running_mean
            )
            self.running_var = (
                self.momentum * ((var + self.var_last_batch) * 0.5) * n / (n - 1)
                + (1 - self.momentum) * self.running_var
            )
            self.mean_last_batch = None
            self.var_last_batch = None
            self.momentum_cosine_decay()

        return x    

    
class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None, norm_layer = None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes,norm_layer)
        else:
            self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None, norm_layer = None):
        super(Bottleneck_IBN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes,norm_layer)
        else:
            self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000,
                 low_dim=128,
                 width=1,
                 MLP="none",
                 CLS=False,
                 hidden_dim=2048,
                 groups=1,
                 bn="vanilla",
                 predictor=False,
                 zero_init_residual=False,
                 replace_stride_with_dilation=None,
                ):
        super(ResNet_IBN, self).__init__()
        if bn == "customized":
            self._norm_layer = SimpleBatchNorm2d
            self._norm1d_layer = SimpleBatchNorm1d
        elif bn == "vanilla":
            self._norm_layer = nn.BatchNorm2d
            self._norm1d_layer = nn.BatchNorm1d
        elif bn == "torchsync":
            self._norm_layer = nn.SyncBatchNorm
            self._norm1d_layer = nn.SyncBatchNorm
        elif bn == "mbn":
            self._norm_layer = MomentumBatchNorm2d
            self._norm1d_layer = MomentumBatchNorm1d
        else:
            raise ValueError("bn should be none or 'cvsync' or 'torchsync', got {}".format(bn))
            
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = 64 * width
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = self._norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.CLS = CLS
        self.predictor = predictor

        if MLP == "moco":
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                Normalize(),
            )
        elif MLP == "simclr":
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, hidden_dim),
                self._norm1d_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                self._norm1d_layer(low_dim),
                Normalize(),
            )
        elif MLP == "vanilla":
            self.fc = nn.Sequential(
                nn.Linear(512 * block.expansion, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, low_dim),
                nn.BatchNorm1d(low_dim),
                Normalize(),
            )
        elif MLP == "byol":
            if predictor:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, hidden_dim),
                    self._norm1d_layer(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, low_dim),
                    nn.Linear(low_dim, hidden_dim),
                    self._norm1d_layer(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, low_dim),
                    Normalize(),
                )
            else:
                self.fc = nn.Sequential(
                    nn.Linear(512 * block.expansion, hidden_dim),
                    self._norm1d_layer(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, low_dim),
                    Normalize(),
                )
        elif MLP == "none":
            self.fc = nn.Sequential(nn.Linear(512 * block.expansion, low_dim), Normalize())
        else:
            raise NotImplementedError("MLP version is wrong!")

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample,norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn, norm_layer = norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.CLS:
            x_flatten = x.clone().detach()
        x = self.fc(x)

        if self.CLS:
            return x, x_flatten
        else:
            return x


def resnet18_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet34_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-34-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-a model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    return model


def resnet18_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-18-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[2, 2, 2, 2],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet34_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-34-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=BasicBlock_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-b model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    return model
