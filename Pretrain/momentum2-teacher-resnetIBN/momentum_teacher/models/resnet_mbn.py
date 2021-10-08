import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
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
        super(ResNet, self).__init__()
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, res5=False):
        # See note [TorchScript super()]
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
        if res5:
            return x.detach()
        if self.CLS:
            x_flatten = x.clone().detach()
        x = self.fc(x)

        if self.CLS:
            return x, x_flatten
        else:
            return x


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
