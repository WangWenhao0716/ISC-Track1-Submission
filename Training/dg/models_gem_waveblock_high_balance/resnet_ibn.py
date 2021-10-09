from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
import random

from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .gem import GeneralizedMeanPoolingP

__all__ = ['ResNetIBN', 'resnet_ibn50a', 'resnet_ibn101a']

class Waveblock(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(0.3 * h)
            sx = random.randint(0, h-rh)
            mask = (x.new_ones(x.size()))*1.5
            mask[:, :, sx:sx+rh, :] = 1
            x = x * mask 
        return x

class ResNetIBN(nn.Module):
    __factory = {
        '50a': resnet50_ibn_a,
        '101a': resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 dev = None):
        super(ResNetIBN, self).__init__()
        assert dev != None and len(dev) ==4
        self.dev = []
        for i in range(4):
            self.dev.append(dev[i])
        
        self.pretrained = False
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNetIBN.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetIBN.__factory[depth](pretrained=self.pretrained)
        
        print("Load the pretrained M2T...")
        import pickle as pkl
        ckpt = '/dev/shm/unsupervised_pretrained_m2t_ibn.pkl'
        print("loading ckpt... ",ckpt)
        para = pkl.load(open(ckpt, 'rb'), encoding='utf-8')['model']
        resnet.load_state_dict(para,strict = False)
        
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        gap = GeneralizedMeanPoolingP() #nn.AdaptiveAvgPool2d(1)
        print("The init norm is ",gap)
        waveblock = Waveblock()
        
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool, resnet.relu,
            resnet.layer1,
            resnet.layer2, waveblock,
            resnet.layer3, waveblock,
            resnet.layer4, gap
        ).cuda()
        
        if not self.cut_at_pooling:
            self.num_features = 2048*4
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                
                projecter = nn.Sequential(
                                nn.Linear(2048, 2048*2),
                                nn.BatchNorm1d(2048*2),
                                nn.LeakyReLU(0.2, inplace = True),
                                nn.Linear(2048*2, 2048*4)
                )
                
                assert num_classes % 4 == 0
                    
                self.classifier_0 = nn.Linear(self.num_features, self.num_classes//4, bias=False).to(self.dev[0])
                init.normal_(self.classifier_0.weight, std=0.001)
                self.classifier_1 = nn.Linear(self.num_features, self.num_classes//4, bias=False).to(self.dev[1])
                init.normal_(self.classifier_1.weight, std=0.001)
                self.classifier_2 = nn.Linear(self.num_features, self.num_classes//4, bias=False).to(self.dev[2])
                init.normal_(self.classifier_2.weight, std=0.001)
                self.classifier_3 = nn.Linear(self.num_features, self.num_classes//4, bias=False).to(self.dev[3])
                init.normal_(self.classifier_3.weight, std=0.001)
                
                
            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = 2048*4
                feat_bn = nn.BatchNorm1d(self.num_features)
            feat_bn.bias.requires_grad_(False)
            
        init.constant_(feat_bn.weight, 1)
        init.constant_(feat_bn.bias, 0)
        
        self.projector_feat_bn = nn.Sequential(
            projecter,
            feat_bn
        ).cuda()
    
    def forward(self, x, feature_withbn=False):
        x = torch.nn.functional.torch.nn.parallel.data_parallel(self.base, x, device_ids=[0, 1, 2, 3])
        x = x.view(x.size(0), -1)
        
        bn_x = torch.nn.functional.torch.nn.parallel.data_parallel(self.projector_feat_bn, x, device_ids=[0, 1, 2, 3])

        # Split FC->
        prob = [None for _ in range(4)]
        prob[0] = (self.classifier_0(bn_x.to(self.dev[0]))).to(self.dev[0])
        prob[1] = (self.classifier_1(bn_x.to(self.dev[1]))).to(self.dev[0])
        prob[2] = (self.classifier_2(bn_x.to(self.dev[2]))).to(self.dev[0])
        prob[3] = (self.classifier_3(bn_x.to(self.dev[3]))).to(self.dev[0])
        
        prob = torch.cat(prob, dim = 1)
        # <-Split FC
        return x, prob


def resnet_ibn50a(**kwargs):
    return ResNetIBN('50a', **kwargs)


def resnet_ibn101a(**kwargs):
    return ResNetIBN('101a', **kwargs)
