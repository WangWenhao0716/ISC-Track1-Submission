import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *
import numpy as np

def LogSumExp(score, mask):
    max_score = score.max()
    max_score = max_score.unsqueeze(0).unsqueeze(1).expand_as(score)
    score = score - max_score * (1-mask)   # elimintate the scores which are of none use
    max_score, _ = score.max(1)
    max_score_reduce = max_score.unsqueeze(1).expand_as(score)
    score = score - max_score_reduce
    return max_score + ((score.exp() * mask).sum(1)).log()

class CosfacePairwiseLoss(nn.Module):
    def __init__(self, m=0.35, s=32):
        super(CosfacePairwiseLoss, self).__init__()
        self.m = m
        self.s = s
        self.simi_pos = None
        self.simi_neg = None

    def forward(self, input, target):
        input = F.normalize(input)
        n = input.size(0)
        target = target.cuda()
        mask = target.expand(n, n).eq(target.expand(n, n).t())
        mask = mask.float()
        mask_self = torch.FloatTensor(np.eye(n)).cuda()
        mask_pos = mask - mask_self
        mask_neg = 1 - mask

        simi = input.mm(input.t())
        self.simi_pos = LogSumExp(- simi * self.s, mask_pos).mean() / (- self.s)
        self.simi_neg = LogSumExp(simi * self.s, mask_neg).mean() / self.s
        simi = (simi - self.m * mask) * self.s

        pos_LSE_cmp = LogSumExp(- simi, mask_pos)
        neg_LSE_cmp = LogSumExp(simi, mask_neg)

        loss_cmp = F.softplus(pos_LSE_cmp + neg_LSE_cmp)
        
        '''
        mask_pos, mask_neg = mask_pos.bool(), mask_neg.bool()
        pos_pairs = torch.masked_select(simi, mask_pos).reshape(n, -1)
        neg_pairs = torch.masked_select(simi, mask_neg).reshape(n, -1)
        pos_LSE = torch.logsumexp(- pos_pairs, 1)
        neg_LSE = torch.logsumexp(neg_pairs, 1)
        loss = F.softplus(pos_LSE + neg_LSE)
        '''

        return loss_cmp.mean()
