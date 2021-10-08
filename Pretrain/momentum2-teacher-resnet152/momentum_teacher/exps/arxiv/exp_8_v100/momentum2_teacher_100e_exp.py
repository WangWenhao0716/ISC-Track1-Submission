# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

import os
from momentum_teacher.exps.arxiv.momentum2_teacher_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.max_epoch = 100
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
