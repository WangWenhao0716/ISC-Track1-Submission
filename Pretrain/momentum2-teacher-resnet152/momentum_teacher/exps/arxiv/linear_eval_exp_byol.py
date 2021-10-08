# encoding: utf-8
"""
@author: zeming li
@contact: zengarden2009@gmail.com
"""

from momentum_teacher.exps.arxiv.linear_eval_exp import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ----------------------------- byol setting ------------------------------- #
        self.basic_lr_per_img = 0.2 / 256.0
        self.max_epochs = 80
        self.scheduler = "cos"  # "multistep"
        self.epoch_of_stage = None
        self.save_folder_prefix = "byol_"
