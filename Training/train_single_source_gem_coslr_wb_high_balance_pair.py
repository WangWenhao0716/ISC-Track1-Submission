from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import numpy as np
import sys
import math

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from dg import datasets
from dg import models_gem_waveblock_high_balance
from dg.trainers import Trainer
from dg.evaluators import Evaluator
from dg.utils.data import IterLoader
from dg.utils.data import transforms as T
from dg.utils.data.sampler import RandomMultipleGallerySampler
from dg.utils.data.preprocessor import Preprocessor
from dg.utils.logging import Logger
from dg.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dg.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=2000):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.train
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
         T.Resize((height, width)),
         T.ToTensor(),
         normalizer
     ])

    test_transformer = T.Compose([
             T.Resize((height, width)),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader

def main():
    args = parser.parse_args()
    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    
    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    
    dataset_source, num_classes, train_loader, _ = \
        get_data(args.dataset_source, args.data_dir, args.height, args.width, \
             args.batch_size, args.workers, args.num_instances, iters)
    
    # Create model
    model = models_gem_waveblock_high_balance.create(
        args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes,
        dev = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    # model.cuda()
    # model = nn.DataParallel(model)
    
    # Evaluator
    lr_rate = args.lr
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params":[value]}]
    
    def f(epoch):
        if epoch<5:
            return (0.99*epoch / 5 + 0.01)
        elif (epoch >= 5 and epoch<10):
            return 1
        else:
            return 0.5 * (math.cos((epoch - 10)/(25 - 10) * math.pi) + 1)

    lambda1 = lambda epoch: f(epoch)
    optimizer = torch.optim.Adam(params, lr=lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}".format(start_epoch))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
    if args.auto_resume:
        import os
        ckpts = os.listdir(args.logs_dir)
        num = 0
        for c in range(len(ckpts)):
            if ("checkpoint_" in ckpts[c]):
                name = int(ckpts[c].split('_')[1].split('.')[0])
                if(name > num):
                    num = name
        checkpoint = load_checkpoint(args.logs_dir + '/' + 'checkpoint_' + str(num) +'.pth.tar')
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}".format(start_epoch))
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    # Trainer
    trainer = Trainer(model, num_classes, margin=args.margin)
    
    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, train_iters=len(train_loader), print_freq=args.print_freq) 
        lr_scheduler.step()
        is_best = False
        save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint_'+str(epoch)+'.pth.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='randperson',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=512, help="input height")
    parser.add_argument('--width', type=int, default=512, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models_gem_waveblock_high_balance.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--auto_resume', action='store_true',help="auto resume")
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
