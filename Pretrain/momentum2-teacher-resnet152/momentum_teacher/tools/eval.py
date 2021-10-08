import os
import sys
import time
import random
import argparse
import warnings
import subprocess
from tqdm import tqdm

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.nn.parallel import DistributedDataParallel as DDP

from momentum_teacher.utils.log import setup_logger
from momentum_teacher.utils.torch_dist import reduce_tensor_sum, configure_nccl, synchronize
from momentum_teacher.utils import accuracy, adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter

parser = argparse.ArgumentParser("LinearEvaluation")
parser.add_argument("-expn", "--experiment_name", type=str, default="baseline-")

# distributed
parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("-b", "--batch-size", type=int, default=256, help="batch size")
parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")

args = parser.parse_args()
if not args.exp_file:
    from momentum_teacher.exps.arxiv.linear_eval_exp import Exp

    exp = Exp()
else:
    import importlib

    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
    exp = current_exp.Exp()


def main():
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = torch.cuda.device_count()

    if exp.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    nr_machine = int(os.getenv("MACHINE_TOTAL", "1"))
    if nr_gpu > 1:
        args.world_size = nr_gpu * nr_machine
        mp.spawn(main_worker, nprocs=nr_gpu, args=(nr_gpu, args))
    else:
        main_worker(0, nr_gpu, args)


def main_worker(gpu, nr_gpu, args):
    configure_nccl()

    # ------------ set environment variables for distributed training ------------------------------------- #
    rank = gpu
    if nr_gpu > 1:
        rank += int(os.getenv("MACHINE", "0")) * nr_gpu

        if args.dist_url is None:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            master_ip = str(master_ip).strip()
            args.dist_url = "tcp://{}:23456".format(master_ip)

            # ------------------------ for multi-machine training -------------------- #
            if args.world_size > 8:
                ip_add_file = "./" + args.experiment_name + "ip_add.txt"
                if rank == 0:
                    with open(ip_add_file, "w") as ip_add:
                        ip_add.write(args.dist_url)
                else:
                    while not os.path.exists(ip_add_file):
                        time.sleep(0.5)

                    with open(ip_add_file, "r") as ip_add:
                        dist_url = ip_add.readline()
                    args.dist_url = dist_url
        else:
            args.dist_url = "tcp://{}:23456".format(args.dist_url)

        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank
        )
        print("Rank {} initialization finished.".format(rank))
        synchronize()

        if rank == 0:
            if os.path.exists("./" + args.experiment_name + "ip_add.txt"):
                os.remove("./" + args.experiment_name + "ip_add.txt")

    pretrained_file_name = os.path.join(exp.output_dir, args.experiment_name)
    file_name = os.path.join(pretrained_file_name, "{}linear_eval_teacher".format(exp.save_folder_prefix))

    if rank == 0:
        if not os.path.exists(pretrained_file_name):
            os.mkdir(pretrained_file_name)

    logger = setup_logger(file_name, distributed_rank=rank, filename="eval_log.txt", mode="a")
    logger.info("args: {}".format(args))

    data_loader = exp.get_data_loader(batch_size=args.batch_size, is_distributed=nr_gpu > 1)
    train_loader, eval_loader = data_loader["train"], data_loader["eval"]
    model = exp.get_model()

    #  ------------------------------------------- load ckpt ------------------------------------ #
    ckpt_tar = os.path.join(pretrained_file_name, "last_epoch_ckpt.pth.tar")
    ckpt = torch.load(ckpt_tar, map_location="cpu")
    state_dict = {k.replace("module.teacher_encoder.", ""): v for k, v in ckpt["model"].items()}
    missing_keys = []
    matched_state_dict = {}
    for name, param in state_dict.items():
        if "encoder.{}".format(name) not in model.state_dict() or name.startswith("fc"):
            missing_keys.append(name)
        else:
            matched_state_dict["encoder.{}".format(name)] = param
    del state_dict
    msg = model.load_state_dict(matched_state_dict, strict=False)
    del matched_state_dict

    # -------------------------------------- end of the tmp --------------------------------------- #
    if rank == 0:
        logger.info(str(model))
        logger.info("Missing keys: {}".format(missing_keys))
        logger.info("Params {} are not loaded from matched state dict".format(msg.missing_keys))

    optimizer = exp.get_optimizer(args.batch_size)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    if nr_gpu > 1:
        model = DDP(model, device_ids=[gpu])

    #  ---------------------------------- start linear-evaluation ------------------------------ #
    best_top1 = 0
    _best_top5 = 0
    best_top1_epoch = 0

    ITERS_PER_EPOCH = len(train_loader)
    args.lr = exp.basic_lr_per_img * args.batch_size
    args.total_epochs = exp.max_epochs
    args.scheduler = exp.scheduler
    args.milestones = exp.epoch_of_stage
    model.train()

    for epoch in range(0, args.total_epochs):
        if nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)

        for i, (inp, target) in enumerate(train_loader):
            data_time = time.time()
            inp = inp.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
            data_time = time.time() - data_time

            # forward
            logits, loss = model(inp, target)
            top1, top5 = accuracy(logits, target, (1, 5))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_count = epoch * ITERS_PER_EPOCH + i + 1
            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)

            if (i + 1) % exp.print_interval == 0 and rank == 0:
                logger.info(
                    "\tIter: [{}/{}], Data-Time: {:.3f}, LR: {:.4f},"
                    " Loss: {:.4f}, Top-1: {:.2f}, Top-5: {:.2f}".format(
                        i + 1, ITERS_PER_EPOCH, data_time, lr, loss, top1, top5
                    )
                )

        if (epoch + 1) % exp.eval_interval == 0:
            logger.info("start evaluation")
            model.eval()
            eval_top1, eval_top5 = run_eval(model, eval_loader)
            model.train()

            logger.info(
                "\tEval-Epoch: [{}/{}], Top-1: {:.2f}, Top-5: {:.2f}".format(
                    epoch + 1, args.total_epochs, eval_top1, eval_top5
                )
            )

            if eval_top1 > best_top1:
                is_best = True
                best_top1 = eval_top1
                _best_top5 = eval_top5
                best_top1_epoch = epoch + 1
            else:
                is_best = False

            logger.info(
                "\tBest Top-1 at epoch [{}/{}], Best Top-1: {:.2f}, Top-5: {:.2f}".format(
                    best_top1_epoch, args.total_epochs, best_top1, _best_top5
                )
            )
            if rank == 0:
                save_checkpoint(
                    {
                        "start_epoch": epoch + 1,
                        "classifier": model.state_dict(),
                        "best_top1": best_top1,
                        "_best_top5": _best_top5,
                        "best_top1_epoch": best_top1_epoch,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best,
                    file_name,
                    "linear_eval",
                )

    if rank == 0:
        print("Pre-training done.")
        print("Experiment name: {}".format(args.experiment_name))


def run_eval(model, eval_loader):

    top1 = AvgMeter()
    top5 = AvgMeter()

    with torch.no_grad():
        pbar = tqdm(range(len(eval_loader)))
        for _, (inp, target) in zip(pbar, eval_loader):
            inp = inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            logits = model(inp)
            acc1, acc5 = accuracy(logits, target, (1, 5))
            acc1, acc5 = (
                reduce_tensor_sum(acc1) / dist.get_world_size(),
                reduce_tensor_sum(acc5) / dist.get_world_size(),
            )
            top1.update(acc1.item(), inp.size(0))
            top5.update(acc5.item(), inp.size(0))
    return top1.avg, top5.avg


if __name__ == "__main__":
    main()
