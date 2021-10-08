import os
import sys
import argparse
import time
import random
import warnings
import subprocess
import importlib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP
from momentum_teacher.utils.log import setup_logger
from momentum_teacher.utils import adjust_learning_rate_iter, save_checkpoint, parse_devices, AvgMeter
from momentum_teacher.utils.torch_dist import configure_nccl, synchronize

parser = argparse.ArgumentParser("MomentumTeacher")
parser.add_argument("-expn", "--experiment-name", type=str, default=None)

# optimization
parser.add_argument(
    "--scheduler",
    type=str,
    default="warmcos",
    choices=["warmcos", "cos", "linear", "multistep", "step"],
    help="type of scheduler",
)

# distributed
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
parser.add_argument("-b", "--batch-size", type=int, default=256, help="batch size")
parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")

args = parser.parse_args()
if not args.exp_file:
    from momentum_teacher.exps.arxiv import momentum2_teacher_exp
    exp = momentum2_teacher_exp.Exp()
else:
    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])
    exp = current_exp.Exp()


def main():
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = torch.cuda.device_count()

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

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

            # ------------------------hack for multi-machine training -------------------- #
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

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        if not os.path.exists(file_name):
            os.mkdir(file_name)

    logger = setup_logger(file_name, distributed_rank=rank, filename="train_log.txt", mode="a")
    logger.info("gpuid: {}, args: {}".format(rank, args))

    train_loader = exp.get_data_loader(batch_size=args.batch_size, is_distributed=nr_gpu > 1)["train"]
    model = exp.get_model()
    optimizer = exp.get_optimizer(args.batch_size)

    torch.cuda.set_device(gpu)
    resume = torch.load("/gs/home/wangwh/momentum2-teacher-imagenet/outputs/imagenet_baseline/last_epoch_ckpt.pth.tar", map_location=torch.device('cpu'))
    begin_epoch = resume['start_epoch']
    newmodel = {}
    para = resume['model']
    for k, v in para.items():
        if not k.startswith("module."):
            continue
        old_k = k
        k = k.replace("module.", "")
        newmodel[k] = v
    
    model.load_state_dict(newmodel)
    model.cuda(gpu)
    
    if nr_gpu > 1:
        model = DDP(model, device_ids=[gpu])

    cudnn.benchmark = True

    # ------------------------ start training ------------------------------------------------------------ #
    model.train()
    ITERS_PER_EPOCH = len(train_loader)
    if rank == 0:
        logger.info("Training start...")
        logger.info(str(model))

    args.lr = exp.basic_lr_per_img * args.batch_size
    args.warmup_epochs = exp.warmup_epochs
    args.total_epochs = exp.max_epoch
    
    iter_count = begin_epoch*ITERS_PER_EPOCH#0
    
    
    
    for epoch in range(begin_epoch, args.total_epochs):
        if nr_gpu > 1:
            train_loader.sampler.set_epoch(epoch)
        batch_time_meter = AvgMeter()

        for i, (inps, target) in enumerate(train_loader):
            iter_count += 1
            iter_start_time = time.time()

            for indx in range(len(inps)):
                inps[indx] = inps[indx].cuda(gpu, non_blocking=True)

            data_time = time.time() - iter_start_time

            loss = model(inps, update_param=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = adjust_learning_rate_iter(optimizer, iter_count, args, ITERS_PER_EPOCH)
            batch_time_meter.update(time.time() - iter_start_time)
            if rank == 0 and (i + 1) % exp.print_interval == 0:
                remain_time = (ITERS_PER_EPOCH * exp.max_epoch - iter_count) * batch_time_meter.avg
                t_m, t_s = divmod(remain_time, 60)
                t_h, t_m = divmod(t_m, 60)
                t_d, t_h = divmod(t_h, 24)
                remain_time = "{}d.{:02d}h.{:02d}m".format(int(t_d), int(t_h), int(t_m))

                logger.info(
                    "[{}/{}], remain:{}, It:[{}/{}], Data-Time:{:.3f}, LR:{:.4f}, Loss:{:.2f}".format(
                        epoch + 1, args.total_epochs, remain_time, i + 1, ITERS_PER_EPOCH, data_time, lr, loss
                    )
                )

        if rank == 0:
            logger.info(
                "Train-Epoch: [{}/{}], LR: {:.4f}, Con-Loss: {:.2f}".format(epoch + 1, args.total_epochs, lr, loss)
            )

            save_checkpoint(
                {"start_epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                False,
                file_name,
                "last_epoch",
            )

    if rank == 0:
        print("Pre-training of experiment: {} is done.".format(args.exp_file))


if __name__ == "__main__":
    main()
