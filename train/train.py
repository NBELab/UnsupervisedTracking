import torch
import argparse
import collections
import numpy as np
import os
import sys

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
if module_path not in sys.path:
    sys.path.append(module_path)
import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from net_utils import *
from net import *
import time
import shutil
from tqdm import *
import torch.nn as nn
from datetime import datetime

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

config = TrackerConfig()
gpu_num = torch.cuda.device_count()

# globals
model = None
criterion = None
optimizer = None
batch_size = None
target = None
num_bins = None
print_freq = None


def output_drop(output, target):
    delta1 = (output - target) ** 2
    batch_sz = delta1.shape[0]
    delta = delta1.view(batch_sz, -1).sum(dim=1)
    sort_delta, index = torch.sort(delta, descending=True)
    # unreliable samples (10% of the total) do not produce grad (we simply copy the groundtruth label)
    for i in range(int(round(0.1 * batch_sz))):
        output[index[i], ...] = target[index[i], ...]
    return output


def adjust_learning_rate(optimizer, epoch, total_epochs):
    lr = np.logspace(-2, -5, num=total_epochs)[epoch]
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, save_path):
    torch.save(state, os.path.join(save_path, "checkpoint.pth.tar"))
    if is_best:
        shutil.copyfile(
            os.path.join(save_path, "checkpoint.pth.tar"),
            os.path.join(save_path, "model_best.pth.tar"),
        )


def get_features(x, is_frame):
    if is_frame:
        x_enc = model.enc_f(x)
        x_old = model.frame_feat(x)
    else:
        x_enc = model.enc_e(x)
        x_old = model.event_feat(x)
    return model.dec(x_enc) + x_old


def get_response(template_feat, search1_feat, label, initial_y):
    with torch.no_grad():
        s1_response = model(template_feat, search1_feat, label)
    # label transform
    peak, index = torch.max(s1_response.view(batch_size * gpu_num, -1), 1)
    r_max, c_max = np.unravel_index(index.cpu(), [config.output_sz, config.output_sz])
    fake_y = np.zeros((batch_size * gpu_num, 1, config.output_sz, config.output_sz))
    # label shift
    for j in range(batch_size * gpu_num):
        shift_y = np.roll(initial_y, r_max[j])
        fake_y[j, ...] = np.roll(shift_y, c_max[j])
    fake_yf = torch.fft.rfft2(
        torch.Tensor(fake_y)
        .view(batch_size * gpu_num, 1, config.output_sz, config.output_sz)
        .cuda()
    )
    fake_yf = torch.view_as_real(fake_yf)

    return fake_yf.cuda(non_blocking=True)


def train(train_loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    label = config.yf.repeat(batch_size * gpu_num, 1, 1, 1, 1).cuda(non_blocking=True)
    initial_y = config.y.copy()

    end = time.time()

    for i, x in enumerate(train_loader):
        template = x[0]
        search1, search2 = x[1], x[2]

        if template["events"].shape[0] != label.shape[0]:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        template_e = get_features(template["events"].cuda(non_blocking=True), False)
        template_f = get_features(template["frame"].cuda(non_blocking=True), True)
        template_feat = template_e + template_f

        search1_e = get_features(search1["events"].cuda(non_blocking=True), False)
        search1_f = get_features(search1["frame"].cuda(non_blocking=True), True)
        search1_feat = search1_e + search1_f

        search2_e = get_features(search2["events"].cuda(non_blocking=True), False)
        search2_f = get_features(search2["frame"].cuda(non_blocking=True), True)
        search2_feat = search2_e + search2_f

        # forward tracking 1
        fake_yf = get_response(template_feat, search1_feat, label, initial_y)

        # forward tracking 2
        fake_yf = get_response(search1_feat, search2_feat, fake_yf, initial_y)

        # backward tracking
        output = model(search2_feat, template_feat, fake_yf)
        output = output_drop(
            output, target
        )  # the sample dropout is necessary, otherwise we find the loss tends to become unstable

        output_e = model(search2_e, template_e, fake_yf)
        output_e = output_drop(output_e, target)

        output_f = model(search2_f, template_f, fake_yf)
        output_f = output_drop(output_f, target)

        # consistency loss. target is the initial Gaussian label
        loss = (
            0.5 * criterion(output, target)
            + 0.2 * criterion(output_e, target)
            + 0.3 * criterion(output_f, target)
        ) / template_feat.size(0)

        # measure accuracy and record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if torch.isnan(loss):
            return False

        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )

    return True


def validate(val_loader):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()
    initial_y = config.y.copy()
    label = config.yf.repeat(batch_size * gpu_num, 1, 1, 1, 1).cuda(non_blocking=True)

    with torch.no_grad():
        end = time.time()
        for i, x in enumerate(val_loader):
            # compute output
            template = x[0]
            search1, search2 = x[1], x[2]

            if template["events"].shape[0] != label.shape[0]:
                continue
            template_e = get_features(template["events"].cuda(non_blocking=True), False)
            template_f = get_features(template["frame"].cuda(non_blocking=True), True)
            template_feat = template_e + template_f

            search1_e = get_features(search1["events"].cuda(non_blocking=True), False)
            search1_f = get_features(search1["frame"].cuda(non_blocking=True), True)
            search1_feat = search1_e + search1_f

            search2_e = get_features(search2["events"].cuda(non_blocking=True), False)
            search2_f = get_features(search2["frame"].cuda(non_blocking=True), True)
            search2_feat = search2_e + search2_f

            # forward tracking 1
            fake_yf = get_response(template_feat, search1_feat, label, initial_y)

            # forward tracking 2
            fake_yf = get_response(search1_feat, search2_feat, fake_yf, initial_y)

            # backward tracking
            output = model(search2_feat, template_feat, fake_yf)
            output = output_drop(
                output, target
            )  # the sample dropout is necessary, otherwise we find the loss tends to become unstable

            output_e = model(search2_e, template_e, fake_yf)
            output_e = output_drop(output_e, target)

            output_f = model(search2_f, template_f, fake_yf)
            output_f = output_drop(output_f, target)

            # consistency loss. target is the initial Gaussian label
            loss = (
                0.5 * criterion(output, target)
                + 0.2 * criterion(output_e, target)
                + 0.3 * criterion(output_f, target)
            ) / (batch_size * gpu_num)

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        i, len(val_loader), batch_time=batch_time, loss=losses
                    )
                )

        print(" * Loss {loss.val:.4f} ({loss.avg:.4f})".format(loss=losses))

    return losses.avg


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=os.getcwd() + "/train/config/config.json",
        type=str,
        help="config file path (default: CWD +/train/config/config.json)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    parser.add_argument(
        "--limited_memory",
        default=False,
        action="store_true",
        help='prevent "too many open files" error by setting pytorch multiprocessing to "file_system".',
    )

    return ConfigParser.from_args(parser)


def init_globals(args):
    global num_bins
    num_bins = args.config["data_loader"]["args"]["sequence_kwargs"]["dataset_kwargs"][
        "num_bins"
    ]

    global model
    model = DCFNet(num_bins=num_bins, config=config).cuda()

    global criterion
    criterion = nn.MSELoss(size_average=False).cuda()

    global optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.config["trainer"]["lr"],
        momentum=args.config["trainer"]["momentum"],
        weight_decay=args.config["trainer"]["weight_decay"],
    )
    global batch_size
    batch_size = args.config["data_loader"]["args"]["batch_size"]

    global target
    target = (
        torch.Tensor(config.y)
        .cuda()
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(batch_size * gpu_num, 1, 1, 1)
    )  # for training

    global print_freq
    print_freq = args.config["trainer"]["print_freq"]


def main(args):
    # init globals
    init_globals(args)
    global model
    # setup data_loader instances
    train_loader = args.init_obj("data_loader", module_data)
    val_loader = args.init_obj("valid_data_loader", module_data)

    dt_string = datetime.now().strftime("%d%m%Y_%H%M%S")
    save_path = os.path.join(
        args.config["trainer"]["save_dir"], "{}_{:d}".format(dt_string, num_bins)
    )
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if gpu_num > 1:
        model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

    best_loss = 1e6
    start_epoch = args.config["trainer"]["start_epoch"]
    total_epochs = args.config["trainer"]["epochs"]
    try:
        for epoch in tqdm(range(start_epoch, total_epochs)):
            adjust_learning_rate(optimizer, epoch, total_epochs)

            # train for one epoch
            worked = train(train_loader, epoch)
            if not worked:
                print("Broken")
                break
            # evaluate on validation set
            loss = validate(val_loader)
            if epoch % 5 == 0:
                print(loss)
            # remember best loss and save checkpoint
            is_best = loss < best_loss
            best_loss = min(best_loss, loss)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                save_path,
            )

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main(parse_args())
