import argparse
import time
import os
import random
import shutil
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from darknet import DarkNet


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--no-bn', dest='bn', action='store_false',
                    help='Use batch normalization layers')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--tb-log-interval', type=int, default=10,
					help='how many batches to wait before saving training status to TensorBoard (default: 10)')
parser.add_argument('--tb-log-dir', '-o', default=None,
					help='directory under `results/darknet` to output TensorBoard'
               'event file and model weight file (default: <DATETIME>)')

best_acc1 = 0


def main():
  args = parser.parse_args()

  # Open TensorBoardX summary writer
  log_dir = args.tb_log_dir if (args.tb_log_dir is not None) else datetime.now().strftime('%b%d_%H-%M-%S')
  log_dir = os.path.join('results/darknet', log_dir)
  writer = SummaryWriter()

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, writer, log_dir, args))
  else:
    main_worker(args.gpu, ngpus_per_node, writer, log_dir, args)

  writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    data_time.update(time.time() - end)
    if args.gpu is not None:
      input = input.cuda(args.gpu, non_blocking=True)
    target = target.cuda(args.gpu, non_blocking=True)

    output = model(input)
    loss = criterion(output, target)

    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), input.size(0))
    top1.update(acc1[0], input.size(0))
    top5.update(acc5[0], input.size(0))

    # commpute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % args.print_freq == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses, top1=top1, top5=top5))

    # TensorBoard
    n_iter = epoch * len(train_loader) + i
    if n_iter % args.tb_log_interval == 0:
      writer.add_scalar('train/loss', losses.val, n_iter)
      writer.add_scalar('train/top1', top1.val, n_iter)
      writer.add_scalar('train/top5', top5.val, n_iter)


def validate(val_loader, model, criterion, epoch, writer, args):
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
      # TODO
      pass


def save_checkpoint(state, is_best, log_dir, filename='checkpoint.pth.tar'):
  path = os.path.join(log_dir, filename)
  torch.save(state, path)
  if is_best:
    path_best = os.path.join(log_dir, 'model_best.pth.tar')
    shutil.copyfile(path, path_best)


def main_worker(gpu, ngpus_per_node, writer, log_dir, args):
  global best_acc1
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

  print("=> creating model DarkNet (bn: {}) ...".format(args.bn))
  model = DarkNet()

  if args.distributed:
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model.cuda(args.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.workers = int(args.workers / ngpus_per_node)
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
      model.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      model = torch.nn.parallel.DistributedDataParallel(model)
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
  else:
    model.conv = torch.nn.DataParallel(model.conv)
    model.cuda()

  # define loss function (criterion) and optimizer
  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  optimizer = torch.optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)

  # optionally resume from a checkpoint
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume)
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  # Data loading code
  traindir = os.path.join(args.data, 'train')
  valdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

  scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                lr_lambda = lambda epoch: args.lr * (0.1 ** (epoch // 30)))

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
    train_sampler = None

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

  # evaluate then return!!!
  if args.evaluate:
    validate(val_loader, model, criterion, None, args)
    return

  for epoch in range(args.start_epoch, args.epochs):
    if args.distributed:
      train_sampler.set_epoch(epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, writer, args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion, epoch, writer, args)

    scheduler.step()

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            #'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, log_dir)

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

def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
  main()
