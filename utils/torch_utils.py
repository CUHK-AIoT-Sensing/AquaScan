from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

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


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def load_match_dict(model, model_path):
    # model: single gpu model, please load dict before warp with nn.DataParallel
    pretrain_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # the pretrain dict may be multi gpus, cleaning
    pretrain_dict = {k.replace('module.', ''): v for k, v in pretrain_dict.items()}
    # 1. filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if
                       k in model_dict and v.shape == model_dict[k].shape}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrain_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    

def save_model(model, optimizer, devices_ids, epoch, save_file):
    print('==> Saving...')
    if len(devices_ids) > 1:
        state = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    elif len(devices_ids) == 1 or len(devices_ids) == 0:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, save_file)
    del state

def allocate_devices(model, device_ids):
    if len(device_ids) >= 1:
        model = nn.DataParallel(model.cuda(), device_ids=device_ids, output_device=device_ids[0])
    elif len(device_ids) == 0:
        model = model.to('cpu')
    return model

def freeze_layers(model, freeze_layers):
    for name, param in model.named_parameters():
        if name in freeze_layers:
            param.requires_grad = False
    return model
