from torch.utils.tensorboard import SummaryWriter
#import wandb
import os
import torch
import pprint, yaml
import math

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


class Logger:
    def __init__(self, args):
        r'''
        If you use wandb, you can just use wandb.log() to log everything.
        '''
        # wandb init
        columns=["image", "prediction", "truth", "filename"]
        #if args.train:
        #    self.train_table = wandb.Table(columns=columns)
        #self.val_table = wandb.Table(columns=columns)

        # wandb Sweep
        self.sweep_id = None
        '''
        if args.wandb_sweep_path != '':
            config = Sweep.read_sweep_config(args.wandb_sweep_path)
            self.sweep_id = wandb.sweep(sweep=config, project='drowning_detection')
        else:
            # change this to your own project name
            if args.enable_wandb:
                self.writer = wandb.init(project='underwater', name=args.save_prefix)
            else:
                self.writer = wandb.init(project='underwater', name=args.save_prefix, mode='disabled')
        '''
        # model save
        self.model_dir = args.model_dir


    def save_imgs(self, imgs, preds, labels, filenames, is_train):
        ''' This function cannot use now due to memory issue.
        '''
        return
        if is_train:
            for idx in range(imgs.shape[0]):
                if idx > 64:
                    break
                self.train_table.add_data(
                                    (imgs[idx].cpu().numpy().transpose(1, 2, 0)),
                                    int(preds[idx].cpu().numpy()),
                                    int(labels[idx].cpu().numpy()),
                                    str(filenames[idx])
                                )
            #wandb.log({"train_imgs": self.train_table})
                
        else:
            for idx in range(imgs.shape[0]):
                if idx > 64:
                    break
                self.val_table.add_data(
                                    (imgs[idx].cpu().numpy().transpose(1, 2, 0)),
                                    int(preds[idx].cpu().numpy()),
                                    int(labels[idx].cpu().numpy()),
                                    str(filenames[idx])
                                )
            #wandb.log({"val_imgs": self.val_table})

    def save_check_point(self, model, epoch, step=0):
        model_name = '{epoch:03d}_{step:04d}.pth'.format(epoch=epoch, step=step)
        path = os.path.join(self.model_dir, model_name)
        latest_path = os.path.join(self.model_dir, 'latest.pth')
        # save model state dict
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        #torch.save(state_dict, path)
        torch.save(state_dict, latest_path)


class Sweep:
    def __init__(self, args):
        self.read_sweep_config(args.wandb_sweep_path)
    
    def read_sweep_config(path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
