# from data.LocalizeDataset import LocalizeDataset
from data_process.ClassifyDataset import ClassifyDataset
from torch.utils.data import DataLoader
from data_process.augment import get_transforms
from data_process.prepare_data import object_polar2cart


def get_dataset_by_type(args, is_train=False):
    """
    Train or eval/test is different for dataset because they have different dataset list.
    """
    dtype = args.data_type
    if dtype == 'localize':
        raise NotImplementedError
    elif dtype == 'classify':
        return ClassifyDataset(args, is_train)
    else:
        raise NotImplementedError


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, True)
    print('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader


def select_eval_loader(args):
    eval_dataset = get_dataset_by_type(args)
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset, args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
    return val_loader


