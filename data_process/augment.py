import torchvision.transforms as transforms


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class EraseChannel:
    def __init__(self, erase_channel=[0, 1]):
        self.erase_channel = erase_channel

    def __call__(self, x):
        # x should be a tensor [C, H, W]
        # assert max(self.erase_channel) < x.shape[0]
        if x.shape[0] == 1:
            return x
        for channel in self.erase_channel:
            x[channel, :, :] = 0
        return x
    
def get_transforms(transform_type):
    data_transforms = {
        'none': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 128)),
        ]),
        'denoise': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ]),
        'random': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomChoice([
                # transforms.Resize((256, 128)),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=5, scale=(0.7, 1.3)),
                transforms.RandomRotation(degrees=10),
                transforms.CenterCrop([128, 64]),
                # transforms.RandomResizedCrop(size=(256, 128), scale=(0.6, 1.4)),
                # transforms.RandomErasing(p=0.6),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
                EraseChannel(erase_channel=[0, 1]),
                EraseChannel(erase_channel=[0]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.AugMix(severity=1, alpha=0.3, mixture_width=1),
                    transforms.ToTensor(),
                ]),
            ]),
            transforms.Resize((256, 128)),
        ]),
        'clean_random':transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomChoice([
                # transforms.Resize((256, 128)),
                transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=3, scale=(0.7, 1.3)),
                transforms.RandomRotation(degrees=7),
                transforms.CenterCrop([128, 64]),
                # transforms.RandomResizedCrop(size=(256, 128), scale=(0.6, 1.4)),
                # transforms.RandomErasing(p=0.6),
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
                EraseChannel(erase_channel=[0, 1]),
                EraseChannel(erase_channel=[0]),
            ]),
            transforms.Resize((256, 128)),
        ]),
        'reconstruct': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomChoice([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=3, scale=(0.7, 1.3)),
                transforms.RandomRotation(degrees=7),
                transforms.CenterCrop([128, 64]),
            ]),
            transforms.Resize((128, 64)),
        ]),
        'reconstruct_clean': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 64)),
        ]),
        'totensor': transforms.ToTensor(),
    }
    if transform_type == 'two_crop': 
        return TwoCropTransform(data_transforms['clean_random'])

    return data_transforms[transform_type]
