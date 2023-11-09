import os
import torch
import torchvision



def get_nbclasses(dataset):
    return {
        'cifar10': 10,
        'imagenet': 1000,
    }[dataset]

def get_dataset(dataset, split):
    return {
        'cifar10': get_cifar10,
        'imagenet': get_imagenet,
    }[dataset](split)



def get_cifar10(split):
    if split == 'train_repr' or split == 'eval_repr':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=(split == 'train_repr'), download=True)

        inv1_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur(kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (32, 32)), sigma=(0.1, 2.0))
            ], p=1.0),
            torchvision.transforms.RandomSolarize(threshold=128.0, p=0.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        inv2_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur(kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (32, 32)), sigma=(0.1, 2.0))
            ], p=0.1),
            torchvision.transforms.RandomSolarize(threshold=128.0, p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

        return AugDatasetWrapper(dataset, inv1_transform, inv2_transform)

    elif split == 'train_clsf' or split == 'eval_clsf':
        if split == 'train_clsf':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=(32, 32), scale=(0.08, 1.0), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(36, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(32),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
            ])

        return torchvision.datasets.CIFAR10(root='./data', train=(split == 'train_clsf'), transform=transform)

    raise Exception('Unknown split for CIFAR10: {}'.format(split))



def get_imagenet(split):
    if split == 'train_repr' or split == 'eval_repr':
        dataset = torchvision.datasets.ImageNet(root=(os.environ['DSDIR'] + '/imagenet'), split='train' if split == 'train_repr' else 'val')

        inv1_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0))
            ], p=1.0),
            torchvision.transforms.RandomSolarize(threshold=128.0, p=0.0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        inv2_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([
                torchvision.transforms.GaussianBlur(kernel_size=(23, 23), sigma=(0.1, 2.0))
            ], p=0.1),
            torchvision.transforms.RandomSolarize(threshold=128.0, p=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return AugDatasetWrapper(dataset, inv1_transform, inv2_transform)

    elif split == 'train_clsf' or split == 'eval_clsf':
        if split == 'train_clsf':
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(size=(224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        formal_split = 'val' if split == 'eval_clsf' else 'train'
        return torchvision.datasets.ImageNet(root=(os.environ['DSDIR'] + '/imagenet'), split=formal_split, transform=transform)

    raise Exception('Unknown split for ImageNet: {}'.format(split))





class AugDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, inv1_transform, inv2_transform):
        self.dataset = dataset
        self.inv1_transform = inv1_transform
        self.inv2_transform = inv2_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img_1 = self.inv1_transform(img)
        img_2 = self.inv2_transform(img)

        return ((img_1, img_2), label)
