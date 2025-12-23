from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
from config import load_yaml_config, create_augmentations
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torchvision.datasets.folder import default_loader

from timm.data import Mixup
from torch.utils.data import default_collate


def create_dataloaders(batch_size, device, config, teacher_transform):
    # ImageNet dataset paths (adjust as needed)
    train_transform, val_transform = create_augmentations(config['augmentation'])
    train_dir = config['train_data']
    val_dir = config['val_data']

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset =   datasets.ImageFolder(root=val_dir, transform=val_transform)

    if config['augmentation']['cutmix_mixup']:
        cutmix_or_mixup = Mixup(
            mixup_alpha=0.2, cutmix_alpha=1., prob=0.9,
            switch_prob=0.5, mode='batch',
            label_smoothing=config['training']['label_smoothing'],
            num_classes=1000
            )
        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))
    else:
        collate_fn = default_collate

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=8,
        drop_last=False
    )

    # Wrap with PerDeviceLoader
    train_per_device_loader = pl.MpDeviceLoader(train_loader, device)
    val_per_device_loader = pl.MpDeviceLoader(val_loader, device)

    return train_per_device_loader, val_per_device_loader

