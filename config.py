import yaml
import os
from pathlib import Path
from collections.abc import Mapping

import torchvision.transforms as transforms

default_config = {
        'model': {
            'image_size': 224,
            'patch_size': 16,
            'num_classes': 1000,
            'embed_dim': 768,
            'atten_dim': 384,
            'depth': 12,
            'num_heads': 3,
            'mlp_dim': 768,
            'channels': 3,
            'dropout': 0.0,
            'drop_path': 0.1,
            'attention_scale': 0.0,
            'mask_threshold': 0.05,
            },
        'training': {
            'batch_size': 1024,
            'epochs': 350,
            'warmup_epochs': 5,
            'cooldown_epochs': 50,
            'wd': 0.2,
            'max_grad_norm': 1.0,
            'alpha': 0.7,
            't': 4.0,
            'label_smoothing': 0.1,
            'max_mask_weight': 0.25,
            'mask_scale': 8.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'model_ema_steps': -1,
            'model_ema_decay': 0.999,
            'loss_term1': True,
            'loss_term2': True,
            },
        'augmentation': {
            'resize': 232,
            'crop': 224,
            'three_aug': False,
            'color_jitter': False,
            'random_resized_crop': False,
            'hflip': False,
            'random_erasing': False,
            'rand_augment': False,
            }
        }

def merge_dicts(dict1, dict2):
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_yaml_config(yaml_file):
    """
    Load model, training, and data augmentation information from a YAML file.

    Args:
        yaml_file (str or Path): Path to the YAML file.

    Returns:
        dict: Dictionary containing model, training, augmentation, and checkpoint info.
    """
    yaml_file = Path(yaml_file)

    if not yaml_file.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    if "base" in config and config["base"] is not None:
        base = load_yaml_config(
                os.path.join(os.path.dirname(yaml_file), config["base"])
                )
    else:
        base = default_config

    config = merge_dicts(base, config)
    # Calculate dynamic fields
    if "training" not in config or config["training"] is None:
        config["training"] = {}
    if not "batch_size" in config["training"]:
        config["training"]["batch_size"] = 512
    if not "lr" in config["training"]:
        config["training"]["lr"] = (
            config["training"]["batch_size"] * 5e-4 / 512
        )
    # Return the structured dictionary
    return config

def create_augmentations(augmentation_config):
    """
    Create training and validation augmentations based on configuration.

    Args:
        augmentation_config (dict): Dictionary containing augmentation parameters.

    Returns:
        tuple: Training and validation augmentation pipelines.
    """
    resize = augmentation_config.get("resize", 232)
    crop = augmentation_config.get("crop", 224)
    random_resized_crop = augmentation_config.get("random_resized_crop", False)
    hflip = augmentation_config.get("hflip", False)
    random_erasing = augmentation_config.get("random_erasing", False)
    color_jitter = augmentation_config.get("color_jitter", False)
    three_aug = augmentation_config.get("three_aug", False)
    rand_aug = augmentation_config.get("rand_augment", False)

    train_transforms = []
    if random_resized_crop:
        train_transforms.append(transforms.RandomResizedCrop(crop))
    else:
        train_transforms.append(transforms.Resize(resize))
        train_transforms.append(transforms.CenterCrop(crop))

    if hflip:
        train_transforms.append(transforms.RandomHorizontalFlip(hflip))
    if rand_aug:
        train_transforms.append(transforms.RandAugment())

    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if random_erasing:
        train_transforms.append(transforms.RandomErasing(p=random_erasing))
    if color_jitter:
        train_transforms.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    if three_aug:
        train_transforms.append(
                transforms.RandomChoice([
                    transforms.Grayscale(3),
                    transforms.GaussianBlur(11),
                    transforms.RandomInvert(1.)
                    ])
                )

    val_transforms = [
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)

