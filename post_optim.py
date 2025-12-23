from functools import partial
import argparse
import os
import sys
import math
import time
import json

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from fvcore.nn import FlopCountAnalysis

from tqdm import tqdm
from utils import model_from_file, get_dataloader
from timm.data import Mixup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a checkpoint file with an optional threshold."
    )
    parser.add_argument(
        "checkpoint_file",
        type=str,
        help="Checkpoint filename"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.05,
        help="Threshold value (default: 0.05)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to output checkpoint"
    )
    return parser.parse_args()


args = parse_args()

checkpoint_file = args.checkpoint_file
thresh = args.thresh

full_path = args.output_file

if os.path.isfile(full_path):
    print("Output file already exists", sys.stderr)
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, transforms = model_from_file(checkpoint_file, thresh, device, augment=True)
val_loader = get_dataloader('imagenet/train/', transforms, 32, shuffle=True, mixup=True, label_smoothing=0.11)

# Device configuration
model = model.to(device)

# Make LayerNorm parameters trainable
for param in model.parameters():
    param.requires_grad = True

trainable_params = []
for module in model.modules():
    if isinstance(module, nn.LayerNorm):
        trainable_params.extend(list(module.parameters()))

print(f"Trainable parameters {sum(p.numel() for p in trainable_params)}", file=sys.stderr)

# Define the scheduler function
def cosine_scheduler_with_warmup(optimizer, total_steps, warmup_steps, cooldown_steps, initial_lr=0.1, end_lr=0.01):
    warmup_steps = warmup_steps
    cooldown_steps = cooldown_steps
    cosine_steps = total_steps - warmup_steps - cooldown_steps

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return initial_lr + (1 - initial_lr) * (current_step / warmup_steps)
        elif current_step < cosine_steps + warmup_steps:
            # Cosine decay
            progress = (current_step - warmup_steps) / cosine_steps
            return end_lr + (1 - end_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return end_lr

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Optimizer with only LayerNorm parameters
total_samples = 200000
num_steps = math.ceil(total_samples / BATCH_SIZE)
optimizer = optim.Adam(trainable_params, lr=4e-3)
scheduler = cosine_scheduler_with_warmup(optimizer, num_steps, int(num_steps / 10), int(num_steps/10))

criterion = nn.CrossEntropyLoss()
model.eval()
cutmix_or_mixup = Mixup(
        mixup_alpha=0.2, cutmix_alpha=1., prob=0.8,
        switch_prob=0.5, mode='batch',
        label_smoothing=0.11,
        num_classes=1000
        )

num_epochs = 1
for epoch in range(num_epochs):
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, ylabels in tqdm(val_loader, desc=f"Epoch {epoch+1} [Train]"):
        images = images.to(device)
        ylabels = ylabels.to(device)
        images, labels = cutmix_or_mixup(images, ylabels)

        
        optimizer.zero_grad()
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
            outputs, _, masks = model(images, False)
            loss = criterion(outputs, labels)
            if masks is not None:
                masks = torch.stack(masks, dim=0)
                mask_loss = (
                        masks + 1 - (
                            (16 * masks).softmax(-1) * masks
                            ).sum(dim=-1, keepdim=True)).mean()
            else:
                mask_loss = 0
        (loss + 0.25 * mask_loss).backward()
        optimizer.step()
        
        train_loss += loss.item() * ylabels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += ylabels.size(0)
        correct += (predicted == ylabels).sum().item()
        if total >= total_samples:
            break
    
    train_acc = correct / total
    train_loss = train_loss / total
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}", file=sys.stderr)

os.makedirs(os.path.dirname(full_path), exist_ok=True)

torch.save({
    'config': torch.load(checkpoint_file, weights_only=True)['config'],
    'model': model.state_dict(),
    }, full_path)
