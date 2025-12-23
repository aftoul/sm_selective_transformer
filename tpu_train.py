from functools import partial
import os
from glob import glob
from collections import defaultdict
import math
from urllib.parse import urlparse
import random
import time
from itertools import chain

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models
from sparsevit import VisionTransformer
from dataloader import create_dataloaders

from tqdm import tqdm

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import argparse
from config import load_yaml_config, create_augmentations

DS_SIZE = 1281167

def collective_broadcast(tensors, groups=None, pin_layout=None):
  with torch.no_grad():
    # We must produce the exact same graph in each replica to prevent hanging,
    # so each replica must have the same multiply op with the same parameters.
    for tensor in tensors:
      scale = torch.tensor(
          1 if xm.is_master_ordinal(local=False) else 0,
          dtype=tensor.dtype)
      # Transfer scale tensor as device data instead of constant 1 or 0.
      xscale = xm.send_cpu_data_to_device(scale, tensor.device)
      tensor.mul_(xscale)

  xm.all_reduce(xm.REDUCE_SUM, tensors, groups=groups, pin_layout=pin_layout)

def broadcast_master_param(model):
  parameters_and_buffers = list(
      chain(model.parameters(), model.buffers()))
  collective_broadcast(parameters_and_buffers)
  xm.mark_step()

# Knowledge distillation loss function
def distillation_loss(y_student, y_distil, y_teacher, labels, temperature, alpha, label_smoothing, hard_labels):
    """
    Combines cross-entropy loss with distillation loss.

    y_student: Output logits from the student model
    y_teacher: Output logits from the teacher model
    labels: Ground truth labels
    temperature: Temperature parameter for distillation
    alpha: Weight between distillation loss and classification loss
    """
    if y_distil is None:
        y_distil = y_student
    # Cross-entropy loss with ground truth labels
    ce_loss = F.cross_entropy(y_student, labels, label_smoothing=label_smoothing)
    
    if y_teacher is not None:
        # Soft targets from teacher
        if hard_labels:
            d_loss = F.cross_entropy(y_distil, y_teacher.argmax(-1))
        else:
            p_teacher = nn.functional.softmax(y_teacher / temperature, dim=1)
            # Log probabilities from student
            p_student_log = nn.functional.log_softmax(y_distil / temperature, dim=1)
            # KL divergence loss
            d_loss = nn.KLDivLoss(reduction='batchmean')(p_student_log, p_teacher) * (temperature ** 2)
        # Combined loss
        loss = alpha * d_loss + (1 - alpha) * ce_loss
    else:
        loss = ce_loss
    return loss

# Define the scheduler function
def cosine_scheduler_with_warmup(optimizer, total_epochs, steps_per_epoch, warmup_epochs, cooldown_epochs, initial_lr=0.1, end_lr=0.01):
    warmup_steps = warmup_epochs * steps_per_epoch
    cooldown_steps = cooldown_epochs * steps_per_epoch
    cosine_steps = total_epochs * steps_per_epoch - warmup_steps - cooldown_steps

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


def filter_parameters(module, exclude):
    no_decay_params = []
    decay_params = []

    for name, param in module.named_parameters():
        if name in exclude or name.endswith('bias') or ('.' in name and isinstance(module.get_submodule(name.rsplit('.', 1)[0]), nn.LayerNorm)):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return decay_params, no_decay_params

def checkpoint_load(checkpoint_file, student_model, model_ema, optimizer, scheduler):
    start_epoch = 0
    train_history = []
    test_history = []
    ema_history = []
    time_history = []
    if os.path.isfile(checkpoint_file):
        st_dict = torch.load(checkpoint_file, weights_only=True)
        student_model.load_state_dict(st_dict['model'])
        optimizer.load_state_dict(st_dict['optimizer'])
        scheduler.load_state_dict(st_dict['scheduler'])
        if st_dict['ema'] is not None and model_ema is not None:
            model_ema.load_state_dict(st_dict['ema'])
        time_history, train_history, test_history, ema_history = st_dict['history']
        start_epoch = st_dict['epoch']
    return start_epoch, time_history, train_history, test_history, ema_history


def update_mask_weight(mask_weight, epoch, max_weight, epochs, warmup_epochs):
    if epoch < warmup_epochs:
        val = 0.
    elif epoch < epochs + 20:
        val = max_weight * min((epoch - warmup_epochs + 1) / (epochs - warmup_epochs), 1)
    else:
        val = max_weight
    mask_weight.fill_(val)
    return mask_weight


def eval_on_val(val_loader, student_model, device):
    student_model.eval()
    val_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():
        for (images, labels) in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            student_outputs, _, _ = student_model(images)
            loss = criterion(student_outputs, labels)

            val_loss += loss * labels.size(0)
            _, predicted = torch.max(student_outputs.data, 1)
            correct += (predicted == labels).float().sum()
            total += labels.size(0)

    global_val_loss = xm.mesh_reduce("val_loss", val_loss.item(), sum)
    global_correct = xm.mesh_reduce("val_correct", correct.item(), sum)
    global_total = xm.mesh_reduce("val_total", total, sum)

    avg_val_loss = global_val_loss / global_total
    val_accuracy = 100.0 * global_correct / global_total

    return avg_val_loss, val_accuracy


def get_teacher_model(tc):
    if tc['name'] == 'swin_v2_b':
        teacher_model = models.swin_v2_b(weights='IMAGENET1K_V1')
    elif tc['name'] == 'swin_b':
        teacher_model = models.swin_b(weights='IMAGENET1K_V1')
    elif tc['name'] == 'regnet_y_16gf':
        teacher_model = models.regnet_y_16gf(weights='IMAGENET1K_V2')
    elif tc['name'] == 'resnet152':
        teacher_model = models.resnet152(weights='IMAGENET1K_V2')
    elif tc['name'] == 'vit_l_16':
        teacher_model = models.vit_l_16(weights='IMAGENET1K_V1')
    elif tc['name'] == 'convnext_base':
        teacher_model = models.convnext_base(weights='IMAGENET1K_V1')
    elif tc['name'] == 'wide_resnet101_2':
        teacher_model = models.wide_resnet101_2(weights='IMAGENET1K_V2')
    else:
        raise ValueError(f"Unknown teacher model: {tc['name']}")

    teacher_transforms = None
    if 'transforms' in tc and tc['transforms'] is not None:
        teacher_transforms, _ = create_augmentations(tc['transforms'])
    return teacher_model, teacher_transforms
        


SERIAL_EXEC = xmp.MpSerialExecutor()

def train(device_id, config):
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    SERIAL_EXEC.run(
            lambda: print(f'Device {str(device)} is rank {rank} of {world_size}')
            )

    if xm.is_master_ordinal():
        os.makedirs(os.path.dirname(config['checkpoint']), exist_ok=True)

    training_args = config['training']

    batch_size = training_args['batch_size'] // world_size
    # Load pre-trained ViT model as teacher
    if 'teacher' in config and config['teacher'] is not None:
        teacher_model, teacher_transforms = SERIAL_EXEC.run(
                lambda: get_teacher_model(config['teacher']))
    else:
        teacher_model, teacher_transforms = None, None

    # Build datasets
    train_loader, val_loader = create_dataloaders(
            batch_size, device, config, teacher_transforms)
    xm.rendezvous("loaded dataset")

    # Define the student model
    student_model = VisionTransformer(
            **config['model']
    )
    if teacher_model is not None:
        teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    broadcast_master_param(student_model)
    if config['resume'] is not None:
        checkpoint = SERIAL_EXEC.run(lambda: torch.load(config['resume'], weights_only=True))
        st_dict = checkpoint['model']
        old_st_dict = student_model.state_dict()
        keys = []
        for k in st_dict:
            if k in old_st_dict and st_dict[k].shape != old_st_dict[k].shape:
                keys.append(k)
        for k in keys:
            del st_dict[k]
        student_model.load_state_dict(st_dict, strict=False)
    model_params = sum(p.numel() for p in student_model.parameters())

    decay_params, no_decay_params = filter_parameters(student_model, ['cls_token', 'pos_embed'])

    xm.master_print(f'Model parameters: {model_params:,d}')

    # Set up the optimizer
    optimizer = optim.AdamW(
            [
                {'params': decay_params, 'weight_decay': training_args['wd']},
                {'params': no_decay_params, 'weight_decay': 0.0}
                ],
            lr=training_args['lr'],
            betas=(training_args['adam_beta1'], training_args['adam_beta2'])
            )

    scheduler = cosine_scheduler_with_warmup(
        optimizer,
        total_epochs=training_args['epochs'],
        steps_per_epoch=len(train_loader),
        warmup_epochs=training_args['warmup_epochs'],
        cooldown_epochs=training_args['cooldown_epochs'],
        initial_lr=0.001,
        end_lr=0.02
    )

    # Create the EMA model
    if training_args['model_ema_steps'] > 0:
        decay = training_args['model_ema_decay']
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param
        model_ema = optim.swa_utils.AveragedModel(
                student_model,
                avg_fn=ema_avg
                )
    else:
        model_ema = None

    xm.rendezvous("loaded model and optimizer")

    (start_epoch,
     time_history,
     train_history,
     test_history,
     ema_history) = SERIAL_EXEC.run(
                    lambda: checkpoint_load(
                        config['checkpoint'],
                        student_model,
                        model_ema,
                        optimizer,
                        scheduler)
                    )
    
    xm.rendezvous("loaded weights")
    xm.master_print("training begins")

    mask_weight = torch.zeros((), device=device)

    loss_term1, loss_term2 = 0, 0
    for epoch in range(start_epoch, training_args['epochs']):
        xm.master_print(f"Starting epoch {epoch + 1}")
        start_time = time.time()
        student_model.train()
        total_loss = torch.zeros((), device=device)
        total_mask = torch.zeros((), device=device)
        local_total_batches = 0
        update_mask_weight(
                mask_weight, epoch,
                training_args['max_mask_weight'],
                training_args['epochs'] - training_args['cooldown_epochs'],
                training_args['warmup_epochs']
                )
        if xm.is_master_ordinal():
            train_loader = tqdm(train_loader)
        for step, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)
            
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(images)
            else:
                teacher_outputs = None
            # Get student predictions
            student_outputs, distil_token, masks = student_model(images)
            if masks is not None and (training_args['loss_term1'] or training_args['loss_term2']):
                if training_args['loss_term1']:
                    loss_term1 = masks
                if training_args['loss_term2']:
                    loss_term2 = 1 - (
                            (training_args['mask_scale'] * masks).softmax(-1)
                            * masks
                            ).sum(dim=-1, keepdim=True)
                mask_loss = (loss_term1 + loss_term2).mean()
                weighted_mask_loss = mask_loss * mask_weight
            else:
                mask_loss = 0
                weighted_mask_loss = 0.

            # Compute distillation loss
            loss = weighted_mask_loss + distillation_loss(
                    student_outputs, distil_token, teacher_outputs,
                    labels, training_args['t'],
                    training_args['alpha'],
                    0.0,
                    training_args.get('hard_labels', False),
                    )
            optimizer.zero_grad()
            loss.backward()
            # Do optimizer step with gradient clipping
            xm.reduce_gradients(optimizer, pin_layout=True)
            torch.nn.utils.clip_grad_norm_(
                    student_model.parameters(), training_args['max_grad_norm']
                    )
            optimizer.step()
            scheduler.step()
            if model_ema is not None and (step % training_args['model_ema_steps']) == 0:
                model_ema.update_parameters(student_model)
                if epoch < training_args['warmup_epochs']:
                    # Reset ema buffer to keep copying weights during warmup period
                    model_ema.n_averaged.fill_(0)
            xm.mark_step()
            with torch.no_grad():
                total_loss += loss * images.size(0)
                total_mask += mask_loss * images.size(0)
                local_total_batches += images.size(0)

        # Aggregate metrics across devices
        xm.master_print()
        global_loss = xm.mesh_reduce(
                "total_loss", total_loss.item(), sum)
        global_mask_loss = xm.mesh_reduce(
                "total_mask_loss", total_mask.item(), sum)
        global_batches = xm.mesh_reduce(
                "total_batches", local_total_batches, sum)

        average_loss = global_loss / global_batches
        average_mask_loss = global_mask_loss / global_batches

        xm.master_print(f"Epoch [{epoch+1}/{training_args['epochs']}], "
                        f"Training Loss: {average_loss:.4f}, "
                        f"Mean Mask: {average_mask_loss:.4f}")

        # Evaluation loop
        val_loss, val_accuracy = eval_on_val(
                val_loader, student_model, device)
        time_taken = time.time() - start_time
        xm.master_print(f"Validation Loss: {val_loss:.4f}, "
                        f"Validation Accuracy: {val_accuracy:.4f}%")
        xm.master_print(f"Epoch time: {time_taken:.2f}s")

        train_history.append([average_loss, average_mask_loss])
        test_history.append([val_loss, val_accuracy])
        time_history.append(time_taken)
        if model_ema is not None:
            ema_loss, ema_accuracy = eval_on_val(
                    val_loader, model_ema, device)
            xm.master_print(f"EMA Validation Loss: {ema_loss:.4f}, "
                            f"EMA Validation Accuracy: {ema_accuracy:.4f}%")
            ema_history.append([ema_loss, ema_accuracy])
        xm.save({
            'model': student_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema': model_ema.state_dict() if model_ema is not None else None,
            'history': [time_history, train_history, test_history, ema_history],
            'epoch': epoch + 1,
            'config': config
        }, config['checkpoint'])

    xm.master_print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load YAML configuration and create augmentations.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data folder")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data folder")
    parser.add_argument("--fine_tune", type=str, required=False, help="Path to a checkpoint to be used as a starting point")
    args = parser.parse_args()

    # Load configuration
    config = load_yaml_config(args.config)
    config['train_data'] = args.train_data
    config['val_data'] = args.val_data
    config['resume'] = args.fine_tune

    xmp.spawn(train, args=(config,), nprocs=None, start_method='fork')
