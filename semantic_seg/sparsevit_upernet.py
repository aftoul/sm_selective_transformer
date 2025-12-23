import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from collections import OrderedDict

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock, FeaturePyramidNetwork
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNConvFCHead, RPNHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNHeads, MaskRCNNPredictor

from sparsevit import VisionTransformer

import warnings
from collections import OrderedDict
from typing import Optional, Union

import torch
from torch import nn
from upernet import UPerNetDecoder, SegmentationHead


class SparseTransformerUPerNet(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_channels,
        decoder_channels: int = 256,
        in_channels: int = 3,
        classes: int = 1,
        upsampling: int = 1
    ):
        super().__init__()

        self.encoder = encoder

        self.decoder = UPerNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels,
            out_channels=classes,
            kernel_size=1,
            upsampling=upsampling,
        )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        features, masks = self.encoder(x)
        decoder_output = self.decoder(features)

        outputs = self.segmentation_head(decoder_output)

        return outputs, masks


class DeiTBackboneForFPN(nn.Module):
    """
    Wraps your custom DeiT model to expose intermediate feature maps
    at layers 3, 6, 9, 12 for use with an FPN.
    """

    def __init__(self, vit_model, feat_indices=[3, 6, 9, 12]):
        super().__init__()
        self.vit = vit_model
        self.feat_indices = set(feat_indices)

        # Determine patch resolution
        H = vit_model.image_size // vit_model.patch_size
        W = H
        self.grid_size = (H, W)

        self.deconvs = nn.ModuleList()
        channels = 16
        for i in range(len(feat_indices)):
            self.deconvs.append(
                    nn.ConvTranspose2d(
                        in_channels=vit_model.embed_dim,
                        out_channels=channels,
                        kernel_size=vit_model.patch_size//(2**i),
                        stride=vit_model.patch_size//(2**i)
                        )
                    )
            channels *= 2

    def interpolated_pos_embed(self, x_shape):
        pos_embed = self.vit.pos_embed
        if self.vit.dis_token is not None:
            first = 2
        else:
            first = 1
        patch_pos_embed = pos_embed[:, first:].permute(0, 2, 1).unflatten(2, self.grid_size)
        interpolated = F.interpolate(patch_pos_embed, (x_shape[-2], x_shape[-1]))
        interpolated = interpolated.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat([self.vit.pos_embed[:, :first], interpolated], dim=1)
        return new_pos_embed


    def forward(self, x):
        """
        Returns:
            dict[str, Tensor]  -- features for FPN (C, H, W)
        """
        B = x.size(0)

        # ---- Patch embedding ----
        x = self.vit.patch_embed(x)                   # (B, C, H', W')
        H, W = x.shape[-2:]
        pos_embed = self.interpolated_pos_embed(x.shape)
        x = x.flatten(2).transpose(1, 2)              # (B, N, C)

        # ---- Add DIS token ----
        offset = 1
        if self.vit.dis_token is not None:
            dis_tok = self.vit.dis_token.expand(B, -1, -1)
            x = torch.cat((dis_tok, x), dim=1)
            offset += 1

        # ---- Add CLS token ----
        cls_tok = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)

        # ---- Add positional encoding ----
        x = x + pos_embed
        x = self.vit.dropout(x)

        # ---- Collect features from layers ----
        feats = []
        c = 0
        masks = []
        for i, blk in enumerate(self.vit.encoder_layers, 1):
            x, mask = blk(x)
            if mask is not None:
                masks.append(mask)
            masks.append(mask)
            if i in self.feat_indices:
                # Remove cls (+ dis) tokens, reshape to feature map
                spatial = x[:, offset:, :]
                C = spatial.shape[-1]
                fmap = spatial.reshape(B, H, W, C).permute(0, 3, 1, 2)
                fmap = self.deconvs[c](fmap)
                feats.append(fmap)
                c += 1

        if len(masks) > 0:
            mask_mean = torch.stack(masks, dim=0)
        else:
            mask_mean = None

        return feats, mask_mean

def deit_fpn_extractor(
    backbone: nn.Module,
    feat_layers=[3, 6, 9, 12],
):
    # Wrap DeiT
    vit = DeiTBackboneForFPN(backbone, feat_indices=feat_layers)

    # All DeiT stages have same dim
    in_channels_list = [16, 32, 64, 128]
    return vit, in_channels_list

def upernet_from_backbone(base_model, feat_layers=[3, 6, 9, 12], num_classes=3, **kwargs):
    fpn_backbone, channels = deit_fpn_extractor(base_model, feat_layers=feat_layers)
    model = SparseTransformerUPerNet(
        fpn_backbone, channels,
        decoder_channels = 256,
        in_channels = 3,
        classes = num_classes,
        upsampling = 1
    )
    return model
