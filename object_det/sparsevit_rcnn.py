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


class FasterTransformerRCNN(FasterRCNN):
    def forward(
        self,
        images: list[torch.Tensor],
        targets: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(
                            False,
                            f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                        )

        original_image_sizes: list[tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features, masks = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(
            detections, images.image_sizes, original_image_sizes
        )  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections, masks
        else:
            return self.eager_outputs(losses, detections), masks


class AttnBackboneWithFPN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        in_channels_list,
        out_channels,
        extra_blocks=None,
    ) -> None:
        super(AttnBackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.body = backbone
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x, masks = self.body(x)
        x = self.fpn(x)
        return x, masks


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
        feats = OrderedDict()
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
                feats[str(c)] = fmap
                c += 1

        if len(masks) > 0:
            mask_mean = torch.stack(masks, dim=0)
        else:
            mask_mean = None

        return feats, mask_mean

def deit_fpn_extractor(
    backbone: nn.Module,
    out_channels=256,
    feat_layers=[3, 6, 9, 12],
    extra_blocks=None,
    norm_layer=None
):
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    # Wrap DeiT
    vit = DeiTBackboneForFPN(backbone, feat_indices=feat_layers)

    # All DeiT stages have same dim
    in_channels_list = [16, 32, 64, 128]

    fpn = AttnBackboneWithFPN(
        vit,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=extra_blocks,
    )
    return fpn

def faster_rcnn_from_backbone(base_model, feat_layers=[3, 6, 9, 12], num_classes=3, **kwargs):
    fpn_backbone = deit_fpn_extractor(base_model, out_channels=256, feat_layers=feat_layers)
    # STEP 2 — Anchor generator
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # STEP 3 — Heads
    rpn_head = RPNHead(
        fpn_backbone.out_channels,
        anchor_generator.num_anchors_per_location()[0],
        conv_depth=2
    )

    norm_layer = partial(nn.GroupNorm, 1)

    box_head = FastRCNNConvFCHead(
        (fpn_backbone.out_channels, 7, 7), [256]*8, [1024],
        norm_layer=norm_layer
    )

    box_pred = FastRCNNPredictor(1024, num_classes)

    # STEP 4 — Build Mask R-CNN
    model = FasterTransformerRCNN(
        fpn_backbone,
        rpn_anchor_generator=anchor_generator,
        num_classes=None,
        rpn_head=rpn_head,
        box_head=box_head,
        box_predictor=box_pred,
        box_detections_per_img=100,
        box_score_thresh=0.05,
        **kwargs
    )
    return model
