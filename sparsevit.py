import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import math
import warnings
try:
    import torch.nn.attention.varlen as varlen
    HAS_VARLEN_FLASH_ATTENTION = True
except ImportError:
    warnings.warn(
        "Could not import torch.nn.attention.varlen, variable length Flash Attention is disabled.",
        category=UserWarning,
        stacklevel=2,
    )
    HAS_VARLEN_FLASH_ATTENTION = False

enable_fa = os.environ.get('DISABLE_FA', '0').lower() not in {"1", "true", "yes", "y", "on"}
HAS_VARLEN_FLASH_ATTENTION = HAS_VARLEN_FLASH_ATTENTION and enable_fa

class SoftMaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
                 add_bias_kv=True, kdim=None, vdim=None,
                 scale=8., device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.scale = scale

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias and add_bias_kv, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias and add_bias_kv, **factory_kwargs)

        self.dropout_layer = nn.Dropout(dropout)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)

        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)


    def eager_forward(self, query, key, value, key_padding_mask=None,
            attn_mask=None, average_attn_weights=True):
        batch_size, tgt_len, embed_dim = query.size()
        batch_size, src_len, _ = key.size()

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape q, k, v for multihead attention
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1,2)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores shape: (batch_size, num_heads, tgt_len, src_len)

        # Apply the soft [0, 1] mask
        if attn_mask is not None:
            # Ensure attn_mask values are in (0, 1] to avoid log(0)
            # attn_mask shape [b, l]
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
            eps = 1e-6
            attn_mask_l = attn_mask.clip(min=eps).log()
            if not self.training:
                attn_mask_l = attn_mask_l.masked_fill((attn_mask == 0.), float('-inf'))
            attn_mask = attn_mask_l
            # attn_mask shape [b, 1, 1, l]
            scores = scores + self.scale * attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
            scores = scores.masked_fill(key_padding_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        # attn_output shape: (batch_size, num_heads, tgt_len, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output

    def flash_forward(
        self,
        query, key, value,
        cu_seq_q, cu_seq_k,
        max_q, max_k,
        attn_mask=None,
    ):
        """
        FlashAttention-compatible soft-masked attention using varlen_attn
        """

        q = self.q_proj(query)   # (Tq, H*D)
        k = self.k_proj(key)     # (Tk, H*D)
        v = self.v_proj(value)   # (Tk, H*D)

        Tq = q.shape[0]
        Tk = k.shape[0]

        q = q.view(Tq, self.num_heads, self.head_dim)
        k = k.view(Tk, self.num_heads, self.head_dim)
        v = v.view(Tk, self.num_heads, self.head_dim)

        # Apply the soft [0, 1] mask
        if attn_mask is not None:
            # attn_mask: (Tk,) or (B, Lk) flattened to match Tk
            eps = 1e-6
            attn_mask_l = attn_mask.clip(min=eps).log()
            if not self.training: # Inference mode can have infinite atten scores
                attn_mask_l = attn_mask_l.masked_fill((attn_mask == 0.), float('-inf'))
            log_m = attn_mask_l

            # Broadcast to (Tk, H, 1)
            log_m = log_m.view(Tk, 1, 1).expand(-1, self.num_heads, 1)
            k_zeros = torch.zeros_like(log_m).expand(-1, -1, 7)

            # Augment K and Q
            # We want:
            #   (qk^T)/sqrt(d) + scale * log(m)
            scale_attn = 1.0 / math.sqrt(self.head_dim)

            k_extra = log_m * (self.scale / scale_attn)

            k = torch.cat([k, k_extra, k_zeros], dim=-1)     # (Tk, H, D+1)

            v_zeros = torch.zeros(Tk, self.num_heads, 8, device=v.device, dtype=v.dtype)
            v = torch.cat([v, v_zeros], dim=-1)

            q_ones = torch.ones(
                Tq, self.num_heads, 8,
                device=q.device, dtype=q.dtype
            )
            q = torch.cat([q, q_ones], dim=-1)      # (Tq, H, D+1)

            attn_dim = self.head_dim + 1
        else:
            attn_dim = self.head_dim
            scale_attn = 1.0 / math.sqrt(self.head_dim)

        # FlashAttention varlen call
        out = varlen.varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            scale=scale_attn,
        )

        # Merge heads and output projection
        out = out[..., :self.head_dim]
        out = out.reshape(Tq, self.num_heads * self.head_dim)
        out = self.out_proj(out)

        return out

    def forward(self, query, key, value, method="eager", **kwargs):
        if method == 'eager':
            out = self.eager_forward(query, key, value, **kwargs)
        elif method == "fa":
            out = self.flash_forward(query, key, value, **kwargs)
        else:
            raise ValueError(f"No attention method named {method}.")
        return out


def get_ffn(input_dim, output_dim, middle_dim, dropout=0.1):
    fc1 = nn.Linear(input_dim, middle_dim)
    fc2 = nn.Linear(middle_dim, output_dim)
    fc3 = nn.Identity()
    return nn.Sequential(
            fc1,
            nn.GELU(),
            nn.Dropout(dropout),
            fc2,
            nn.Dropout(dropout),
            fc3
            )

# Assuming SoftMaskedMultiheadAttention is already defined as provided earlier
class EncoderBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, mlp_dim, dropout=0.1, drop_path=0.0, patch_drop=0.0, attention_scale=2., mask_threshold=0.05):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.self_attn = SoftMaskedMultiheadAttention(
            embed_dim, num_heads, dropout=dropout, scale=attention_scale
        )
        if attention_scale > 0:
            self.linear_mask = nn.Linear(input_dim, 1)  # Linear layer to compute mask scores
            self.patch_drop = nn.Dropout(patch_drop)
        else:
            self.linear_mask = None
        if input_dim != embed_dim:
            raise ValueError("embed_dim must equal atten_dim but {input_dim}!={embed_dim}")
        else:
            self.embed = nn.Identity()
            self.project = nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network (MLP)
        self.mlp = get_ffn(embed_dim, embed_dim, mlp_dim, dropout=dropout)
        self.path_drop = StochasticDepth(drop_path, mode='row')
        self.norm3 = nn.LayerNorm(input_dim)

    def _reset_parameters(self):
        for n, m in self.named_modules():
            if n.startswith('self_attn'):
                continue
            if isinstance(m, (nn.Linear, GroupedLinear)):
                nn.init.trunc_normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)
        nn.init.zeros_(self.norm3.weight)
        nn.init.zeros_(self.norm3.bias)

    def forward_common(self, x, mask, skip_masks=False):
        # Compute mask scores: (batch_size, seq_len, 1)
        x1 = x
        x = self.embed(x)
        x = self.norm1(x)
        # Apply attention mechanism
        attn_output = self.self_attn(x, x, x, attn_mask=mask if not skip_masks else None, method="eager")
        # Add & Norm
        x = x + self.path_drop(attn_output)
        x = self.norm2(x)
        # Feed-forward network
        mlp_output = self.mlp(x)
        # Add & Norm
        x = self.path_drop(self.project(x + mlp_output))
        x = self.norm3(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x1 + x
        return x

    def flash_forward(self, x, mask, skip_masks=False):
        # x: [B, N, C]
        # mask: [B, N]

        B, N, C = x.shape

        x_res = x  # residual

        binary_mask = mask >= self.mask_threshold
        seq_lengths = binary_mask.sum(dim=1, dtype=torch.int32)
        mean_len = seq_lengths.float().square().mean().sqrt().item()
        max_len = seq_lengths.amax().item()
        min_len = seq_lengths.amin().item()

        # Early exit if nothing selected
        if not binary_mask.any():
            return x

        # Check if nonselective or topk would be easier
        if ((mean_len / x.shape[1]) > 0.90):
            x_sel = x.flatten(0, 1)
            flat_idx = None

            if not skip_masks:
                sel_mask = mask.flatten(0, 1)
            else:
                sel_mask = None

            cu_seqlens = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=x.device)
        elif max_len > 32:
            # Regular selective model
            idx = binary_mask.nonzero(as_tuple=False)
            b_idx = idx[:, 0]
            t_idx = idx[:, 1]
            flat_idx = b_idx * N + t_idx

            # Pack selected tokens
            x_sel = x[b_idx, t_idx]

            if not skip_masks:
                sel_mask = mask[b_idx, t_idx]
            else:
                sel_mask = None

            # cu_seqlens for varlen FA
            cu_seqlens = torch.zeros(binary_mask.shape[0]+1, dtype=torch.int, device=binary_mask.device)
            cu_seqlens[1:] = seq_lengths.cumsum(-1)
        else:
            # Small kept lengths: use top-k packing, but keep varlen FA interface
            k = max_len

            # topk over score/mask values
            top_vals, top_idx = mask.topk(k, dim=1, largest=True, sorted=False)  # [B, k]
            b_idx = torch.arange(B, device=mask.device)[:, None].expand_as(top_idx)
            flat_idx = (b_idx * N + top_idx).reshape(-1)

            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, C)   # [B, k, C]
            x_top = x.gather(1, gather_idx)                        # [B, k, C]

            # Flatten, then keep only valid entries so packed layout matches varlen FA
            x_sel = x_top.flatten(0, 1)

            if not skip_masks:
                sel_mask = top_vals.flatten(0, 1)
            else:
                sel_mask = None

            cu_seqlens = torch.arange(0, (B + 1) * max_len, step=max_len, dtype=torch.int32, device=x.device)

        cu_seqlens = cu_seqlens.to(torch.int32)

        # Block
        x_sel = self.embed(x_sel)
        x_sel = self.norm1(x_sel)

        attn_output = self.self_attn(
            x_sel, x_sel, x_sel,
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_len,
            max_k=max_len,
            attn_mask=None if skip_masks else sel_mask,
            method="fa",
        )

        x_sel = x_sel + self.path_drop(attn_output)
        x_sel = self.norm2(x_sel)

        mlp_output = self.mlp(x_sel)
        x_sel = self.path_drop(self.project(x_sel + mlp_output))
        x_sel = self.norm3(x_sel)

        if sel_mask is not None:
            x_sel.mul_(sel_mask.unsqueeze(-1))

        # Scatter back directly into residual output
        if flat_idx is None:
            x_out = x_res + x_sel.view(*x_res.shape)
        else:
            B, N, C = x_res.shape
            flat_out = x_res.reshape(B * N, C)
            if torch.is_grad_enabled():
                flat_out = flat_out.clone()
            flat_out.index_add_(0, flat_idx, x_sel)
            x_out = flat_out.view(B, N, C)

        return x_out

    def get_groups(self, mask, full=False):
        n_items, index = (mask != 0.0).sum(-1).cpu().sort(descending=True)
        n_items, index = n_items.tolist(), index.tolist()
        groups = []
        t = 1.0 if full else 1.2
        for ni, ii in zip(n_items, index):
            if ni == 0:
                break
            if len(groups) == 0 or groups[-1][1] / ni > t:
                groups.append(([], ni))
            groups[-1][0].append(ii)
        return groups

    def eager_forward(self, x, mask, full=False, skip_masks=False):
        # Step 1: Threshold the mask without in-place ops
        mask_thresholded = mask * (mask >= self.mask_threshold)
        # Step 2: Prepare output tensor (copy of x)
        x_out = x.clone()
        # Step 3: Group samples by number of kept tokens
        groups = self.get_groups(mask_thresholded, full)
        # Step 4: Process each group
        for batch_indices, n_keep in groups:
            x_sel = x[batch_indices]                     # (Bg, seq_len, input_dim)
            mask_sel = mask_thresholded[batch_indices]   # (Bg, seq_len)
            # Top-k selection and sorting
            topk_vals, topk_idx_unsorted = torch.topk(mask_sel, k=n_keep, dim=1, sorted=False)
            topk_idx_sorted, _ = topk_idx_unsorted.sort(dim=1)
            # Gather tokens in sorted order
            idx_expanded = topk_idx_sorted.unsqueeze(-1).expand(-1, -1, x_sel.size(-1))
            X_topk = torch.gather(x_sel, dim=1, index=idx_expanded)
            mask_topk = torch.gather(mask_sel, dim=1, index=topk_idx_sorted)
            # Run forward pass
            results = self.forward_common(X_topk, mask_topk, skip_masks)
            # Scatter results into a new x_sel tensor
            x_sel_updated = x_sel.clone()
            x_sel_updated = x_sel_updated.scatter(1, idx_expanded, results)
            # Write the updated batch slice into the new output tensor
            x_out[batch_indices] = x_sel_updated
        return x_out

    def forward(self, x, full=False, skip_masks=False):
        if self.linear_mask is not None:
            attn_mask = self.patch_drop(self.linear_mask(x).sigmoid().squeeze(-1))
        else:
            attn_mask = None
        if not self.training and not attn_mask is None and self.mask_threshold >= 0:
            if (
                    HAS_VARLEN_FLASH_ATTENTION and
                    'cuda' in x.device.type and
                    x.dtype in (torch.bfloat16, torch.float16)
                    ):
                x = self.flash_forward(x, attn_mask, skip_masks)
            else:
                warnings.warn(
                    "Flash Attention requirements not met, falling back to eager attention.",
                    category=UserWarning,
                    stacklevel=2,
                )
                x = self.eager_forward(x, attn_mask, full, skip_masks)
        else:
            x = self.forward_common(x, attn_mask, skip_masks)
        return x, attn_mask


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        atten_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
        channels=3,
        dropout=0.1,
        drop_path=0.1,
        patch_drop=0.1,
        attention_scale=2.,
        mask_threshold=0.05,
        use_distil_token=False
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding layer
        self.patch_embed = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 + (1 if use_distil_token else 0), embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Encoder blocks
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(
                embed_dim, atten_dim,
                num_heads, mlp_dim,
                dropout, drop_path * i / (depth - 1),
                patch_drop=patch_drop,
                attention_scale=attention_scale,
                mask_threshold=mask_threshold,
                )
            for i in range(depth)
        ])

        # Classification head
        self.post_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        if use_distil_token:
            self.dis_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.dis_head = nn.Linear(embed_dim, num_classes)
        else:
            self.dis_token = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for n, m in self.named_modules():
            if n.startswith('encoder_layers'):
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight.data, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight.data)
                nn.init.zeros_(m.bias.data)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dis_token is not None:
            nn.init.trunc_normal_(self.dis_token, std=0.02)

    def forward_features(
        self,
        pixel_values,
        full=False,
        output_hidden_states=False,
        skip_masks=False
    ):
        batch_size = pixel_values.size(0)
        hidden_states = []

        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)

        # Distillation token
        if self.dis_token is not None:
            dis_tokens = self.dis_token.expand(batch_size, -1, -1)
            x = torch.cat((dis_tokens, x), dim=1)

        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Position + dropout
        x = x + self.pos_embed
        x = self.dropout(x)

        masks = []

        for layer in self.encoder_layers:
            x, mask = layer(x, full, skip_masks=skip_masks)

            if output_hidden_states:
                hidden_states.append(x)

            if mask is not None:
                masks.append(mask)

        x = self.post_norm(x)

        if output_hidden_states:
            hidden_states = tuple(hidden_states)
        else:
            hidden_states = None

        if len(masks) > 0:
            masks = tuple(masks)
        else:
            masks = None

        return x, hidden_states, masks


    def forward_classifier(self, hidden_states):
        cls_token = hidden_states[:, 0]
        logits = self.head(cls_token)

        dis_logits = None
        if self.dis_token is not None:
            dis_cls_token = hidden_states[:, 1]
            dis_logits = self.dis_head(dis_cls_token)

            if not self.training:
                logits = (logits + dis_logits) / 2

        return logits, dis_logits

    def forward(self, x, full=False, skip_masks=False):
        last_hidden_states, hidden_states, masks = self.forward_features(x, full, skip_masks=skip_masks)
        logits, dis_logits = self.forward_classifier(last_hidden_states)
        return logits, dis_logits, masks
