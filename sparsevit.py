import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth
import math

class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, groups=1, bias=True, reverse_groups=False, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.reverse_groups = reverse_groups
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        assert in_features % groups == 0, (f"Input features ({in_features}) "
                                           f"must be divisible by number of groups ({groups})")
        assert out_features % groups == 0, (f"Output features ({out_features}) "
                                            f"must be divisible by number of groups ({groups})")
        if self.reverse_groups:
            self.reversed_groups = groups
            self.groups = min(in_features, out_features)//groups
        else:
            self.reversed_groups = None
            self.groups = groups
        self.weight = nn.Parameter(
            torch.empty((self.groups, out_features//self.groups, in_features//self.groups), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.reversed_groups is not None:
            x = x.unflatten(-1, (self.reversed_groups, -1))
            x = x.transpose(-1, -2).flatten(-2, -1)
        x = x.unflatten(-1, (self.groups, -1))
        x = torch.einsum('...gi,goi->...go', x, self.weight)
        return x.flatten(-2, -1)

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

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, average_attn_weights=True):
        """
        query, key, value: shape (L, N, E)
        where L is the sequence length, N is the batch size, E is the embedding dimension.
        """
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
            if not self.training:
                scores = scores.masked_fill((attn_mask == 0.), float('-inf'))
            eps = 1e-6
            attn_mask = attn_mask.clip(min=eps).log()
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

        if need_weights:
            # Optionally average attention weights over heads
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            else:
                attn_weights = attn_weights
        else:
            attn_weights = None

        return attn_output, attn_weights

def get_ffn(input_dim, output_dim, middle_dim, groups=None, dropout=0.1):
    assert groups is None or isinstance(groups, int), 'FFN groups must be an integer or None'
    if groups is None or groups == 1:
        fc1 = nn.Linear(input_dim, middle_dim)
        fc2 = nn.Linear(middle_dim, output_dim)
        fc3 = nn.Identity()
    else:
        fc1 = GroupedLinear(input_dim, middle_dim, groups=groups)
        fc2 = GroupedLinear(middle_dim, output_dim, groups=groups, reverse_groups=True)
        fc3 = nn.Linear(output_dim, output_dim)
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
    def __init__(self, input_dim, num_groups, embed_dim, num_heads, mlp_dim, dropout=0.1, drop_path=0.0, patch_drop=0.0, attention_scale=2., mask_threshold=0.05, ffn_groups=None):
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
            self.embed = nn.Sequential(
                    GroupedLinear(input_dim, embed_dim, num_groups),
                    nn.GELU()
                    )
            self.project = nn.Sequential(
                    GroupedLinear(embed_dim, input_dim, num_groups),
                    nn.GELU()
                    )
        else:
            self.embed = nn.Identity()
            self.project = nn.Identity()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-forward network (MLP)
        self.mlp = get_ffn(embed_dim, embed_dim, mlp_dim, groups=ffn_groups, dropout=dropout)
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

    def forward_common(self, x, mask):
        """
        x: shape (batch_size, seq_len, embed_dim)
        """
        # Compute mask scores: (batch_size, seq_len, 1)
        x1 = x
        x = self.embed(x)
        x = self.norm1(x)
        # Apply attention mechanism
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
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

    def infer_forward(self, x, mask, full=False):
        """
        The “sparse‐inference” path: for each group of batch‐samples that have the same
        number n of tokens ≥ mask_threshold, gather only those top‐n tokens (in original order),
        run forward_common on the smaller (b’, n, dim) tensor, then scatter the results back.
        Fully masked tokens are left untouched.
        """
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
            results = self.forward_common(X_topk, mask_topk)
            # Scatter results into a new x_sel tensor
            x_sel_updated = x_sel.clone()
            x_sel_updated = x_sel_updated.scatter(1, idx_expanded, results)
            # Write the updated batch slice into the new output tensor
            x_out[batch_indices] = x_sel_updated
        return x_out

    def forward(self, x, full=False):
        if self.linear_mask is not None:
            attn_mask = self.patch_drop(self.linear_mask(x).sigmoid().squeeze(-1))
        else:
            attn_mask = None
        if not self.training and not attn_mask is None and self.mask_threshold >= 0:
            x = self.infer_forward(x, attn_mask, full)
        else:
            x = self.forward_common(x, attn_mask)
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
        num_groups=6,
        mlp_dim=768,
        channels=3,
        dropout=0.1,
        drop_path=0.1,
        patch_drop=0.1,
        attention_scale=2.,
        mask_threshold=0.05,
        ffn_groups=None,
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
                embed_dim, num_groups, atten_dim,
                num_heads, mlp_dim,
                dropout, drop_path * i / (depth - 1),
                patch_drop=patch_drop,
                attention_scale=attention_scale,
                mask_threshold=mask_threshold,
                ffn_groups=ffn_groups
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

    def forward(self, x, full=True):
        """
        x: shape (batch_size, channels, height, width)
        """
        batch_size = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, num_patches_height, num_patches_width)
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        if self.dis_token is not None:
            dis_tokens = self.dis_token.expand(batch_size, -1, -1)
            x = torch.cat((dis_tokens, x), dim=1)
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (batch_size, num_patches + 1, embed_dim)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply encoder blocks
        masks = []
        for encoder_layer in self.encoder_layers:
            x, mask = encoder_layer(x, full)
            if mask is not None:
                masks.append(mask)

        # Classification head
        x = self.post_norm(x)
        cls_token_final = x[:, 0]  # Extract the [CLS] token
        logits = self.head(cls_token_final)

        # If using token distil, do logits
        if self.dis_token is not None:
            cls_token_final = x[:, 1]  # Extract the [CLS] token
            dis_logits = self.dis_head(cls_token_final)
            if not self.training:
                logits = (logits + dis_logits) / 2
        else:
            dis_logits = None
        if len(masks) > 0:
            mask_mean = torch.stack(masks, dim=0)
        else:
            mask_mean = None
        return logits, dis_logits, mask_mean
