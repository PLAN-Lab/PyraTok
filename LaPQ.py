import math
import random
import warnings
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def _get_maybe_sync_seed(device: torch.device, max_size: int = 10_000) -> int:
    rand_int = torch.randint(0, max_size, (), device=device)
    if _is_distributed():
        dist.all_reduce(rand_int)
    return int(rand_int.item())


def _pick_num_heads(embed_dim: int, max_heads: int = 8) -> int:
    for heads in range(min(max_heads, embed_dim), 0, -1):
        if embed_dim % heads == 0:
            return heads
    return 1


class LaPQuantizer3D(nn.Module):
    """
    Lookup-free quantizer for 5D latents: (B, C, T, H, W).
    Uses fixed {-1, +1} codebook with optional projection to a bit-dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_codes: int,
        commitment_weight: float = 0.25,
        entropy_weight: float = 0.1,
        inv_temperature: float = 100.0,
        codebook_dim: Optional[int] = None,
        straight_through_activation: Optional[nn.Module] = None,
        text_embed_dim: Optional[int] = None,
        text_condition_heads: Optional[int] = None,
        text_condition_scale: float = 1.0,
    ):
        super().__init__()

        if codebook_dim is None:
            codebook_dim = int(math.log2(num_codes))
        if 2**codebook_dim != num_codes:
            raise ValueError(
                f"LFQuantizer3D requires num_codes to be a power of two. "
                f"Got num_codes={num_codes}, inferred codebook_dim={codebook_dim}."
            )

        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        self.inv_temperature = inv_temperature
        self.text_embed_dim = text_embed_dim

        if text_condition_heads is None:
            text_condition_heads = _pick_num_heads(codebook_dim)
        self.text_condition_scale = 0.0
        if text_condition_heads < 1 or codebook_dim % text_condition_heads != 0:
            raise ValueError(
                f"text_condition_heads must divide codebook_dim. Got text_condition_heads={text_condition_heads}, "
                f"codebook_dim={codebook_dim}."
            )
        self.text_condition_heads = text_condition_heads

        if input_dim != codebook_dim:
            self.project_in = nn.TransposeConv3d(input_dim, codebook_dim, 1)
            self.project_out = nn.Conv3d(codebook_dim, input_dim, 1)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        if straight_through_activation is None:
            straight_through_activation = nn.Identity()
        self.straight_through_activation = straight_through_activation

        bitmask = 2 ** torch.arange(codebook_dim - 1, -1, -1, dtype=torch.long)
        self.register_buffer("bitmask", bitmask)

        code_ids = torch.arange(num_codes, dtype=torch.long)
        bits = ((code_ids[:, None] & bitmask[None, :]) > 0).float()
        codebook = bits * 2.0 - 1.0
        self.register_buffer("codebook", codebook)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=codebook_dim,
            num_heads=self.text_condition_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(codebook_dim)
        self.text_context_norm = nn.LayerNorm(codebook_dim)
        self.text_proj = nn.Linear(text_embed_dim, codebook_dim) if text_embed_dim is not None else None
        self._warned_non_finite = False

    def _project_text_context(self, text_embeds: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if text_embeds.ndim != 3:
            raise ValueError(f"text_embeds must be (B, L, D), got shape={tuple(text_embeds.shape)}.")
        if text_embeds.shape[0] != z.shape[0]:
            raise ValueError(f"Batch mismatch between z ({z.shape[0]}) and text_embeds ({text_embeds.shape[0]}).")

        if self.text_proj is None:
            self.text_embed_dim = int(text_embeds.shape[-1])
            self.text_proj = nn.Linear(self.text_embed_dim, self.codebook_dim)
            self.text_proj = self.text_proj.to(device=z.device, dtype=z.dtype)
        elif text_embeds.shape[-1] != self.text_proj.in_features:
            raise ValueError(
                f"text_embeds last dim ({text_embeds.shape[-1]}) does not match expected text_embed_dim "
                f"({self.text_proj.in_features})."
            )

        proj_weight = self.text_proj.weight
        text_in = text_embeds.to(device=proj_weight.device, dtype=proj_weight.dtype)
        text_context = self.text_proj(text_in)
        text_context = self.text_context_norm(text_context)
        return text_context

    def _apply_text_condition(
        self,
        z: torch.Tensor,
        text_embeds: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if text_embeds is None or self.text_condition_scale == 0.0:
            return z

        bsz, channels, frames, height, width = z.shape
        query = z.permute(0, 2, 3, 4, 1).reshape(bsz, frames * height * width, channels)
        query_norm = self.query_norm(query)

        text_context = self._project_text_context(text_embeds=text_embeds, z=z)

        attn_dtype = self.cross_attn.in_proj_weight.dtype
        attn_device = self.cross_attn.in_proj_weight.device
        query_attn = query_norm.to(device=attn_device, dtype=attn_dtype)
        context_attn = text_context.to(device=attn_device, dtype=attn_dtype)

        key_padding_mask = None

        attn_out = self.cross_attn(
            query=query_attn,
            key=context_attn,
            value=context_attn,
            need_weights=False,
        )[0]
        attn_out = attn_out.to(device=query.device, dtype=query.dtype)
        conditioned = query + self.text_condition_scale * attn_out
        
        return conditioned.view(bsz, frames, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()

    def _entropy_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        # Low-memory entropy on independent bit probabilities.
        # This avoids constructing an enormous (num_tokens x num_codes) matrix.
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, self.codebook_dim).float()
        logits_bits = (2.0 * self.inv_temperature * z_flat).clamp(-30.0, 30.0)

        probs_pos = torch.sigmoid(logits_bits).clamp(1e-6, 1.0 - 1e-6)
        probs_neg = 1.0 - probs_pos

        ent_components = -(probs_pos * probs_pos.log() + probs_neg * probs_neg.log()).sum(dim=-1).mean()

        mixture_pos = probs_pos.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        mixture_neg = 1.0 - mixture_pos
        ent_mixture = -(mixture_pos * mixture_pos.log() + mixture_neg * mixture_neg.log()).sum()

        entropy_loss = z.new_tensor(math.log(self.num_codes), dtype=torch.float32) + ent_components - ent_mixture
        return entropy_loss.to(dtype=z.dtype)

    def forward(
        self,
        z_e: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        z = self.project_in(z_e)
        
        z = self._apply_text_condition(
            z=z,
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        )
        z_q = torch.where(z >= 0, z.new_ones(()), -z.new_ones(()))
        
        z_act = self.straight_through_activation(z)
        z_q_st = z_q + (z_act - z_act.detach())
        

        commitment_loss = F.mse_loss(z, z_q.detach()).to(z.dtype)
        loss = z.new_tensor(self.commitment_weight) * commitment_loss

        bits = (z_q > 0).to(torch.long)
        indices = (bits * self.bitmask.view(1, -1, 1, 1, 1)).sum(dim=1)

        if self.entropy_weight > 0.0 and self.training:
            
            entropy_loss = self._entropy_regularizer(z)
            # print("entropy loss: ", entropy_loss)
            loss = loss + z.new_tensor(self.entropy_weight) * entropy_loss
        z_q_st = self.project_out(z_q_st)
    
        return z_q_st, loss, indices


class ResidualLaPQunatizer3D(nn.Module):
    """
    Residual LFQ for 5D latents: (B, C, T, H, W).
    Quantizes residuals across multiple LFQ stages.
    """

    def __init__(
        self,
        input_dim: int,
        num_codes: int,
        num_quantizers: int,
        commitment_weight: float = 0.25,
        entropy_weight: float = 0.1,
        inv_temperature: float = 100.0,
        codebook_dim: Optional[int] = None,
        quantize_dropout: bool = False,
        quantize_dropout_cutoff_index: int = 0,
        quantize_dropout_multiple_of: int = 1,
        text_embed_dim: Optional[int] = None,
        text_condition_heads: Optional[int] = None,
        text_condition_scale: float = 1.0,
    ):
        super().__init__()

        if codebook_dim is None:
            codebook_dim = int(math.log2(num_codes))
        if 2**codebook_dim != num_codes:
            raise ValueError(
                f"ResidualLaPQunatizer3D requires num_codes to be a power of two. "
                f"Got num_codes={num_codes}, inferred codebook_dim={codebook_dim}."
            )

        self.num_codes = num_codes
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight
        self.inv_temperature = inv_temperature
        self.num_quantizers = num_quantizers
        self.text_embed_dim = text_embed_dim
        self.text_condition_scale = float(text_condition_scale)

        if text_condition_heads is None:
            text_condition_heads = _pick_num_heads(codebook_dim)
        if text_condition_heads < 1 or codebook_dim % text_condition_heads != 0:
            raise ValueError(
                f"text_condition_heads must divide codebook_dim. Got text_condition_heads={text_condition_heads}, "
                f"codebook_dim={codebook_dim}."
            )
        self.text_condition_heads = text_condition_heads

        if input_dim != codebook_dim: # Upsample
            self.project_in = nn.TransposeConv3d(input_dim, codebook_dim, 1)
            self.project_out = nn.Conv3d(codebook_dim, input_dim, 1)
        else:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()

        self.layers = nn.ModuleList(
            [
                LaPQuantizer3D(
                    input_dim=codebook_dim,
                    num_codes=num_codes,
                    commitment_weight=commitment_weight,
                    entropy_weight=entropy_weight,
                    inv_temperature=inv_temperature,
                    codebook_dim=codebook_dim,
                    text_embed_dim=text_embed_dim,
                    text_condition_heads=text_condition_heads,
                    text_condition_scale=text_condition_scale,
                )
                for _ in range(num_quantizers)
            ]
        )

        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        self.quantize_dropout_cutoff_index = max(0, int(quantize_dropout_cutoff_index))
        self.quantize_dropout_multiple_of = max(1, int(quantize_dropout_multiple_of))

    def forward(
        self,
        z_e: torch.Tensor,
        text_embeds: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        z = self.project_in(z_e)
        # print(z.min(), z.max(), z.mean())
        residual = z
        quantized_sum = z.new_zeros(z.shape)

        all_losses = []
        all_indices = []
        all_perplexities = []

        should_quantize_dropout = self.training and self.quantize_dropout and torch.is_grad_enabled()

        if should_quantize_dropout:
            rand_seed = _get_maybe_sync_seed(z.device)
            rng = random.Random(rand_seed)
            drop_index = rng.randrange(self.quantize_dropout_cutoff_index, self.num_quantizers)
            if self.quantize_dropout_multiple_of != 1:
                drop_index = (
                    math.ceil((drop_index + 1) / self.quantize_dropout_multiple_of) * self.quantize_dropout_multiple_of
                ) - 1
            null_indices = torch.full(z.shape[0:1] + z.shape[2:], -1, device=z.device, dtype=torch.long)
            null_loss = z.new_tensor(0.0)
            null_perplexity = z.new_tensor(0.0)
        else:
            drop_index = None

        for idx, layer in enumerate(self.layers):
            if should_quantize_dropout and idx > drop_index:
                all_indices.append(null_indices)
                all_losses.append(null_loss)
                continue

            z_q_st, loss, indices = layer(
                residual,
                text_embeds=text_embeds,
                text_attention_mask=text_attention_mask,
            )
            scale = z.new_tensor(2.0 ** (-idx))

            z_q_st = z_q_st * scale

            residual = residual - z_q_st.detach()
            quantized_sum = quantized_sum + z_q_st

            all_losses.append(loss)
            # all_perplexities.append(perplexity)
            all_indices.append(indices)

        quantized_sum = self.project_out(quantized_sum)

        loss = torch.stack(all_losses, dim=0).sum()
        # perplexity = torch.stack(all_perplexities, dim=0).mean()
        indices = torch.stack(all_indices, dim=-1)

        return quantized_sum, loss, indices


# Backwards-compatible aliases
ResidualLaPQuantizer3D = ResidualLaPQunatizer3D
ResidualLFQuantizer3D = ResidualLaPQunatizer3D
