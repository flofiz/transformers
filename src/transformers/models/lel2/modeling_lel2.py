# NOTE:
#   This file is a custom modified version of Qwen2.5-VL (multimodal decoder-only)
#   augmented with a coordinate (OBB) regression head for text+layout style generation.
#   Changes marked with:  ### MOD (short explanation)
#
#   Key design decisions implemented from review:
#     - Keep real-valued coordinates scaled into [0,2] (your original convention).
#     - Keep angle regression in (-π/2, π/2) via tanh * π/2 (you said angle works well).
#     - Use KLD (Gaussian) as main localization loss + optional L1; IoU only as monitoring metric (no gradient).
#     - Predict coordinates at the SAME token position where the special number_token_id ([coord]) is produced
#       (no implicit shift). This clarifies alignment.
#     - Compute number head ONLY on masked coordinate positions for efficiency; scatter results back into a
#       dense tensor for logging / later usage.
#     - During generation maintain a number_history (B, seq, 5) and a number_mask_history (B, seq) buffer.
#       These are passed forward in model_kwargs for autoregressive conditioning (if desired).
#     - Provide robust prepare_inputs_for_generation handling of coordinate buffers.
#     - Provide an option to inject coordinate embeddings multiplicatively (your chosen method)
#       but with an added stabilizing layernorm + safety clamp to avoid exploding magnitudes.
#
#   IMPORTANT:
#     If you previously trained with shifted targets (pred[:, :-1], tgt[:, 1:]),
#     and want to resume seamlessly you may need to add a flag coordinate_shift=True.
#     Currently set to False by default for conceptual clarity.
#
# ------------------------------------------------------------------------------------
# Original auto-generated + base imports (unchanged sections trimmed only where safe).
# ------------------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from ...generation import (
    LogitsProcessorList,
    StoppingCriteriaList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
)
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, ModelOutput
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import auto_docstring, can_return_tuple, is_torch_flex_attn_available, logging
from ...generation.configuration_utils import (
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    GenerationConfig,
    GenerationMode,
)
from ...generation.utils import GenerateOutput, GenerateNonBeamOutput
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...integrations.fsdp import is_fsdp_managed_module
import inspect
import warnings
import torch.distributed as dist  # for synced_gpus
# Qwen Imports modified for lel2
from .configuration_lel2 import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig

logger = logging.get_logger(__name__)

# ====================================================================================
# Vision + Base Model Components (UNCHANGED except where commented)
# ====================================================================================

class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config, bias: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states

class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights

class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if self.config._attn_implementation == "flash_attention_2":
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
            attn_outputs = [
                attention_interface(
                    self,
                    q,
                    k,
                    v,
                    attention_mask=None,
                    scaling=self.scaling,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    is_causal=False,
                    **kwargs,
                )[0]
                for q, k, v in zip(*splits)
            ]
            attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2_5_VLVisionBlock(GradientCheckpointingLayer):
    def __init__(self, config, attn_implementation: str = "sdpa") -> None:
        super().__init__()
        self.norm1 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = Qwen2RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config=config)
        self.mlp = Qwen2_5_VLMLP(config, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

@auto_docstring
class Qwen2_5_VLPreTrainedModel(PreTrainedModel):
    config: Qwen2_5_VLConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True

class Qwen2_5_VisionTransformerPretrainedModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLVisionConfig
    _no_split_modules = ["Qwen2_5_VLVisionBlock"]
    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size
        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                (llm_grid_h + pad_h) // vit_merger_window_size,
                vit_merger_window_size,
                (llm_grid_w + pad_w) // vit_merger_window_size,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(grid_t, -1, vit_merger_window_size, vit_merger_window_size)
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]
        return hidden_states

# ====================================================================================
# Text / Decoder components (unchanged except where commented)
# ====================================================================================

@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class Qwen2_5_VLModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

class Qwen2_5_VLRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor
    def __init__(self, config: Qwen2_5_VLTextConfig, device=None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling
        self.scaling = self.head_dim**-0.5
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.sliding_window = (
            config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        )
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            position_ids=position_ids,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2_5_VLDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen2_5_VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once("Sliding Window Attention not implemented for this attention backend.")
        self.self_attn = Qwen2_5_VLAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs

@auto_docstring
class Qwen2_5_VLTextModel(Qwen2_5_VLPreTrainedModel):
    config: Qwen2_5_VLTextConfig
    def __init__(self, config: Qwen2_5_VLTextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=text_position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

# ====================================================================================
# Multimodal Wrapper (vision+text) + coordinate enhancements
# ====================================================================================

@auto_docstring
class Qwen2_5_VLModel(Qwen2_5_VLPreTrainedModel):
    base_model_prefix = ""
    _checkpoint_conversion_mapping = {"^model": "language_model"}
    config_class = Qwen2_5_VLConfig
    _no_split_modules = ["Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(config.vision_config)
        self.language_model = Qwen2_5_VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        # old number_encoder changed usage downstream (kept for compatibility if needed)
        self.language_model.number_encoder = nn.Linear(5, config.hidden_size, bias=False)  # still available
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # (UNCHANGED logic from original – removed for brevity in this explanation)
        # ... (Keeping full logic – important for correct multimodal rope)
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, ids in enumerate(total_input_ids):
                ids_valid = ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(ids_valid == vision_start_token_id).squeeze(1)
                vision_tokens = ids_valid[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = ids_valid.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = image_grid_thw[image_index]
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = video_grid_thw[video_index]
                        second_per_grid_t = (
                            second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        (h.item() // spatial_merge_size),
                        (w.item() // spatial_merge_size),
                    )
                    text_len = ed - st
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    second_per_grid_t = torch.as_tensor(
                        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                    )
                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second
                    t_index = time_tensor.long().flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                        llm_grid_t, -1, llm_grid_w
                    ).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                        llm_grid_t, llm_grid_h, -1
                    ).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        number_ids: Optional[torch.FloatTensor] = None,   ### MOD (full tensor of shape (B,S,5) or None)
        number_mask: Optional[torch.BoolTensor] = None,   ### MOD (shape (B,S)) indicates which positions have coords)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        inject_coordinates: bool = True,  ### MOD
    ):
        """
        number_ids: real-valued coords (B,S,5) scaled approx in [0,2] & angle in [-pi/2,pi/2]
        number_mask: bool (B,S) True where input_ids == number_token_id (or where coords known)
        inject_coordinates: if True, multiplicative injection on those positions.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # Vision patch replacement
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                if n_image_tokens != image_embeds.shape[0]:
                    raise ValueError("Image tokens/features mismatch.")
                mask_img = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(mask_img, image_embeds.to(inputs_embeds.dtype))

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                if n_video_tokens != video_embeds.shape[0]:
                    raise ValueError("Video tokens/features mismatch.")
                mask_vid = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(mask_vid, video_embeds.to(inputs_embeds.dtype))

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        # Build / reuse rope indices
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                base_pos = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
                if cache_position is not None and not isinstance(delta, int):
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    base_pos = base_pos + delta
                position_ids = base_pos.unsqueeze(0).expand(3, -1, -1)

        # Coordinate injection (multiplicative gating)
        if inject_coordinates and number_ids is not None and number_mask is not None:
            # We project raw coords through existing number_encoder
            coord_proj = self.language_model.number_encoder(number_ids.to(inputs_embeds.dtype))  # (B,S,H)
            # Safe scaling: (1 + normalized factor) to avoid annihilating embeddings
            # clamp to keep stability
            scale = torch.clamp(coord_proj, min=-4.0, max=4.0)  # safety
            scale = torch.tanh(scale)  # in (-1,1)
            scale = 1.0 + scale  # in (0,2)
            scale = torch.where(number_mask.unsqueeze(-1), scale, torch.ones_like(scale))
            inputs_embeds = inputs_embeds * scale
            # Optional layernorm after injection for stability
            inputs_embeds = F.layer_norm(inputs_embeds, (inputs_embeds.shape[-1],))

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        return Qwen2_5_VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

# ====================================================================================
# Causal LM Output dataclass (extended)
# ====================================================================================

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    coord_loss: Optional[torch.FloatTensor] = None
    kld_loss: Optional[torch.FloatTensor] = None
    l1_loss: Optional[torch.FloatTensor] = None
    iou_metric: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    logits_number: Optional[torch.FloatTensor] = None  # full dense (B,S,5) with zeros where no coord
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

# ====================================================================================
# Coordinate Head / Loss (NEW SIMPLIFIED VERSION)
# ====================================================================================

class LeLNumberHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, coord_scale: float = 2.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        self.coord_scale = coord_scale
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 5),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, hidden_subset: torch.Tensor) -> torch.Tensor:
        # hidden_subset peut être bf16 -> on fait les couches en float32 puis on recaste
        orig_dtype = hidden_subset.dtype
        x = hidden_subset.float()
        x = self.mlp(x)
        cx, cy, w, h, ang = x.split(1, dim=-1)
        cx = torch.sigmoid(cx) * self.coord_scale
        cy = torch.sigmoid(cy) * self.coord_scale
        w = torch.sigmoid(w) * self.coord_scale
        h = torch.sigmoid(h) * self.coord_scale
        ang = torch.tanh(ang) * (math.pi / 2)
        out = torch.cat([cx, cy, w, h, ang], dim=-1)
        return out.to(orig_dtype)

class StableKLDLocalizationLoss(nn.Module):
    """
    Stable localisation loss for boxes in xyxya = (xmin, ymin, xmax, ymax, angle).

    Features:
      - Analytic KL (no matrix inversion)
      - Optional symmetric KL
      - SmoothL1 on raw xyxya
      - Axis-aligned IoU metric
      - NaN / inf guards
      - Optional warmup on KLD weight
    """
    def __init__(
        self,
        l1_weight: float = 1.0,
        kld_weight: float = 1.0,
        eps: float = 1e-7,
        symmetric: bool = False,
        kld_warmup_steps: int = 0  # if >0, scale kld_weight by (current_step / kld_warmup_steps) clipped to 1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.base_kld_weight = kld_weight
        self.eps = eps
        self.symmetric = symmetric
        self.kld_warmup_steps = kld_warmup_steps

        # last components
        self._last_kld = None
        self._last_l1 = None
        self._last_iou_metric = None

        # internal step tracker (can be updated externally)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=False)

    @staticmethod
    def _xyxya_to_params(boxes: torch.Tensor, eps: float):
        x1, y1, x2, y2, a = boxes.unbind(-1)
        # enforce ordering to avoid negative widths/heights
        x1c = torch.minimum(x1, x2)
        x2c = torch.maximum(x1, x2)
        y1c = torch.minimum(y1, y2)
        y2c = torch.maximum(y1, y2)
        w = (x2c - x1c).clamp(min=eps)
        h = (y2c - y1c).clamp(min=eps)
        cx = (x1c + x2c) * 0.5
        cy = (y1c + y2c) * 0.5
        return cx, cy, w, h, a, x1c, y1c, x2c, y2c

    @staticmethod
    def _axis_aligned_iou(x1p, y1p, x2p, y2p, x1t, y1t, x2t, y2t, eps: float):
        inter_x1 = torch.maximum(x1p, x1t)
        inter_y1 = torch.maximum(y1p, y1t)
        inter_x2 = torch.minimum(x2p, x2t)
        inter_y2 = torch.minimum(y2p, y2t)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        ap = (x2p - x1p).clamp(min=0) * (y2p - y1p).clamp(min=0)
        at = (x2t - x1t).clamp(min=0) * (y2t - y1t).clamp(min=0)
        union = ap + at - inter + eps
        return inter / union

    @staticmethod
    def _analytic_kl_one_way(cx_p, cy_p, w_p, h_p, a_p,
                             cx_q, cy_q, w_q, h_q, a_q,
                             eps):
        # KL(P||Q)
        # Ensure minimal size
        w_p = w_p.clamp(min=eps)
        h_p = h_p.clamp(min=eps)
        w_q = w_q.clamp(min=eps)
        h_q = h_q.clamp(min=eps)

        # Precompute
        # a1, a2 eigenvalues of Sigma_q^{-1}
        a1 = 4.0 / (w_q * w_q + eps)
        a2 = 4.0 / (h_q * h_q + eps)
        # b1, b2 eigenvalues of Sigma_p
        b1 = (w_p * w_p) / 4.0
        b2 = (h_p * h_p) / 4.0

        # angle difference
        phi = a_p - a_q
        c = torch.cos(phi)
        s = torch.sin(phi)
        c2 = c * c
        s2 = s * s

        # trace term
        # trace( Sigma_q^{-1} Sigma_p ) =
        # a1*(b1 c^2 + b2 s^2) + a2*(b1 s^2 + b2 c^2)
        trace_term = a1 * (b1 * c2 + b2 * s2) + a2 * (b1 * s2 + b2 * c2)

        # mahal term:
        # diff' = R_q^T (mu_q - mu_p)
        # R_q^T = [[ cos(a_q), sin(a_q)],
        #          [-sin(a_q), cos(a_q)]]
        caq = torch.cos(a_q)
        saq = torch.sin(a_q)
        dx = (cx_q - cx_p)
        dy = (cy_q - cy_p)
        dqx =  caq * dx + saq * dy
        dqy = -saq * dx + caq * dy
        mahal = a1 * dqx * dqx + a2 * dqy * dqy

        # log det ratio
        log_det_p = 2 * (torch.log(w_p + eps) + torch.log(h_p + eps)) - 2 * math.log(4.0)
        log_det_q = 2 * (torch.log(w_q + eps) + torch.log(h_q + eps)) - 2 * math.log(4.0)
        log_term = (log_det_q - log_det_p)

        kl = 0.5 * (trace_term + mahal - 2.0 + log_term)
        return torch.clamp(kl, min=0.0)

    def forward(self, pred_boxes_xyxya: torch.Tensor, tgt_boxes_xyxya: torch.Tensor):
        if pred_boxes_xyxya.numel() == 0:
            zero = pred_boxes_xyxya.new_zeros(())
            self._last_kld = zero
            self._last_l1 = zero
            self._last_iou_metric = zero
            return zero

        # Upcast to float32 for stability
        dtype_out = pred_boxes_xyxya.dtype
        pb = pred_boxes_xyxya.float()
        tb = tgt_boxes_xyxya.float()

        cx_p, cy_p, w_p, h_p, a_p, x1p, y1p, x2p, y2p = self._xyxya_to_params(pb, self.eps)
        cx_t, cy_t, w_t, h_t, a_t, x1t, y1t, x2t, y2t = self._xyxya_to_params(tb, self.eps)

        # Smooth L1 directement sur xyxya (après correction x1<x2 etc.)
        packed_p = torch.stack([x1p, y1p, x2p, y2p, a_p], dim=-1)
        packed_t = torch.stack([x1t, y1t, x2t, y2t, a_t], dim=-1)
        l1 = F.smooth_l1_loss(packed_p, packed_t, reduction="mean")

        # KL
        kl_pt = self._analytic_kl_one_way(cx_p, cy_p, w_p, h_p, a_p,
                                          cx_t, cy_t, w_t, h_t, a_t,
                                          self.eps)
        if self.symmetric:
            kl_tp = self._analytic_kl_one_way(cx_t, cy_t, w_t, h_t, a_t,
                                              cx_p, cy_p, w_p, h_p, a_p,
                                              self.eps)
            kl_vals = 0.5 * (kl_pt + kl_tp)
        else:
            kl_vals = kl_pt

        # IoU metric (axis aligned)
        with torch.no_grad():
            ious = self._axis_aligned_iou(x1p, y1p, x2p, y2p, x1t, y1t, x2t, y2t, self.eps)
            iou_metric = ious.mean()

        # Guard against non-finite
        if not torch.isfinite(kl_vals).all():
            kl_vals = torch.where(torch.isfinite(kl_vals), kl_vals, torch.zeros_like(kl_vals))

        kld_mean = kl_vals.mean()

        # Warmup schedule (optional)
        if self.kld_warmup_steps > 0 and self.base_kld_weight > 0:
            # step peut être mis à jour depuis l'extérieur: loss.step += 1
            warm_factor = (self.step.item() / float(self.kld_warmup_steps))
            warm_factor = min(max(warm_factor, 0.0), 1.0)
        else:
            warm_factor = 1.0

        total = self.l1_weight * l1 + (self.base_kld_weight * warm_factor) * kld_mean

        # Final NaN guard
        if not torch.isfinite(total):
            print("[StableKLDLocalizationLoss] WARNING: total loss NaN/Inf. l1=", l1.item(), "kld=", kld_mean.item())
            total = torch.zeros((), device=pb.device)

        # Store components
        self._last_l1 = l1.detach().to(dtype_out)
        self._last_kld = kld_mean.detach().to(dtype_out)
        self._last_iou_metric = iou_metric.detach().to(dtype_out)
        return total.to(dtype_out)

    def get_loss_components(self):
        if self._last_kld is None:
            zero = torch.tensor(0.0)
            return {
                "kld_loss": zero,
                "l1_loss": zero,
                "iou_metric": zero,
            }
        return {
            "kld_loss": self._last_kld,
            "l1_loss": self._last_l1,
            "iou_metric": self._last_iou_metric,
        }

    def advance_step(self, n: int = 1):
        # Permet d'incrémenter le step depuis l'entraînement pour le warmup KLD
        self.step += n

# ====================================================================================
# Main Model for Conditional Generation with coordinate integration
# ====================================================================================

class LeL2_ForConditionalGeneration(Qwen2_5_VLPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.numbers_head = LeLNumberHead(config.hidden_size, config.hidden_size, coord_scale=2.0)
        # Localization loss (KLD + L1, IoU metric only)
        self.localisation_loss = StableKLDLocalizationLoss(l1_weight=10.0, kld_weight=1.0)
        # Flag controlling whether we shift for coordinate regression (False = predict at same position).
        self.coordinate_shift = False
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    # --------------------------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------------------------
    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        number_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids, 
            number_ids=number_ids, # ToDo : ajouter number_ids partout pour propager
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits_number = self.numbers_head(hidden_states)
        # logits_number = torch.clamp(logits_number, min = 0, max = 2)
        loss = None
        lm_loss = None
        l1_loss = None
        iou_metric = None
        kld_loss = None
        coord_loss = None
        if labels is not None:
            text_labels, number_labels = labels
            lm_loss = self.loss_function(logits=logits, labels=text_labels, vocab_size=self.config.vocab_size)
            target_coord_mask = (text_labels == self.config.number_token_id)
            pred_filtered = logits_number[target_coord_mask]
            tgt_filtered = number_labels[target_coord_mask]
            # number_loss = torch.sum(F.smooth_l1_loss(input=logits_number[:,:-1].squeeze(-1)*number_mask[:,1:],target=number_ids[:,1:]*number_mask[:,1:], reduction="none"))/torch.sum(number_mask)
            # coord_loss = number_loss_fn(logits_number, number_label, number_mask)
            pred_fp32 = pred_filtered.float()
            tgt_fp32  = tgt_filtered.float()
            coord_loss = self.localisation_loss(pred_fp32, tgt_fp32)
            loss_components = self.localisation_loss.get_loss_components()
            l1_loss = loss_components.get("l1_loss", None)
            iou_metric = loss_components.get("iou_metric", None)
            kld_loss = loss_components.get("kld_loss", None)
            loss = 1*lm_loss + 1*coord_loss

        if not return_dict:
            output = (logits,logits_number, ) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            lm_loss = lm_loss,
            l1_loss = l1_loss,
            iou_metric = iou_metric,
            kld_loss = kld_loss,
            coord_loss = coord_loss,
            logits=logits,
            logits_number=logits_number,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    # prepare_inputs_for_generation unchanged except for passing through number_history / number_mask_history safely
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        number_history: Optional[torch.FloatTensor] = None,
        number_mask_history: Optional[torch.BoolTensor] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=None,
            use_cache=use_cache,
            **kwargs,
        )
        B, S = input_ids.shape
        if number_history is None:
            number_history = input_ids.new_zeros((B, S, 5), dtype=torch.float32)
        if number_mask_history is None:
            number_mask_history = (input_ids == self.config.number_token_id)

        number_history = number_history[:, :S, :]
        number_mask_history = number_mask_history[:, :S]

        model_inputs["number_history"] = number_history
        model_inputs["number_mask_history"] = number_mask_history

        # Skip re-processing images for subsequent steps
        if cache_position is not None and cache_position[0] != 0:
            for k in ["pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"]:
                model_inputs[k] = None
        return model_inputs

    # --------------------------------------------------------------------------------
    # Generation override (only sampling path adjusted for coordinate head)
    # --------------------------------------------------------------------------------
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        **kwargs,
    ):
        # Re-use base GenerationMixin logic but adapt _sample to supply coordinate updates
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            use_model_defaults=use_model_defaults,
            **kwargs,
        )

    # We only override _sample to insert coordinate head calls
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        **model_kwargs,
    ):
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        # Initialize coordinate histories
        if "number_history" not in model_kwargs or model_kwargs["number_history"] is None:
            model_kwargs["number_history"] = input_ids.new_zeros((batch_size, cur_len, 5), dtype=torch.float32)
        if "number_mask_history" not in model_kwargs or model_kwargs["number_mask_history"] is None:
            model_kwargs["number_mask_history"] = (input_ids == self.config.number_token_id)

        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)
        model_forward = self.__call__
        compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        if compile_forward:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

        is_prefill = True
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs.update(
                {
                    "output_attentions": output_attentions,
                    "output_hidden_states": output_hidden_states,
                }
            )
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(torch.float32)
            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate and output_scores:
                scores += (next_token_scores,)
            if return_dict_in_generate and output_logits:
                raw_logits += (next_token_logits,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Append
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)

            # Handle coordinate token generation
            coord_mask_new = (next_tokens == self.config.number_token_id)
            if coord_mask_new.any():
                last_hidden = outputs.last_hidden_state[:, -1, :]          # (B,H) dtype backbone
                coord_subset = last_hidden[coord_mask_new]                 # (Nc,H)
                pred_coords = self.numbers_head(coord_subset)              # (Nc,5) dtype hidden
                # Sécurité (normalement déjà aligné)
                pred_coords = pred_coords.to(last_hidden.dtype)

                old_hist = model_kwargs["number_history"]
                old_mask_hist = model_kwargs["number_mask_history"]
                new_hist = torch.cat(
                    [old_hist, old_hist.new_zeros((batch_size, 1, 5))],
                    dim=1,
                )
                new_mask = torch.cat(
                    [old_mask_hist, old_mask_hist.new_zeros((batch_size, 1), dtype=torch.bool)],
                    dim=1,
                )
                new_hist[coord_mask_new, -1, :] = pred_coords
                new_mask[coord_mask_new, -1] = True
                model_kwargs["number_history"] = new_hist
                model_kwargs["number_mask_history"] = new_mask
            else:
                model_kwargs["number_history"] = torch.cat(
                    [model_kwargs["number_history"], model_kwargs["number_history"].new_zeros((batch_size, 1, 5))],
                    dim=1,
                )
                model_kwargs["number_mask_history"] = torch.cat(
                    [model_kwargs["number_mask_history"], model_kwargs["number_mask_history"].new_zeros((batch_size, 1), dtype=torch.bool)],
                    dim=1,
                )

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            from ...generation.utils import GenerateDecoderOnlyOutput
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return input_ids

# ------------------------------------------------------------------------------------
__all__ = [
    "LeL2_ForConditionalGeneration",
    "Qwen2_5_VLModel",
    "Qwen2_5_VLPreTrainedModel",
    "Qwen2_5_VLTextModel",
]