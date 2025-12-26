# video_vae_modular_final.py

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import math
import os
import sys

import importlib.abc
import importlib.util
import inspect
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass
from types import SimpleNamespace

_CURRENT_DIR = os.path.dirname(__file__)
_DIFFUSERS_SRC = os.path.join(os.path.dirname(__file__), "src")
if os.path.isdir(_DIFFUSERS_SRC) and _DIFFUSERS_SRC not in sys.path:  # pragma: no cover - local dev convenience
    sys.path.insert(0, _DIFFUSERS_SRC)

if "huggingface_hub" not in sys.modules:  # pragma: no cover - provide lightweight stub for local usage
    class _HFStubLoader(importlib.abc.Loader):
        def create_module(self, spec):  # pragma: no cover - loader protocol
            return None

        def exec_module(self, module):  # pragma: no cover - loader protocol
            return None

    hf_stub = types.ModuleType("huggingface_hub")
    hf_stub.__path__ = []  # mark as package
    hf_stub.__spec__ = importlib.util.spec_from_loader("huggingface_hub", _HFStubLoader())

    constants_module = types.ModuleType("huggingface_hub.constants")
    constants_module.HF_HOME = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    constants_module.__spec__ = importlib.util.spec_from_loader("huggingface_hub.constants", _HFStubLoader())

    utils_module = types.ModuleType("huggingface_hub.utils")
    utils_module.is_jinja_available = lambda: False
    class _RevisionNotFoundError(RuntimeError):
        pass

    utils_module.RevisionNotFoundError = _RevisionNotFoundError
    utils_module.validate_hf_hub_args = lambda *_, **__: None
    utils_module.__spec__ = importlib.util.spec_from_loader("huggingface_hub.utils", _HFStubLoader())

    hf_stub.constants = constants_module
    hf_stub.utils = utils_module
    hf_stub.hf_hub_download = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("hf_hub_download stubbed"))
    hf_stub.model_info = lambda *_, **__: (_ for _ in ()).throw(RuntimeError("model_info stubbed"))

    sys.modules["huggingface_hub"] = hf_stub
    sys.modules["huggingface_hub.constants"] = constants_module
    sys.modules["huggingface_hub.utils"] = utils_module

if "diffusers" not in sys.modules:  # pragma: no cover - lightweight diffusers structure for local usage
    diffusers_root = types.ModuleType("diffusers")
    diffusers_root.__path__ = []
    diffusers_root.__spec__ = importlib.util.spec_from_loader("diffusers", _HFStubLoader())
    sys.modules["diffusers"] = diffusers_root

    # configuration_utils
    config_module = types.ModuleType("diffusers.configuration_utils")

    class ConfigMixin:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple config storage
            super().__init__()  # type: ignore[misc]
            if not hasattr(self, "config"):
                self.config = SimpleNamespace()


    def register_to_config(init):
        signature = inspect.signature(init)

        def wrapper(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            config_dict = {
                k: v
                for k, v in bound.arguments.items()
                if k != "self" and not k.startswith("_")
            }
            self.config = SimpleNamespace(**config_dict)
            init(self, *args, **kwargs)

        return wrapper


    config_module.ConfigMixin = ConfigMixin
    config_module.register_to_config = register_to_config
    sys.modules["diffusers.configuration_utils"] = config_module

    # loaders
    loaders_module = types.ModuleType("diffusers.loaders")

    class FromOriginalModelMixin:
        pass


    loaders_module.FromOriginalModelMixin = FromOriginalModelMixin
    sys.modules["diffusers.loaders"] = loaders_module

    # utils.logging
    utils_module_root = types.ModuleType("diffusers.utils")
    logging_module = types.ModuleType("diffusers.utils.logging")

    import logging as _py_logging

    def get_logger(name: str):  # pragma: no cover - thin wrapper
        return _py_logging.getLogger(name)


    logging_module.get_logger = get_logger
    utils_module_root.logging = logging_module

    # utils.accelerate_utils
    accelerate_utils_module = types.ModuleType("diffusers.utils.accelerate_utils")

    def apply_forward_hook(fn):  # pragma: no cover - identity decorator
        return fn


    accelerate_utils_module.apply_forward_hook = apply_forward_hook
    utils_module_root.accelerate_utils = accelerate_utils_module
    sys.modules["diffusers.utils"] = utils_module_root
    sys.modules["diffusers.utils.logging"] = logging_module
    sys.modules["diffusers.utils.accelerate_utils"] = accelerate_utils_module

    # models namespace packages
    models_module = types.ModuleType("diffusers.models")
    models_module.__path__ = []
    sys.modules["diffusers.models"] = models_module

    # models.activations
    activations_module = types.ModuleType("diffusers.models.activations")

    def get_activation(name: str):  # pragma: no cover - minimal activation support
        name = name.lower()
        if name == "gelu":
            return nn.GELU()
        if name in {"silu", "swish"}:
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")


    activations_module.get_activation = get_activation
    sys.modules["diffusers.models.activations"] = activations_module

    # models.modeling_outputs
    modeling_outputs_module = types.ModuleType("diffusers.models.modeling_outputs")

    class AutoencoderKLOutput(SimpleNamespace):
        def __init__(self, latent_dist=None, sample=None):
            super().__init__(latent_dist=latent_dist, sample=sample)


    modeling_outputs_module.AutoencoderKLOutput = AutoencoderKLOutput
    sys.modules["diffusers.models.modeling_outputs"] = modeling_outputs_module

    # models.modeling_utils
    modeling_utils_module = types.ModuleType("diffusers.models.modeling_utils")

    class ModelMixin(nn.Module):
        pass


    modeling_utils_module.ModelMixin = ModelMixin
    sys.modules["diffusers.models.modeling_utils"] = modeling_utils_module

    # models.autoencoders namespace and deps
    autoencoders_module = types.ModuleType("diffusers.models.autoencoders")
    autoencoders_module.__path__ = []
    sys.modules["diffusers.models.autoencoders"] = autoencoders_module

    vae_module = types.ModuleType("diffusers.models.autoencoders.vae")

    class DecoderOutput(SimpleNamespace):
        def __init__(self, sample: torch.Tensor):
            super().__init__(sample=sample)


    class DiagonalGaussianDistribution:
        def __init__(self, parameters: torch.Tensor):
            mean, logvar = torch.chunk(parameters, 2, dim=1)
            self.mean = mean
            self.logvar = torch.clamp(logvar, -30.0, 20.0)

        def sample(self) -> torch.Tensor:
            std = torch.exp(0.5 * self.logvar)
            noise = torch.randn_like(std)
            return self.mean + std * noise

        def mode(self) -> torch.Tensor:
            return self.mean

        def kl(self) -> torch.Tensor:
            return 0.5 * (torch.exp(self.logvar) + self.mean.pow(2) - 1.0 - self.logvar)


    vae_module.DecoderOutput = DecoderOutput
    vae_module.DiagonalGaussianDistribution = DiagonalGaussianDistribution
    sys.modules["diffusers.models.autoencoders.vae"] = vae_module
try:
    from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM  # type: ignore
    _TRANSFORMERS_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AttributeError):  # pragma: no cover - optional dependency for offline tests
    AutoModel = AutoProcessor = AutoModelForCausalLM = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False

_WAN_MODULE_PATH = os.path.join(
    _DIFFUSERS_SRC, "diffusers", "models", "autoencoders", "autoencoder_kl_wan.py"
)

if not os.path.isfile(_WAN_MODULE_PATH):  # pragma: no cover - sanity check
    raise FileNotFoundError(f"Could not locate WAN autoencoder source at {_WAN_MODULE_PATH}")

_WAN_SPEC = importlib.util.spec_from_file_location(
    "diffusers.models.autoencoders.autoencoder_kl_wan", _WAN_MODULE_PATH
)
_WAN_MODULE = importlib.util.module_from_spec(_WAN_SPEC)
assert _WAN_SPEC is not None and _WAN_SPEC.loader is not None
_WAN_SPEC.loader.exec_module(_WAN_MODULE)
AutoencoderKLWan = _WAN_MODULE.AutoencoderKLWan

@dataclass
class VideoVAEConfig:
    in_channels: int = 3
    quant_emb_dim: int = 16
    alignment_dim: int = 256
    quant_align_loss_weight: float = 0.1
    likelihood_loss_weight: float = 0.2
    dino_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    kl_loss_weight: float = 1.0
    freeze_autoencoder: bool = True
    wan_pretrained_path: Optional[str] = None
    wan_subfolder: str = "vae"
    wan_torch_dtype: str = "float32"

# ==============================================================================
# 2. PERCEPTUAL & TEXT MODULES
# ==============================================================================

class DINOv2Extractor(nn.Module):
    """
    A frozen DINOv2 model to extract perceptual features from video frames.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required to load the DINOv2 extractor.")
        self.device = device
        model_name = "facebook/dinov2-base"
        print("Loading DINOv2 model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir="/data2/onkar/llava")
        self.model = AutoModel.from_pretrained(model_name, cache_dir="/data2/onkar/llava").to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("DINOv2 loaded and frozen successfully. 🦖")

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video_tensor.shape
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.processor(images=video_tensor, return_tensors="pt", do_rescale=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the features of the [CLS] token
        return outputs.last_hidden_state[:, 0].view(b, t, -1)

class QwenVLTextEncoder(nn.Module):
    """A frozen Qwen-VL model to extract text embeddings."""
    def __init__(self, device="cuda"):
        super().__init__()
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required to load the Qwen-VL text encoder.")
        model_id = "Qwen/Qwen2.5-VL-Instruct"
        self.device = device
        print("Loading Qwen-VL model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir="/data2/onkar/llava")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True, cache_dir="/data2/onkar/llava").eval()
        for param in self.model.parameters(): param.requires_grad = False
        print("Qwen-VL loaded and frozen successfully. 🥶")

    def forward(self, text_prompts: list[str]):
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in text_prompts]
        text_inputs = self.processor(conversations=messages, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**text_inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].to(self.device)

class TextVideoCrossAttention(nn.Module):
    """Performs cross-attention between video features (Q) and text features (K,V)."""
    def __init__(self, video_channels, text_embed_dim):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(video_channels, video_channels), nn.Linear(text_embed_dim, video_channels), nn.Linear(text_embed_dim, video_channels)
        self.out_proj = nn.Linear(video_channels, video_channels)

    def forward(self, video_feat, text_embedding):
        B, C, T, H, W = video_feat.shape
        video_seq = video_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        q, k, v = self.q_proj(video_seq), self.k_proj(text_embedding), self.v_proj(text_embedding)
        attn_weights = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(q.size(-1)), dim=-1)
        attn_output = attn_weights @ v
        return self.out_proj(attn_output).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

# ==============================================================================
# 3. CORE ARCHITECTURAL BLOCKS
# ==============================================================================

class ProjectedLFQ(nn.Module):
    """Projects features and quantizes them, returning an entropy loss."""
    def __init__(self, in_channels, quant_channels, entropy_loss_weight=0.1):
        super().__init__()
        self.project = nn.Conv3d(in_channels, quant_channels, 1)
        self.entropy_loss_weight = entropy_loss_weight

    def forward(self, x):
        x_proj = self.project(x)
        quantized_x_hard = torch.where(x_proj > 0, 1.0, -1.0)
        quantized_x = x_proj + (quantized_x_hard - x_proj).detach()
        indices = (quantized_x > 0).long()
        probs = indices.float().mean(dim=(0, 2, 3, 4))
        entropy = - (probs * torch.log(probs.clamp(min=1e-8)) + (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))
        entropy_loss = -entropy.mean() * self.entropy_loss_weight
        return quantized_x, indices, entropy_loss


class DummyCausalLM(nn.Module):
    """Lightweight causal LM used for offline testing when the real Qwen model is unavailable."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs_embeds: torch.Tensor, output_hidden_states: bool = False):
        hidden = self.proj(inputs_embeds)
        if output_hidden_states:
            return SimpleNamespace(hidden_states=[inputs_embeds, hidden])
        return SimpleNamespace(hidden_states=[hidden])


class RandomTextEncoderStub(nn.Module):
    """Generates random text embeddings while exposing a Qwen-like interface."""

    def __init__(self, hidden_size: int = 256, seq_len: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.model = DummyCausalLM(hidden_size)

    def forward(self, text_prompts: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        batch = len(text_prompts)
        return torch.randn(batch, self.seq_len, self.hidden_size, device=device)


class RandomDINOExtractorStub(nn.Module):
    """Produces random perceptual features for smoke tests."""

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = video_tensor.shape
        return torch.randn(b, t, self.feature_dim, device=video_tensor.device)

# ==============================================================================
# 4. PRIMARY VideoVAE MODEL
# ==============================================================================

class VideoVAE(nn.Module):
    """Text-conditioned Video VAE that wraps the WAN 2.2 autoencoder and custom quantization losses."""

    def __init__(
        self,
        cfg: VideoVAEConfig,
        device: str = "cuda",
        *,
        text_encoder: Optional[nn.Module] = None,
        dino_extractor: Optional[nn.Module] = None,
        autoencoder: Optional[AutoencoderKLWan] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- WAN 2.2 Autoencoder ---
        if autoencoder is not None:
            self.autoencoder = autoencoder
        elif cfg.wan_pretrained_path:
            dtype = getattr(torch, cfg.wan_torch_dtype, torch.float32)
            self.autoencoder = AutoencoderKLWan.from_pretrained(
                cfg.wan_pretrained_path,
                subfolder=cfg.wan_subfolder,
                torch_dtype=dtype,
                cache_dir="/data2/onkar/llava"
            )
        else:
            self.autoencoder = AutoencoderKLWan(in_channels=cfg.in_channels, out_channels=cfg.in_channels)

        self.autoencoder.to(device)
        if cfg.freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.latent_channels = self.autoencoder.config.z_dim

        # --- Text Encoder ---
        if text_encoder is None:
            self.text_encoder = QwenVLTextEncoder(device=device)
            text_embed_dim = self.text_encoder.model.config.hidden_size
        else:
            self.text_encoder = text_encoder.to(device)
            if hasattr(self.text_encoder, "model") and hasattr(self.text_encoder.model, "config"):
                text_embed_dim = getattr(self.text_encoder.model.config, "hidden_size", None)
            else:
                text_embed_dim = getattr(self.text_encoder, "hidden_size", None)
            if text_embed_dim is None:
                raise ValueError("Provided text_encoder must expose a `hidden_size` attribute.")

        self.text_embed_dim = text_embed_dim

        # --- Perceptual Extractor (optional) ---
        if dino_extractor is not None:
            self.dino_extractor = dino_extractor.to(device)
        elif self.training:
            self.dino_extractor = DINOv2Extractor(device=device)
        else:
            self.dino_extractor = None

        # --- Latent conditioning & quantization ---
        self.text_latent_attn = TextVideoCrossAttention(self.latent_channels, text_embed_dim)
        self.latent_quantizer = ProjectedLFQ(
            in_channels=self.latent_channels,
            quant_channels=cfg.quant_emb_dim,
            entropy_loss_weight=cfg.entropy_loss_weight,
        )

        codebook_size = 2 ** cfg.quant_emb_dim
        self.quant_embedding = nn.Embedding(codebook_size, text_embed_dim)
        self.to_quant_logits = nn.Linear(text_embed_dim, codebook_size)
        self.quant_proj = nn.Linear(cfg.quant_emb_dim, cfg.alignment_dim)
        self.text_proj_for_quant = nn.Linear(text_embed_dim, cfg.alignment_dim)

    def forward(
        self,
        x: torch.Tensor,
        text_prompts: List[str],
        *,
        sample_posterior: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encodes input video, applies text-guided latent modulation, and reconstructs the video."""

        text_embedding = self.text_encoder(text_prompts)
        if not isinstance(text_embedding, torch.Tensor):
            raise TypeError("text_encoder must return a torch.Tensor")
        text_embedding = text_embedding.to(self.device)

        latent_dist = self.autoencoder.encode(x).latent_dist
        latents = latent_dist.sample() if sample_posterior else latent_dist.mode()
        latents = latents.to(self.device)

        attended_latents = latents + self.text_latent_attn(latents, text_embedding)
        quantized_latents, quant_indices, entropy_loss = self.latent_quantizer(attended_latents)
        reconstruction = self.autoencoder.decode(attended_latents).sample
        if reconstruction.shape[2:] != x.shape[2:]:  # align to original spatial/temporal dims if padding occurred
            reconstruction = F.interpolate(
                reconstruction,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return {
            "reconstruction": reconstruction,
            "text_embedding": text_embedding,
            "latent_dist": latent_dist,
            "attended_latents": attended_latents,
            "quantized_latents": quantized_latents,
            "quant_indices": quant_indices,
            "entropy_loss": entropy_loss,
        }

    def calculate_losses(self, original_video: torch.Tensor, forward_outputs: Dict) -> Dict:
        """Compute all training losses. Should only be called when ``self.training`` is ``True``."""

        if not self.training:
            raise RuntimeError("calculate_losses() should only be called in training mode.")

        recon = forward_outputs["reconstruction"]
        text_emb = forward_outputs["text_embedding"]
        quantized_latents = forward_outputs["quantized_latents"]
        quant_indices = forward_outputs["quant_indices"]
        entropy_loss = forward_outputs["entropy_loss"]
        latent_dist = forward_outputs["latent_dist"]

        # 1. Reconstruction loss
        recon_loss = F.mse_loss(recon, original_video)

        # 2. Entropy loss from LFQ module (already weighted inside ProjectedLFQ)
        entropy_loss_value = entropy_loss

        # 3. P(Q | text) likelihood loss
        B = text_emb.size(0)
        bit_sequences = quant_indices.view(B, self.cfg.quant_emb_dim, -1).permute(0, 2, 1)
        powers_of_2 = (2 ** torch.arange(self.cfg.quant_emb_dim, device=self.device)).float()
        quant_token_ids = (bit_sequences * powers_of_2).sum(dim=-1).long()

        quant_embeds = self.quant_embedding(quant_token_ids)
        combined_embeds = torch.cat([text_emb, quant_embeds], dim=1)
        with torch.no_grad():
            qwen_outputs = self.text_encoder.model(inputs_embeds=combined_embeds, output_hidden_states=True)
        last_hidden = qwen_outputs.hidden_states[-1][:, text_emb.shape[1] - 1 : -1, :]
        pred_logits = self.to_quant_logits(last_hidden)
        likelihood_loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), quant_token_ids.reshape(-1))

        # 4. Quantized vector-text alignment loss
        q_pooled = F.adaptive_avg_pool3d(quantized_latents, 1).view(B, -1)
        text_pooled = text_emb.mean(dim=1)
        q_aligned = self.quant_proj(q_pooled)
        text_aligned = self.text_proj_for_quant(text_pooled)
        quant_align_loss = F.cosine_embedding_loss(
            q_aligned, text_aligned, torch.ones(B, device=self.device)
        )

        # 5. DINO perceptual loss (optional)
        if self.dino_extractor is not None:
            orig_dino_feats = self.dino_extractor(original_video)
            recon_dino_feats = self.dino_extractor(recon)
            p = F.softmax(orig_dino_feats, dim=-1)
            q = F.log_softmax(recon_dino_feats, dim=-1)
            dino_loss = F.kl_div(q, p, reduction="batchmean")
        else:
            dino_loss = torch.tensor(0.0, device=self.device)

        # 6. KL regularisation from the latent posterior
        kl_loss = latent_dist.kl().mean()

        total_loss = (
            recon_loss
            + entropy_loss_value
            + self.cfg.likelihood_loss_weight * likelihood_loss
            + self.cfg.quant_align_loss_weight * quant_align_loss
            + self.cfg.dino_loss_weight * dino_loss
            + self.cfg.kl_loss_weight * kl_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "entropy_loss": entropy_loss_value,
            "likelihood_loss": likelihood_loss,
            "quant_alignment_loss": quant_align_loss,
            "dino_perceptual_loss": dino_loss,
            "kl_loss": kl_loss,
        }

# ==============================================================================
# 5. EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running smoke tests on CPU. Expect slow execution.")

    try:
        config = VideoVAEConfig(
            quant_emb_dim=4,
            alignment_dim=64,
            quant_align_loss_weight=0.05,
            likelihood_loss_weight=0.1,
            dino_loss_weight=0.1,
            entropy_loss_weight=0.05,
            kl_loss_weight=0.1,
            freeze_autoencoder=False,

        )



        model = VideoVAE(
            config,
            device=device,
            text_encoder=RandomTextEncoderStub(hidden_size=128, seq_len=12),
            dino_extractor=RandomDINOExtractorStub(feature_dim=96),
            autoencoder=None,
        ).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("-" * 40)
        print(f"Trainable model parameters: {trainable_params:,}")
        print("(Autoencoder is unfrozen in this smoke test configuration)")
        print("-" * 40)

        # --- SIMULATED TRAINING STEP ---
        print("\n--- 1. Simulating Training Step ---")
        model.train()
        batch_size = 2
        video_input = torch.randn(batch_size, config.in_channels, 8, 32, 32, device=device)
        prompts = ["A synthetic test prompt.", "Another synthetic prompt."]

        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
        optimizer.zero_grad()

        forward_outputs = model(video_input, text_prompts=prompts)
        losses = model.calculate_losses(video_input, forward_outputs)

        losses["total_loss"].backward()
        optimizer.step()

        print("Training step successful. Loss components:")
        for name, value in losses.items():
            print(f"  - {name:<25}: {value.item():.4f}")

        # --- SIMULATED INFERENCE STEP ---
        print("\n--- 2. Simulating Inference Step ---")
        model.eval()
        with torch.no_grad():
            inference_outputs = model(video_input, text_prompts=prompts, sample_posterior=False)
            reconstructed_video = inference_outputs["reconstruction"]

        print("Inference step successful.")
        print("Input Video Shape:         ", tuple(video_input.shape))
        print("Reconstructed Video Shape: ", tuple(reconstructed_video.shape))

    except Exception as e:
        print("\n--- ❌ An Error Occurred ---")
        print(f"Error: {e}")
        if "out of memory" in str(e).lower():
            print("\n💡 Suggestion: Reduce the latent `z_dim`, sequence length, or spatial resolution for testing.")
