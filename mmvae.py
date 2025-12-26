# video_vae_modular_final.py

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan
from typing import Dict, List, Mapping, Optional
from dataclasses import dataclass

@dataclass
class VideoVAEConfig:
    in_channels: int = 3
    quant_emb_dim: int = 16
    alignment_dim: int = 256
    quant_align_loss_weight: float = 0.1
    dino_loss_weight: float = 0.25
    entropy_loss_weight: float = 0.1
    commit_loss_weight: float = 0.25
    wan_pretrained_path: Optional[str] = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    wan_subfolder: str = "vae"
    wan_torch_dtype: str = "float16"
    freeze_autoencoder: bool = True
    num_quant_levels: int = 4

# ==============================================================================
# 2. PERCEPTUAL & TEXT MODULES
# ==============================================================================

class DINOv2Extractor(nn.Module):
    """
    A frozen DINOv2 model to extract perceptual features from video frames.
    """
    def __init__(self, device="cuda"):
        super().__init__()
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
        # inputs = self.processor(images=video_tensor, return_tensors="pt", do_rescale=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(video_tensor)
        # Return the features of the [CLS] token
        return outputs.last_hidden_state[:, 0].view(b, t, -1)

class UMT5TextEncoder(nn.Module):
    """Frozen UM-T5 encoder that consumes pre-tokenized inputs."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        model_id = "google/umt5-xxl"
        self.device = device
        print("Loading UM-T5 encoder...")
        self.model = UMT5EncoderModel.from_pretrained(
            model_id,
            torch_dtype="auto",
            cache_dir="/data2/onkar/llava",
        ).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.d_model
        print("UM-T5 encoder loaded and frozen successfully. 🧠")

    def forward(self, tokenized_inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inputs = {
            key: value.to(self.device)
            for key, value in tokenized_inputs.items()
            if torch.is_tensor(value)
        }
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state

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
        attn_output = F.scaled_dot_product_attention(q, k, v)
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
        quantized_x = x_proj + (quantized_x_hard - x_proj).detach().to(torch.float16)
        indices = (quantized_x > 0).long()
        probs = indices.float().mean(dim=(0, 2, 3, 4)).to(torch.float16)
        entropy = - (probs * torch.log(probs.clamp(min=1e-8)) + (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))
        entropy_loss = -entropy.mean() * self.entropy_loss_weight
        return quantized_x, indices, entropy_loss

class ResidualQuantBlock(nn.Module):
    """Residual vector quantization block operating directly in latent space."""

    def __init__(self, latent_channels: int, text_embed_dim: int, quant_emb_dim: int, entropy_weight: float) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(latent_channels),
            nn.GELU(),
            nn.Conv3d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(latent_channels),
            nn.GELU(),
        )
        self.text_cross_attn = TextVideoCrossAttention(latent_channels, text_embed_dim)
        self.lfq = ProjectedLFQ(latent_channels, quant_channels=quant_emb_dim, entropy_loss_weight=entropy_weight)
        self.commit_weight = None  # to be set externally if needed
        self.reconstruct = nn.Conv3d(quant_emb_dim, latent_channels, kernel_size=1)

    def forward(self, residual: torch.Tensor, text_embedding: torch.Tensor):
        features = self.pre(residual)
        features = features + self.text_cross_attn(features, text_embedding)
        quantized, indices, entropy_loss = self.lfq(features)
        quant_recon = self.reconstruct(quantized)
        new_residual = residual - quant_recon
        commit_loss = None
        if self.commit_weight is not None:
            commit_loss = (
                F.mse_loss(quant_recon.detach(), features) +
                F.mse_loss(quant_recon, features.detach())
            ) * self.commit_weight
        return quantized, quant_recon, new_residual, indices, entropy_loss, commit_loss

# ==============================================================================
# 4. PRIMARY VideoVAE MODEL
# ==============================================================================

class VideoVAE(nn.Module):
    """
    A modular, text-conditioned Video VAE with a Pyramidal LFQ structure
    and multiple perception-based losses for high-quality synthesis.
    """
    def __init__(self, cfg: VideoVAEConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- WAN 2.2 Autoencoder ---

        dtype = getattr(torch, cfg.wan_torch_dtype, torch.float16)
        self.autoencoder = AutoencoderKLWan.from_pretrained(
            cfg.wan_pretrained_path,
            subfolder=cfg.wan_subfolder,
            torch_dtype=dtype,
            cache_dir="/data2/onkar/llava"
        )

        self.autoencoder.to(device)
        if cfg.freeze_autoencoder:
            for param in self.autoencoder.parameters():
                param.requires_grad = False

        self.latent_channels = self.autoencoder.config.z_dim

        # --- Sub-models (Text, Perception) ---
        self.text_encoder = UMT5TextEncoder(device=device)
        text_embed_dim = self.text_encoder.hidden_size
        if self.training:  # Only load DINOv2 if we are in training mode
            self.dino_extractor = DINOv2Extractor(device=device)
        else:
            self.dino_extractor = None

        # --- Residual Quantization Pyramid ---
        self.pyramid_blocks = nn.ModuleList()
        for _ in range(cfg.num_quant_levels):
            block = ResidualQuantBlock(
                latent_channels=self.latent_channels,
                text_embed_dim=text_embed_dim,
                quant_emb_dim=cfg.quant_emb_dim,
                entropy_weight=cfg.entropy_loss_weight,
            )
            block.commit_weight = cfg.commit_loss_weight
            self.pyramid_blocks.append(block)

        # --- Loss-specific Modules ---
        self.quant_proj = nn.Linear(cfg.quant_emb_dim * cfg.num_quant_levels, cfg.alignment_dim)
        self.text_proj_for_quant = nn.Linear(text_embed_dim, cfg.alignment_dim)

    def forward(
        self,
        x: torch.Tensor,
        text_inputs: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Core inference path. Encodes, quantizes via pyramid, and decodes.
        Returns all intermediate products needed for loss calculation.
        """

        text_embedding = self.text_encoder(text_inputs)

        latent_dist = self.autoencoder.encode(x).latent_dist
        latents = latent_dist.sample()
        text_embedding = text_embedding.to(latents.dtype)

        residual = latents
        latent_reconstruction = torch.zeros_like(latents)
        pyramid_outputs = {"q": [], "indices": [], "entropies": [], "commit": []}

        for block in self.pyramid_blocks:
            quant_codes, quant_recon, residual, indices, entropy, commit = block(residual, text_embedding)
            latent_reconstruction = latent_reconstruction + quant_recon
            pyramid_outputs["q"].append(quant_codes)
            pyramid_outputs["indices"].append(indices)
            pyramid_outputs["entropies"].append(entropy)
            pyramid_outputs["commit"].append(commit)

        latent_reconstruction = latent_reconstruction + residual
        reconstruction = self.autoencoder.decode(latent_reconstruction).sample
        if reconstruction.shape[2:] != x.shape[2:]:
            reconstruction = F.interpolate(
                reconstruction,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return {
            "reconstruction": reconstruction,
            "text_embedding": text_embedding,
            "pyramid_outputs": pyramid_outputs,
            "latent_dist": latent_dist,
            "residual_reconstruction": latent_reconstruction,
        }

    def calculate_losses(self, original_video: torch.Tensor, forward_outputs: Dict) -> Dict:
        """
        Calculates all training-specific losses. This method should only be
        called during the training loop.
        """
        if not self.training:
            raise RuntimeError("calculate_losses() should only be called in training mode.")
            
        # Unpack forward pass results
        recon = forward_outputs["reconstruction"]
        text_emb = forward_outputs["text_embedding"]
        pyramid_out = forward_outputs["pyramid_outputs"]
        all_q = pyramid_out["q"]
        all_entropies = pyramid_out["entropies"]
        all_commit = pyramid_out.get("commit", [])

        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon, original_video)

        # 2. Entropy Loss
        entropy_terms = [loss for loss in all_entropies if loss is not None]
        entropy_loss = torch.stack(entropy_terms).sum() if entropy_terms else torch.tensor(0.0, device=self.device)

        # 3. Commitment Loss
        commit_terms = [loss for loss in all_commit if loss is not None]
        commit_loss = torch.stack(commit_terms).sum() if commit_terms else torch.tensor(0.0, device=self.device)

        # 4. Quantized/Text KL Alignment
        B = text_emb.size(0)
        q_pooled = [F.adaptive_avg_pool3d(q, 1).view(B, -1) for q in all_q]
        q_pooled_cat = torch.cat(q_pooled, dim=1)
        text_pooled = text_emb.mean(dim=1)
        q_aligned = self.quant_proj(q_pooled_cat)
        text_aligned = self.text_proj_for_quant(text_pooled)
        quant_distribution = F.log_softmax(q_aligned.float(), dim=-1)
        text_distribution = F.softmax(text_aligned.float(), dim=-1)
        quant_align_loss = F.kl_div(quant_distribution, text_distribution, reduction="batchmean")
        
        # 5. DINOv2 Perceptual Loss (KL Divergence)
        if self.dino_extractor is not None:
            orig_dino_feats = self.dino_extractor(original_video)
            recon_dino_feats = self.dino_extractor(recon)
            p = F.softmax(orig_dino_feats, dim=-1)
            q = F.log_softmax(recon_dino_feats, dim=-1)
            dino_loss = F.kl_div(q, p, reduction='batchmean')
        else:
            dino_loss = torch.tensor(0.0, device=self.device)

        # --- Final Weighted Sum ---
        recon_loss = recon_loss.float()
        entropy_loss = entropy_loss.float()
        commit_loss = commit_loss.float()
        quant_align_loss = quant_align_loss.float()
        dino_loss = dino_loss.float()

        total_loss = (
            recon_loss
            + entropy_loss
            + self.cfg.commit_loss_weight * commit_loss
            + self.cfg.quant_align_loss_weight * quant_align_loss
            + self.cfg.dino_loss_weight * dino_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "entropy_loss": entropy_loss,
            "commit_loss": commit_loss,
            "quant_alignment_loss": quant_align_loss,
            "dino_perceptual_loss": dino_loss,
        }

# ==============================================================================
# 5. EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARNING: Running on CPU. This will be extremely slow.")

    config = VideoVAEConfig(quant_emb_dim=16) # Set LFQ size to 16
    model = VideoVAE(config, device=device).to(device, dtype=torch.float16)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-" * 40)
    print(f"Trainable model parameters: {trainable_params:,}")
    print("(This should NOT include frozen DINOv2 or Qwen-VL models)")
    print("-" * 40)

    # --- SIMULATED TRAINING STEP ---
    print("\n--- 1. Simulating Training Step ---")
    model.train() # Set model to training mode
    batch_size = 1
    video_input = torch.randn((batch_size, 3, 41, 720, 720), dtype=torch.float16).to(device).clip(0,1)
    prompts = ["A stunning sunrise over a calm ocean."]
    
    # In a real training loop, this would be inside the loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "google/umt5-xxl",
        cache_dir="/data2/onkar/llava",
    )

    def tokenize(prompts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        return {k: v.to(device) for k, v in encoded.items()}

    tokenized_prompts = tokenize(prompts)

    forward_outputs = model(video_input, text_inputs=tokenized_prompts)

    print(forward_outputs["reconstruction"].shape)
    losses = model.calculate_losses(video_input, forward_outputs)


    
    # Backpropagation
    losses["total_loss"].backward()
    optimizer.step()
    
    print("Training step successful. Losses calculated:")
    for name, value in losses.items(): print(f"  - {name:<25}: {value.item():.4f}")

    # --- SIMULATED INFERENCE STEP ---
    print("\n--- 2. Simulating Inference Step ---")
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Notice we only call the forward pass and don't need the loss function
        inference_outputs = model(video_input, text_inputs=tokenized_prompts)
        reconstructed_video = inference_outputs["reconstruction"]

    print("Inference step successful.")
    print("Input Video Shape:         ", video_input.shape)
    print("Reconstructed Video Shape: ", reconstructed_video.shape)

