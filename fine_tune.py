import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

try:
    from .data import DataConfig, build_train_dataloader
    from .embedder import _get_qwen3_prompt_embeds, tokenize_prompt
    from .model import AutoencoderKLPyraTok
except ImportError:
    from data import DataConfig, build_train_dataloader
    from embedder import _get_qwen3_prompt_embeds, tokenize_prompt
    from model import AutoencoderKLPyraTok
from ssim_loss import VideoSSIMLoss

@dataclass(frozen=True)
class TrainConfig:
    # Paths (hard-coded)
    video_base_path: str = "/data/onkar/hf/data/ditto/videos/global_freeform1/global_freeform1"
    fallback_video_base_paths: tuple[str, ...] = ("/data/onkar/hf/data/ditto/videos/source/source",)
    train_manifest_path: str = "/data/onkar/hf/data/ditto/source_video_captions/source_video_captions_sorted.json"
    pyratok_pretrained_path: str = "/data/onkar/PyraTok/vae"
    qwen_model_path: str = "/data/onkar/PyraTok/text_encoder"
    output_dir: str = "/data/home/onkar/PyraTok/checkpoints_lang_guided_accelerate_bf16"

    # Accelerate / multi-GPU (hard-coded)
    mixed_precision: str = "bf16"
    gradient_accumulation_steps: int = 4
    offload_text_encoder_to_cpu: bool = True

    # Training setup (hard-coded)
    seed: int = 1234
    num_epochs: int = 5
    max_steps: int = 20000
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    grad_clip_norm: float = 4.0
    sample_posterior: bool = False

    # Data setup (hard-coded)
    batch_size: int = 1
    num_workers: int = 1
    num_frames: int = 17
    train_height: int | None = None
    train_width: int | None = None
    max_sequence_length: int = 10
    shuffle: bool = False
    pin_memory: bool = False
    drop_last: bool = True

    # Loss weights (hard-coded)
    recon_weight: float = 1.0
    kl_weight: float = 1e-6
    lapq_weight: float = 1.0

    # Logging/checkpointing (hard-coded)
    log_every: int = 10
    save_every: int = 100

    # LaPQ config (hard-coded)
    use_lapq_quantizer: bool = True
    lapq_num_codes: int = 2**16
    lapq_num_quantizers: int = 4
    lapq_codebook_dim: int | None = 16
    lapq_commitment_weight: float = 0.5
    lapq_entropy_weight: float = 0.5
    lapq_inv_temperature: float = 100.0
    lapq_quantize_dropout: bool = False
    lapq_quantize_dropout_cutoff_index: int = 0
    lapq_quantize_dropout_multiple_of: int = 1
    lapq_text_input_dim: int | None = None
    lapq_text_embed_dim: int | None = 256
    lapq_text_mlp_hidden_dim: int | None = 1024
    lapq_text_condition_heads: int | None = 2
    lapq_text_condition_scale: float = 0.7


def build_data_cfg(cfg: TrainConfig) -> DataConfig:
    return DataConfig(
        video_base_path=cfg.video_base_path,
        fallback_video_base_paths=cfg.fallback_video_base_paths,
        manifest_path=cfg.train_manifest_path,
        num_frames=cfg.num_frames,
        height=cfg.train_height,
        width=cfg.train_width,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=cfg.shuffle,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        verify_video_exists=False,
    )


def build_vae(
    cfg: TrainConfig,
    lapq_text_input_dim: int | None,
    lapq_text_embed_dim: int | None,
) -> AutoencoderKLPyraTok:
    model_kwargs = dict(
        use_lapq_quantizer=cfg.use_lapq_quantizer,
        lapq_num_codes=cfg.lapq_num_codes,
        lapq_num_quantizers=cfg.lapq_num_quantizers,
        lapq_codebook_dim=cfg.lapq_codebook_dim,
        lapq_commitment_weight=cfg.lapq_commitment_weight,
        lapq_entropy_weight=cfg.lapq_entropy_weight,
        lapq_inv_temperature=cfg.lapq_inv_temperature,
        lapq_quantize_dropout=cfg.lapq_quantize_dropout,
        lapq_quantize_dropout_cutoff_index=cfg.lapq_quantize_dropout_cutoff_index,
        lapq_quantize_dropout_multiple_of=cfg.lapq_quantize_dropout_multiple_of,
        lapq_text_input_dim=lapq_text_input_dim,
        lapq_text_embed_dim=lapq_text_embed_dim,
        lapq_text_mlp_hidden_dim=cfg.lapq_text_mlp_hidden_dim,
        lapq_text_condition_heads=cfg.lapq_text_condition_heads,
        lapq_text_condition_scale=cfg.lapq_text_condition_scale,
    )
    # Important: disable low_cpu_mem_usage because LaPQ modules are newly introduced and are
    # not present in the pretrained checkpoint. With meta-init loading, missing params can stay
    # on meta device and later fail when Accelerate moves the model to GPU.
    try:
        return AutoencoderKLPyraTok.from_pretrained(
            cfg.pyratok_pretrained_path,
            low_cpu_mem_usage=False,
            device_map=None,
            **model_kwargs,
        )
    except TypeError:
        # Fallback for older diffusers versions that don't expose these kwargs.
        return AutoencoderKLPyraTok.from_pretrained(cfg.pyratok_pretrained_path, **model_kwargs)


def build_text_encoder(cfg: TrainConfig, device: torch.device) -> tuple[Qwen2TokenizerFast, Qwen3ForCausalLM]:
    tokenizer = Qwen2TokenizerFast.from_pretrained(cfg.qwen_model_path)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    if cfg.offload_text_encoder_to_cpu:
        text_encoder.to("cpu")
    else:
        text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return tokenizer, text_encoder


@torch.no_grad()
def encode_instructions(
    instructions: list[str],
    tokenizer: Qwen2TokenizerFast,
    text_encoder: Qwen3ForCausalLM,
    device: torch.device,
    max_sequence_length: int,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=instructions,
        max_sequence_length=max_sequence_length,
    )
    text_embeds = _get_qwen3_prompt_embeds(
        text_encoder=text_encoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=out_dtype,
        device=device,
    )
    return text_embeds, attention_mask.to(device)


def _distributed_mean(accelerator: Accelerator, scalar_tensor: torch.Tensor) -> float:
    gathered = accelerator.gather(scalar_tensor.detach().float().view(1))
    return float(gathered.mean().item())


def save_checkpoint(
    accelerator: Accelerator,
    cfg: TrainConfig,
    model: AutoencoderKLPyraTok,
    optimizer: AdamW,
    step: int,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"checkpoint_step_{step:07d}.pt"

    unwrapped = accelerator.unwrap_model(model)
    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    accelerator.save(payload, ckpt_path)
    unwrapped.save_pretrained(output_dir / f"vae_checkpoint_step_{step:07d}")
    accelerator.print(f"[checkpoint] saved: {ckpt_path}")


def train() -> None:
    cfg = TrainConfig()
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

    set_seed(cfg.seed, device_specific=True)
    random.seed(cfg.seed + accelerator.process_index)

    if accelerator.is_main_process:
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "hardcoded_train_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2)
    accelerator.wait_for_everyone()

    dataloader = build_train_dataloader(build_data_cfg(cfg))
    tokenizer, text_encoder = build_text_encoder(cfg, accelerator.device)

    qwen_hidden_size = int(text_encoder.config.hidden_size)
    lapq_text_input_dim = cfg.lapq_text_input_dim if cfg.lapq_text_input_dim is not None else qwen_hidden_size
    lapq_text_embed_dim = cfg.lapq_text_embed_dim if cfg.lapq_text_embed_dim is not None else qwen_hidden_size
    model = build_vae(
        cfg,
        lapq_text_input_dim=lapq_text_input_dim,
        lapq_text_embed_dim=lapq_text_embed_dim,
    )
    ssimcriterion = VideoSSIMLoss(data_range=2.0, reduction="mean").to(accelerator.device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_gpus = accelerator.num_processes
    local_batches_per_epoch = len(dataloader)
    num_update_steps_per_epoch = math.ceil(local_batches_per_epoch / cfg.gradient_accumulation_steps)
    total_train_steps = min(cfg.max_steps, cfg.num_epochs * num_update_steps_per_epoch)
    effective_global_batch_size = cfg.batch_size * num_gpus * cfg.gradient_accumulation_steps

    if accelerator.is_main_process:
        accelerator.print(
            "Training setup: "
            f"num_gpus={num_gpus}, "
            f"grad_accum={cfg.gradient_accumulation_steps}, "
            f"epochs={cfg.num_epochs}, "
            f"local_batches_per_epoch={local_batches_per_epoch}, "
            f"updates_per_epoch={num_update_steps_per_epoch}, "
            f"total_updates={total_train_steps}, "
            f"effective_global_batch_size={effective_global_batch_size}, "
            f"text_embed_in={lapq_text_input_dim}, "
            f"text_embed_out={lapq_text_embed_dim}"
        )
    progress_bar = tqdm(
        total=total_train_steps,
        desc="train",
        disable=not accelerator.is_main_process,
    )

    if accelerator.mixed_precision == "bf16":
        text_out_dtype = torch.bfloat16
        data_out_dtype = torch.bfloat16
    else:
        text_out_dtype = next(text_encoder.parameters()).dtype
        data_out_dtype = next(accelerator.unwrap_model(model).parameters()).dtype

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(cfg.num_epochs):
            for batch in dataloader:
                with accelerator.accumulate(model):
                    source_video = batch["source_video"].to(
                        accelerator.device, dtype=data_out_dtype, non_blocking=True
                    )
                    # print(source_video.shape, source_video.dtype)
                    # print(""source_video.min(), source_video.max())
                    target_video = batch["target_video"].to(
                        accelerator.device, dtype=data_out_dtype, non_blocking=True
                    )
                    instructions = batch["instructions"]

                    if cfg.offload_text_encoder_to_cpu:
                        text_encoder.to(accelerator.device)

                    text_embeds, text_attention_mask = encode_instructions(
                        instructions=instructions,
                        tokenizer=tokenizer,
                        text_encoder=text_encoder,
                        device=accelerator.device,
                        max_sequence_length=cfg.max_sequence_length,
                        out_dtype=text_out_dtype,
                    )

                    if cfg.offload_text_encoder_to_cpu:
                        text_encoder.to("cpu")
                        if accelerator.device.type == "cuda":
                            torch.cuda.empty_cache()

                
                    forward_out = model(
                        sample=source_video,
                        sample_posterior=False,
                        return_dict=True,
                        text_embeds=text_embeds,
                        text_attention_mask=text_attention_mask,
                        use_lapq_quantizer=cfg.use_lapq_quantizer,
                        return_loss_details=True,
                    )
                    decoded = forward_out.sample

                    recon_loss = F.l1_loss(decoded, target_video)
                    lapq_loss = (
                        forward_out.quantization_loss
                        if forward_out.quantization_loss is not None
                        else recon_loss.new_zeros(())
                    )
                    ssim = ssimcriterion(decoded, target_video)

                    total_loss = (
                        cfg.recon_weight * recon_loss
                        + cfg.lapq_weight * lapq_loss 
                        + 1.2 * ssim
                    )

                    accelerator.backward(total_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                
                # print("Total Loss: ", total_loss.item())

                if accelerator.sync_gradients and global_step % cfg.log_every == 0:
                    perplexity = (
                        forward_out.lapq_perplexity
                        if forward_out.lapq_perplexity is not None
                        else total_loss.new_zeros(())
                    )
                    mean_total = _distributed_mean(accelerator, total_loss)
                    mean_recon = _distributed_mean(accelerator, recon_loss)
                    mean_lapq = _distributed_mean(accelerator, lapq_loss)
                    mean_ssim = _distributed_mean(accelerator, ssim)
                    # mean_perplexity = _distributed_mean(accelerator, perplexity)

                    accelerator.print(
                        f"[step {global_step:07d}] "
                        f"loss={mean_total:.6f} "
                        f"recon={mean_recon:.6f} "
                        f"kl={ssim:.6f} "
                        f"lapq={mean_lapq:.6f} "
                        # f"perplexity={mean_perplexity:.4f}"
                    )
                    if accelerator.is_main_process:
                        progress_bar.set_postfix(
                            loss=f"{mean_total:.4f}",
                            recon=f"{mean_recon:.4f}",
                            SSIM=f"{mean_ssim:.4f}",
                            lapq=f"{mean_lapq:.4f}",
                        )

                if accelerator.sync_gradients and global_step > 0 and global_step % cfg.save_every == 0:
                    save_checkpoint(accelerator, cfg, model, optimizer, step=global_step, epoch=epoch)

                if global_step >= total_train_steps:
                    break

            if global_step >= total_train_steps:
                break

        save_checkpoint(accelerator, cfg, model, optimizer, step=global_step, epoch=epoch)
        accelerator.print("Training complete.")
    finally:
        progress_bar.close()
        accelerator.end_training()


if __name__ == "__main__":
    train()
