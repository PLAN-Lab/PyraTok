import argparse
import math
import os
from dataclasses import asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from mmvae import VideoVAE, VideoVAEConfig


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _to_numpy_frames(video_entry) -> np.ndarray:
    if isinstance(video_entry, dict):
        if "array" in video_entry:
            arr = video_entry["array"]
        elif "frames" in video_entry:
            frames = video_entry["frames"]
            arr = np.stack([np.asarray(frame) for frame in frames])
        else:
            values = list(video_entry.values())[0]
            arr = np.asarray(values)
    elif hasattr(video_entry, "numpy"):
        arr = video_entry.numpy()
    else:
        arr = np.asarray(video_entry)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _sample_frames(frames: np.ndarray, num_frames: int) -> torch.Tensor:
    total_frames = frames.shape[0]
    if total_frames == 0:
        raise ValueError("Video sample contains zero frames.")
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    else:
        indices = np.concatenate(
            [np.arange(total_frames), np.full(num_frames - total_frames, total_frames - 1, dtype=np.int32)]
        )
    sampled = torch.from_numpy(frames[indices]).float()  # (T, H, W, C)
    sampled = sampled.permute(0, 3, 1, 2)  # (T, C, H, W)
    sampled = sampled / 127.5 - 1.0
    return sampled


class OpenVidDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: AutoTokenizer,
        cache_dir: str,
        num_frames: int,
        frame_size: int,
    ) -> None:
        self.dataset = load_dataset("openvid-1m", split=split, cache_dir=cache_dir)
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.text_keys = [
            key
            for key in ("caption", "text", "prompt", "description")
            if key in self.dataset.column_names
        ]
        if not self.text_keys:
            raise ValueError("Dataset does not contain a text/caption column.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        video_entry = example["video"]
        frames = _to_numpy_frames(video_entry)
        frames = _sample_frames(frames, self.num_frames)
        frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)
        video_tensor = frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

        caption = None
        for key in self.text_keys:
            if example.get(key):
                caption = example[key]
                break
        if caption is None:
            caption = ""

        max_len = min(256, self.tokenizer.model_max_length)
        tokenized = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_len,
        )
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        return {"video": video_tensor, "text_inputs": tokenized}


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    videos = torch.stack([sample["video"] for sample in batch], dim=0)
    text_inputs: Dict[str, List[torch.Tensor]] = {}
    for sample in batch:
        for key, value in sample["text_inputs"].items():
            text_inputs.setdefault(key, []).append(value)
    text_inputs = {key: torch.stack(values, dim=0) for key, values in text_inputs.items()}
    return {"video": videos, "text_inputs": text_inputs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MM-VAE on OpenVid-1M")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/mmvae")
    parser.add_argument("--cache_dir", type=str, default="/data2/onkar/llava")
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--frame_size", type=int, default=1920)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_train_steps", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dino_loss_weight", type=float, default=0.25)
    parser.add_argument("--quant_align_weight", type=float, default=0.1)
    parser.add_argument("--commit_weight", type=float, default=0.25)
    parser.add_argument("--entropy_weight", type=float, default=0.1)
    parser.add_argument("--quant_emb_dim", type=int, default=16)
    parser.add_argument("--num_quant_levels", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--wan_checkpoint", type=str, default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--wan_dtype", type=str, default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dataset_split", type=str, default="train[:1%]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(
        "google/umt5-xxl",
        cache_dir=args.cache_dir,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = OpenVidDataset(
        split=args.dataset_split,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model_cfg = VideoVAEConfig(
        in_channels=3,
        quant_emb_dim=args.quant_emb_dim,
        alignment_dim=512,
        quant_align_loss_weight=args.quant_align_weight,
        dino_loss_weight=args.dino_loss_weight,
        entropy_loss_weight=args.entropy_weight,
        commit_loss_weight=args.commit_weight,
        wan_pretrained_path=args.wan_checkpoint,
        wan_subfolder="vae",
        wan_torch_dtype=args.wan_dtype,
        freeze_autoencoder=True,
        num_quant_levels=args.num_quant_levels,
    )

    model = VideoVAE(model_cfg, device=accelerator.device).to(accelerator.device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.grad_accum)
    max_train_steps = args.max_train_steps
    if max_train_steps <= 0:
        max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    global_step = 0
    model.train()

    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            videos = batch["video"].to(accelerator.device, dtype=torch.float32)
            text_inputs = {k: v.to(accelerator.device) for k, v in batch["text_inputs"].items()}

            outputs = model(videos, text_inputs=text_inputs)
            losses = model.calculate_losses(videos, outputs)
            loss = losses["total_loss"] / args.grad_accum

            accelerator.backward(loss)

            if (step + 1) % args.grad_accum == 0:
                accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process and global_step % args.log_every == 0:
                    log_items = {
                        "loss": losses["total_loss"].item(),
                        "recon_loss": losses["reconstruction_loss"].item(),
                        "entropy_loss": losses["entropy_loss"].item(),
                        "commit_loss": losses["commit_loss"].item(),
                        "quant_align_loss": losses["quant_alignment_loss"].item(),
                    }
                    accelerator.print(f"step {global_step}: {log_items}")

                if accelerator.is_main_process and global_step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    unwrapped = accelerator.unwrap_model(model)
                    save_path = os.path.join(args.output_dir, f"step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(unwrapped.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
                    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
                        f.write(str(asdict(model_cfg)))

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "final")
        os.makedirs(final_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), os.path.join(final_path, "pytorch_model.bin"))
        with open(os.path.join(final_path, "config.json"), "w", encoding="utf-8") as f:
            f.write(str(asdict(model_cfg)))


if __name__ == "__main__":
    main()
