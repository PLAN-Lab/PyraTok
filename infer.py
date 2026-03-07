import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import imageio.v3 as iio

    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from .data import _read_video_tchw, _resize_video
    from .model import AutoencoderKLPyraTok
except ImportError:
    from data import _read_video_tchw, _resize_video
    from model import AutoencoderKLPyraTok
import torch.nn.functional as F
from ssim_loss import VideoSSIMLoss
# Runtime string inputs (no argparse).
# Option 1: set these two strings directly.
# Option 2: pass strings at launch:
#   python infer.py "/abs/or/rel/video.mp4" "your text prompt"
INPUT_VIDEO_PATH = "/data/home/onkar/PyraTok/80e791d3c241bc97a6e2e09d88f8e669_1.mp4"
INPUT_TEXT = "animation"
USE_TEXT_CONDITION = True


@dataclass(frozen=True)
class InferConfig:
    # Model
    vae_model_path: str = "/data/home/onkar/PyraTok/checkpoints_lang_guided_accelerate_bf16/vae"
    lapq_num_codes: int = 2**16
    lapq_num_quantizers: int = 4
    lapq_codebook_dim: int | None = 16
    lapq_commitment_weight: float = 0.8
    lapq_entropy_weight: float = 1.0
    lapq_inv_temperature: float = 100.0
    lapq_quantize_dropout: bool = False
    lapq_quantize_dropout_cutoff_index: int = 0
    lapq_quantize_dropout_multiple_of: int = 1
    lapq_text_input_dim: int | None = None
    lapq_text_embed_dim: int | None = 256
    lapq_text_mlp_hidden_dim: int | None = 1024
    lapq_text_condition_heads: int | None = 2
    lapq_text_condition_scale: float = 0.7

    # Data input
    input_video_path: str | None = None  # Absolute path, or relative path under video_base_path.
    input_caption: str = "two girls drinking wine in a restaurant"
    video_base_path: str = "/data/home/onkar/PyraTok/6236533_anastasia__shuraeva_2.mp4"
    fallback_video_base_paths: tuple[str, ...] = ("/data/onkar/hf/data/ditto/videos/source/source",)
    manifest_path: str = "/data/onkar/hf/data/ditto/source_video_captions/source_video_captions_sorted.json"
    num_samples: int = 4
    random_manifest_samples: bool = True
    seed: int = 1234

    # Preprocess
    num_frames: int = 17
    window_stride: int | None = None  # None -> use num_frames (non-overlapping windows)
    height: int | None = 256
    width: int | None = 512
    fps: int = 16

    # Text-conditioning (optional)
    use_text_condition: bool = True
    qwen_model_path: str = "/data/onkar/PyraTok/text_encoder"
    max_sequence_length: int = 10

    # Runtime
    device: str = "auto"  # auto | cuda | cpu
    dtype: str = "auto"  # auto | bf16 | fp16 | fp32
    output_dir: str = "./reconstructions"


def _select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda:9" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def _select_dtype(dtype_cfg: str, device: torch.device) -> torch.dtype:
    if dtype_cfg == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if _cuda_bf16_supported() else torch.float16
        return torch.float32
    if dtype_cfg == "bf16":
        if device.type == "cpu":
            return torch.float32
        return torch.bfloat16
    if dtype_cfg == "fp16":
        if device.type == "cpu":
            return torch.float32
        return torch.float16
    if dtype_cfg == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype config: {dtype_cfg}")


def _resolve_video_path(
    path: str,
    video_base_path: str,
    fallback_video_base_paths: tuple[str, ...],
) -> str:
    p = Path(path)
    if p.is_absolute():
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Input video path not found: {path}")

    roots = (video_base_path, *fallback_video_base_paths)
    tried = []
    for root in roots:
        candidate = Path(root) / path
        tried.append(str(candidate))
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"Video file not found for '{path}'. Tried: {tried}")


def _load_manifest(manifest_path: str) -> list[dict[str, Any]]:
    if manifest_path.endswith(".jsonl"):
        rows: list[dict[str, Any]] = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if manifest_path.endswith(".json"):
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON manifest must be a list.")
        return data

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _build_eval_samples(cfg: InferConfig) -> list[dict[str, str]]:
    if cfg.input_video_path is not None:
        return [{"path": cfg.input_video_path, "caption": cfg.input_caption}]

    rows = _load_manifest(cfg.manifest_path)
    if len(rows) == 0:
        raise RuntimeError(f"No rows found in manifest: {cfg.manifest_path}")

    num = min(cfg.num_samples, len(rows))
    if cfg.random_manifest_samples:
        rng = random.Random(cfg.seed)
        picks = rng.sample(rows, k=num)
    else:
        picks = rows[:num]

    out = []
    for row in picks:
        out.append(
            {
                "path": str(row["path"]),
                "caption": str(row.get("caption", "")),
            }
        )
    return out


def _read_runtime_inputs() -> tuple[str, str]:
    if len(sys.argv) >= 2:
        video_path = sys.argv[1]
        text = sys.argv[2] if len(sys.argv) >= 3 else ""
    else:
        video_path = INPUT_VIDEO_PATH
        text = INPUT_TEXT

    video_path = video_path.strip()
    if not video_path:
        raise ValueError(
            "Missing input video path. Set INPUT_VIDEO_PATH in infer.py or run: "
            "python infer.py \"/path/to/video.mp4\" \"text prompt\""
        )
    return video_path, text


def _cthw_to_uint8_thwc(video_cthw: torch.Tensor):
    video = video_cthw.detach().float().cpu().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().numpy()


def _write_video(path: Path, video_cthw: torch.Tensor, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = _cthw_to_uint8_thwc(video_cthw)

    if _HAS_IMAGEIO:
        try:
            iio.imwrite(path, frames, fps=fps)
            return
        except Exception:
            pass

    if _HAS_CV2:
        t, h, w, _ = frames.shape
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open OpenCV VideoWriter for: {path}")
        try:
            for i in range(t):
                writer.write(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return

    raise ImportError("Need imageio or opencv-python to write videos.")


def _load_full_video_as_cthw(path: str, height: int | None, width: int | None) -> torch.Tensor:
    video_tchw = _read_video_tchw(path)
    video_tchw = _resize_video(video_tchw, height=height, width=width)
    if video_tchw.dtype != torch.float32:
        video_tchw = video_tchw.float()
    video_tchw = video_tchw / 255.0
    video_tchw = video_tchw * 2.0 - 1.0
    return video_tchw.permute(1, 0, 2, 3).contiguous()


def _pad_window_to_num_frames(window_cthw: torch.Tensor, num_frames: int) -> tuple[torch.Tensor, int]:
    valid_len = int(window_cthw.shape[1])
    if valid_len <= 0:
        raise RuntimeError("Sliding window received an empty temporal chunk.")
    if valid_len == num_frames:
        return window_cthw, valid_len
    if valid_len > num_frames:
        return window_cthw[:, :num_frames], num_frames

    pad = window_cthw[:, -1:, :, :].repeat(1, num_frames - valid_len, 1, 1)
    return torch.cat([window_cthw, pad], dim=1), valid_len


@torch.no_grad()
def _run_sliding_window_inference(
    model: AutoencoderKLPyraTok,
    source_cthw: torch.Tensor,
    cfg: InferConfig,
    device: torch.device,
    dtype: torch.dtype,
    text_embeds: torch.Tensor | None,
    text_attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    window_size = int(cfg.num_frames)
    if window_size <= 0:
        raise ValueError(f"num_frames must be > 0, got {cfg.num_frames}.")
    stride = int(cfg.window_stride) if cfg.window_stride is not None else window_size
    if stride <= 0:
        raise ValueError(f"window_stride must be > 0, got {cfg.window_stride}.")

    total_frames = int(source_cthw.shape[1])
    starts = list(range(0, total_frames, stride))
    recon_sum = torch.zeros_like(source_cthw, dtype=torch.float32)
    recon_count = torch.zeros(total_frames, dtype=torch.float32)

    for win_idx, start in enumerate(starts):
        end = min(start + window_size, total_frames)
        window_cthw = source_cthw[:, start:end]
        window_cthw, valid_len = _pad_window_to_num_frames(window_cthw, num_frames=window_size)
        window = window_cthw.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)

        out = model(
            sample=window,
            sample_posterior=False,
            return_dict=True,
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
            use_lapq_quantizer=cfg.use_lapq_quantizer,
            return_loss_details=False,
        )
        recon_window = out.sample.squeeze(0).float().cpu()[:, :valid_len]
        recon_sum[:, start:end] += recon_window
        recon_count[start:end] += 1.0
        print(
            f"[infer] window {win_idx + 1}/{len(starts)} "
            f"frames={start}:{end} valid={valid_len} padded={window_size - valid_len}"
        )

    return recon_sum / recon_count.clamp_min(1.0).view(1, total_frames, 1, 1)


def _resolve_lapq_text_dims(cfg: InferConfig, text_encoder) -> tuple[int | None, int | None]:
    hidden_size = None
    if text_encoder is not None:
        hidden_size = int(text_encoder.config.hidden_size)

    lapq_text_input_dim = cfg.lapq_text_input_dim
    lapq_text_embed_dim = cfg.lapq_text_embed_dim
    if cfg.use_lapq_quantizer:
        if lapq_text_input_dim is None:
            lapq_text_input_dim = hidden_size
        if lapq_text_embed_dim is None:
            lapq_text_embed_dim = hidden_size
    return lapq_text_input_dim, lapq_text_embed_dim


def _load_vae(
    cfg: InferConfig,
    device: torch.device,
    dtype: torch.dtype,
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

    try:
        model = AutoencoderKLPyraTok.from_pretrained(
            cfg.vae_model_path,
            low_cpu_mem_usage=False,
            device_map=None,
            **model_kwargs,
        )
    except TypeError:
        model = AutoencoderKLPyraTok.from_pretrained(cfg.vae_model_path, **model_kwargs)

    model.to(device=device, dtype=dtype)
    # net = torch.load("/data/home/onkar/PyraTok/checkpoints_lang_guided_accelerate_bf16/checkpoint_step_0000500.pt")
    print(model)
    model.eval()
    model.requires_grad_(False)
    return model


def _load_text_modules(
    cfg: InferConfig,
    device: torch.device,
    dtype: torch.dtype,
):
    if not cfg.use_text_condition:
        return None, None, None

    from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

    try:
        from .embedder import _get_qwen3_prompt_embeds, tokenize_prompt
    except ImportError:
        from embedder import _get_qwen3_prompt_embeds, tokenize_prompt

    tokenizer = Qwen2TokenizerFast.from_pretrained(cfg.qwen_model_path)
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
    )
    text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    return tokenizer, text_encoder, (_get_qwen3_prompt_embeds, tokenize_prompt)


@torch.no_grad()
def _encode_text(
    caption: str,
    cfg: InferConfig,
    tokenizer,
    text_encoder,
    embedder_fns,
    device: torch.device,
    dtype: torch.dtype,
):
    if tokenizer is None or text_encoder is None or embedder_fns is None:
        return None, None

    get_prompt_embeds, tokenize_prompt = embedder_fns
    input_ids, attention_mask = tokenize_prompt(
        tokenizer=tokenizer,
        prompt=[caption],
        max_sequence_length=cfg.max_sequence_length,
    )
    text_embeds = get_prompt_embeds(
        text_encoder=text_encoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        dtype=dtype,
        device=device,
    )
    return text_embeds, attention_mask.to(device)


@torch.no_grad()
def main() -> None:
    input_video_path, input_text = _read_runtime_inputs()
    cfg = InferConfig(
        input_video_path=input_video_path,
        input_caption=input_text,
        use_text_condition=USE_TEXT_CONDITION,
    )
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = _select_device(cfg.device)
    dtype = _select_dtype(cfg.dtype, device=device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, text_encoder, embedder_fns = _load_text_modules(cfg, device=device, dtype=dtype)
    lapq_text_input_dim, lapq_text_embed_dim = _resolve_lapq_text_dims(cfg, text_encoder=text_encoder)
    model = _load_vae(
        cfg,
        device=device,
        dtype=dtype,
        lapq_text_input_dim=lapq_text_input_dim,
        lapq_text_embed_dim=lapq_text_embed_dim,
    )
    samples = _build_eval_samples(cfg)

    print(
        f"[infer] samples={len(samples)} device={device} dtype={dtype} "
        f"use_lapq_quantizer={cfg.use_lapq_quantizer} use_text_condition={cfg.use_text_condition} "
        f"window_size={cfg.num_frames} window_stride={cfg.window_stride or cfg.num_frames}"
    )

    records: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        rel_or_abs_path = sample["path"]
        caption = sample.get("caption", "")
        source_path = _resolve_video_path(
            rel_or_abs_path,
            video_base_path=cfg.video_base_path,
            fallback_video_base_paths=cfg.fallback_video_base_paths,
        )

        source_cthw = _load_full_video_as_cthw(
            source_path,
            height=cfg.height,
            width=cfg.width,
        )

        text_embeds, text_attention_mask = _encode_text(
            caption=caption,
            cfg=cfg,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            embedder_fns=embedder_fns,
            device=device,
            dtype=dtype,
        )

        recon_cthw = _run_sliding_window_inference(
            model=model,
            source_cthw=source_cthw,
            cfg=cfg,
            device=device,
            dtype=dtype,
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        )

        source_batch = source_cthw.unsqueeze(0)
        recon_batch = recon_cthw.unsqueeze(0)
        ssim = VideoSSIMLoss(reduction="mean")(recon_batch, source_batch)
        recon = F.l1_loss(recon_batch, source_batch, reduction="mean")
        print("Reconstrcution: ", recon.item())
        print("SSIM: ", 1 - ssim.item())
        source_cthw = source_cthw.float().cpu()
        side_by_side = torch.cat([source_cthw, recon_cthw], dim=-1)

        stem = f"{idx:04d}_{Path(source_path).stem}"
        input_out = output_dir / f"{stem}_input.mp4"
        recon_out = output_dir / f"{stem}_recon.mp4"
        sbs_out = output_dir / f"{stem}_sbs.mp4"

        _write_video(input_out, source_cthw, fps=cfg.fps)
        _write_video(recon_out, recon_cthw, fps=cfg.fps)
        _write_video(sbs_out, side_by_side, fps=cfg.fps)

        rec = {
            "index": idx,
            "source_path": source_path,
            "caption": caption,
            "input_video": str(input_out),
            "reconstruction_video": str(recon_out),
            "side_by_side_video": str(sbs_out),
            "num_frames": int(source_cthw.shape[1]),
            "height": int(source_cthw.shape[2]),
            "width": int(source_cthw.shape[3]),
            "infer_window_size": int(cfg.num_frames),
            "infer_window_stride": int(cfg.window_stride) if cfg.window_stride is not None else int(cfg.num_frames),
        }
        records.append(rec)
        print(f"[infer] done {idx + 1}/{len(samples)} -> {sbs_out}")

    meta_path = output_dir / "reconstruction_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)
    print(f"[infer] wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
