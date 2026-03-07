import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision.io import read_video as tv_read_video

    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

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


@dataclass(frozen=True)
class DataConfig:
    video_base_path: str
    fallback_video_base_paths: tuple[str, ...]
    manifest_path: str
    num_frames: int
    height: int | None
    width: int | None
    batch_size: int
    num_workers: int
    shuffle: bool
    pin_memory: bool
    drop_last: bool
    verify_video_exists: bool = False


def _resolve_path(video_base_path: str, path: str, fallback_video_base_paths: tuple[str, ...] = ()) -> str:
    if os.path.isabs(path):
        return path
    search_roots = (video_base_path, *fallback_video_base_paths)
    tried = [str(Path(root) / path) for root in search_roots]
    for candidate in tried:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Video file not found for relative path '{path}'. Tried: {tried}")


def _read_video_tchw(path: str) -> torch.Tensor:
    errors: list[str] = []

    if _HAS_TORCHVISION:
        try:
            video, _, _ = tv_read_video(path, pts_unit="sec", output_format="TCHW")
            return video
        except Exception as exc:
            errors.append(f"torchvision: {exc}")

    if _HAS_IMAGEIO:
        try:
            frames = []
            for frame in iio.imiter(path):
                t = torch.from_numpy(frame)
                if t.ndim == 2:
                    t = t.unsqueeze(-1).repeat(1, 1, 3)
                frames.append(t)

            if not frames:
                raise RuntimeError(f"No frames found in video: {path}")

            video_thwc = torch.stack(frames, dim=0)
            video_tchw = video_thwc.permute(0, 3, 1, 2).contiguous()
            return video_tchw
        except Exception as exc:
            errors.append(f"imageio: {exc}")

    if _HAS_CV2:
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video with OpenCV: {path}")

            frames = []
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame_rgb))
            cap.release()

            if not frames:
                raise RuntimeError(f"No frames found in video: {path}")

            video_thwc = torch.stack(frames, dim=0)
            video_tchw = video_thwc.permute(0, 3, 1, 2).contiguous()
            return video_tchw
        except Exception as exc:
            errors.append(f"cv2: {exc}")

    if not (_HAS_TORCHVISION or _HAS_IMAGEIO or _HAS_CV2):
        raise ImportError("Need torchvision, imageio, or opencv-python (cv2) to load videos.")
    raise RuntimeError(f"Failed to decode video '{path}'. Backends tried: {' | '.join(errors)}")


def _get_target_hw(
    in_height: int,
    in_width: int,
    height: int | None,
    width: int | None,
) -> tuple[int, int]:

    # Orientation-aware default resize requested by user:
    # landscape -> 256x512, portrait/square -> 512x256.
    if in_width > in_height:
        return 256, 512
    return 512, 256


def _resize_video(video_tchw: torch.Tensor, height: int | None, width: int | None) -> torch.Tensor:
    if video_tchw.dtype != torch.float32:
        video_tchw = video_tchw.float()
    _, _, in_height, in_width = video_tchw.shape
    # print(in_height, in_width)
    target_h, target_w = _get_target_hw(
        in_height=in_height,
        in_width=in_width,
        height=height,
        width=width,
    )
    # print(target_h, target_w)
    video_tchw = F.interpolate(video_tchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return video_tchw


def _sample_frames(video_tchw: torch.Tensor, num_frames: int) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError(f"num_frames must be > 0, got {num_frames}.")

    total_frames = int(video_tchw.shape[0])
    if total_frames <= 0:
        raise RuntimeError("Video has zero frames.")

    if total_frames >= num_frames:
        # Randomly choose a contiguous temporal chunk of size num_frames.
        max_start = total_frames - num_frames
        if max_start > 0:
            start = int(torch.randint(0, max_start + 1, size=(1,)).item())
        else:
            start = 0
        indices = torch.arange(start, start + num_frames, dtype=torch.long)
    else:
        # Keep all available frames, then pad by repeating the last frame.
        pad = torch.full((num_frames - total_frames,), total_frames - 1, dtype=torch.long)
        indices = torch.cat([torch.arange(total_frames, dtype=torch.long), pad], dim=0)

    return video_tchw.index_select(0, indices)


def load_video_as_cthw(path: str, num_frames: int, height: int | None, width: int | None) -> torch.Tensor:
    video_tchw = _read_video_tchw(path)
    video_tchw = _sample_frames(video_tchw, num_frames=num_frames)
    video_tchw = _resize_video(video_tchw, height=height, width=width)
    video_tchw = video_tchw / 255.0
    video_tchw = video_tchw * 2.0 - 1.0
    video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()
    return video_cthw


def _load_manifest(manifest_path: str) -> list[dict[str, Any]]:
    if manifest_path.endswith(".jsonl"):
        rows = []
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
            raise ValueError("JSON manifest must be a list of samples.")
        return data

    raise ValueError(f"Unsupported manifest format: {manifest_path}")


class PyraTokLanguageDataset(Dataset):
    """
    Ditto caption manifest schema (source_video_captions_sorted.json):
    - path: relative path under video_base_path
    - caption: language instruction/caption string
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.samples = _load_manifest(cfg.manifest_path)
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in manifest: {cfg.manifest_path}")

        required_keys = {"path", "caption"}
        first = self.samples[0]
        missing = required_keys - set(first.keys())
        if missing:
            raise KeyError(
                f"Manifest rows must include keys {sorted(required_keys)}. Missing keys in first row: {sorted(missing)}"
            )

        if cfg.verify_video_exists:
            kept = []
            for row in self.samples:
                try:
                    _ = _resolve_path(
                        self.cfg.video_base_path,
                        row["path"],
                        fallback_video_base_paths=self.cfg.fallback_video_base_paths,
                    )
                    kept.append(row)
                except FileNotFoundError:
                    pass
            self.samples = kept
            if len(self.samples) == 0:
                raise RuntimeError("No valid videos found after path verification.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        total = len(self.samples)
        for offset in range(total):
            sample = self.samples[(idx + offset) % total]
            source_rel = sample["path"]
            instruction = sample.get("caption", "")
            try:
                source_path = _resolve_path(
                    self.cfg.video_base_path,
                    source_rel,
                    fallback_video_base_paths=self.cfg.fallback_video_base_paths,
                )
                target_path = source_path

                source_video = load_video_as_cthw(
                    source_path,
                    num_frames=self.cfg.num_frames,
                    height=self.cfg.height,
                    width=self.cfg.width,
                )
                # VAE reconstruction training: source and target are identical.
                target_video = source_video.clone()

                return {
                    "source_video": source_video,
                    "target_video": target_video,
                    "instruction": instruction,
                    "source_path": source_path,
                    "target_path": target_path,
                    "spatial_size": (int(source_video.shape[-2]), int(source_video.shape[-1])),
                }
            except Exception as exc:
                print(f"[PyraTokLanguageDataset] Skipping unreadable video '{source_rel}': {exc}")

        raise RuntimeError("No readable videos available in dataset.")


def _pad_video_to_hw(video_cthw: torch.Tensor, out_h: int, out_w: int, pad_value: float = -1.0) -> torch.Tensor:
    _, _, h, w = video_cthw.shape
    pad_h = out_h - h
    pad_w = out_w - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Invalid pad target {(out_h, out_w)} for video shape {tuple(video_cthw.shape)}.")
    if pad_h == 0 and pad_w == 0:
        return video_cthw
    return F.pad(video_cthw, (0, pad_w, 0, pad_h), value=pad_value)


def pyratok_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    heights = [item["source_video"].shape[-2] for item in batch]
    widths = [item["source_video"].shape[-1] for item in batch]
    max_h = max(heights)
    max_w = max(widths)

    source_video = torch.stack([_pad_video_to_hw(item["source_video"], max_h, max_w) for item in batch], dim=0)
    target_video = torch.stack([_pad_video_to_hw(item["target_video"], max_h, max_w) for item in batch], dim=0)
    instructions = [item["instruction"] for item in batch]
    source_paths = [item["source_path"] for item in batch]
    target_paths = [item["target_path"] for item in batch]
    spatial_sizes = [item["spatial_size"] for item in batch]

    return {
        "source_video": source_video,
        "target_video": target_video,
        "instructions": instructions,
        "source_paths": source_paths,
        "target_paths": target_paths,
        "spatial_sizes": spatial_sizes,
    }


def build_train_dataloader(cfg: DataConfig) -> DataLoader:
    dataset = PyraTokLanguageDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=pyratok_collate_fn,
    )
