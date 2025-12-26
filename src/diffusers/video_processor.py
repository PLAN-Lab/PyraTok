# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import List, Optional, Union

import numpy as np
import PIL
import torch

from .image_processor import VaeImageProcessor, is_valid_image, is_valid_image_imagelist
from .utils.loading_utils import load_video


class VideoProcessor(VaeImageProcessor):
    r"""Simple video processor."""

    _MAX_VIDEO_FRAMES = 104
    _FRAME_DIVISIBILITY = 8

    def _sample_video_frames(self, frames: List[PIL.Image.Image]) -> List[PIL.Image.Image]:
        """
        Uniformly sample frames so their count follows `k * _FRAME_DIVISIBILITY + 1`, capped by `_MAX_VIDEO_FRAMES`.
        With the defaults, the largest compatible count under the cap is 97 (= 12 * 8 + 1).
        """
        if len(frames) == 0:
            raise ValueError("Input video contains no frames.")

        capped_frames = min(len(frames), self._MAX_VIDEO_FRAMES)

        if capped_frames <= self._FRAME_DIVISIBILITY + 1:
            target = capped_frames
        else:
            # Enforce k * 8 + 1 structure (e.g., 97 when max is 104).
            target = ((capped_frames - 1) // self._FRAME_DIVISIBILITY) * self._FRAME_DIVISIBILITY + 1

        if target == len(frames):
            return frames

        indices = np.linspace(0, len(frames) - 1, num=target, dtype=int)
        return [frames[i] for i in indices]

    def preprocess_video(self, video, height: Optional[int] = None, width: Optional[int] = None) -> torch.Tensor:
        r"""
        Preprocesses input video(s).

        Args:
            video (`List[PIL.Image]`, `List[List[PIL.Image]]`, `torch.Tensor`, `np.array`, `List[torch.Tensor]`, `List[np.array]`):
                The input video. It can be one of the following:
                * Path or URL to a video file. Frames will be auto-loaded and uniformly sampled (max 104, or the
                  largest `k * 8 + 1` count under that cap, i.e., 97 by default).
                * List of the PIL images.
                * List of list of PIL images.
                * 4D Torch tensors (expected shape for each tensor `(num_frames, num_channels, height, width)`).
                * 4D NumPy arrays (expected shape for each array `(num_frames, height, width, num_channels)`).
                * List of 4D Torch tensors (expected shape for each tensor `(num_frames, num_channels, height,
                  width)`).
                * List of 4D NumPy arrays (expected shape for each array `(num_frames, height, width, num_channels)`).
                * 5D NumPy arrays: expected shape for each array `(batch_size, num_frames, height, width,
                  num_channels)`.
                * 5D Torch tensors: expected shape for each array `(batch_size, num_frames, num_channels, height,
                  width)`.
            height (`int`, *optional*, defaults to `None`):
                The height in preprocessed frames of the video. If `None`, will use the `get_default_height_width()` to
                get default height.
            width (`int`, *optional*`, defaults to `None`):
                The width in preprocessed frames of the video. If `None`, will use get_default_height_width()` to get
                the default width.
        """
        should_sample = isinstance(video, str)
        if should_sample:
            video = load_video(video)

        if isinstance(video, list) and isinstance(video[0], np.ndarray) and video[0].ndim == 5:
            warnings.warn(
                "Passing `video` as a list of 5d np.ndarray is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d np.ndarray",
                FutureWarning,
            )
            video = np.concatenate(video, axis=0)
        if isinstance(video, list) and isinstance(video[0], torch.Tensor) and video[0].ndim == 5:
            warnings.warn(
                "Passing `video` as a list of 5d torch.Tensor is deprecated."
                "Please concatenate the list along the batch dimension and pass it as a single 5d torch.Tensor",
                FutureWarning,
            )
            video = torch.cat(video, axis=0)

        # ensure the input is a list of videos:
        # - if it is a batch of videos (5d torch.Tensor or np.ndarray), it is converted to a list of videos (a list of 4d torch.Tensor or np.ndarray)
        # - if it is a single video, it is converted to a list of one video.
        if isinstance(video, (np.ndarray, torch.Tensor)) and video.ndim == 5:
            video = list(video)
        elif isinstance(video, list) and is_valid_image(video[0]) or is_valid_image_imagelist(video):
            video = [video]
        elif isinstance(video, list) and is_valid_image_imagelist(video[0]):
            video = video
        else:
            raise ValueError(
                "Input is in incorrect format. Currently, we only support numpy.ndarray, torch.Tensor, PIL.Image.Image"
            )

        if should_sample:
            video = [self._sample_video_frames(frames) for frames in video]

        video = torch.stack([self.preprocess(img, height=height, width=width) for img in video], dim=0)

        # move the number of channels before the number of frames.
        video = video.permute(0, 2, 1, 3, 4)

        return video

    def postprocess_video(
        self, video: torch.Tensor, output_type: str = "np"
    ) -> Union[np.ndarray, torch.Tensor, List[PIL.Image.Image]]:
        r"""
        Converts a video tensor to a list of frames for export.

        Args:
            video (`torch.Tensor`): The video as a tensor.
            output_type (`str`, defaults to `"np"`): Output type of the postprocessed `video` tensor.
        """
        batch_size = video.shape[0]
        outputs = []
        for batch_idx in range(batch_size):
            batch_vid = video[batch_idx].permute(1, 0, 2, 3)
            batch_output = self.postprocess(batch_vid, output_type)
            outputs.append(batch_output)

        if output_type == "np":
            outputs = np.stack(outputs)
        elif output_type == "pt":
            outputs = torch.stack(outputs)
        elif not output_type == "pil":
            raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

        return outputs
