# $\color{orange}{\textbf{{[CVPR 2026]}}}$ PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation

<div align="center">
  <a href="https://plan-lab.github.io/pyratok"><img src="https://img.shields.io/badge/Project-Website-blue?style=for-the-badge&logo=googlechrome"></a>
  <a href="https://arxiv.org/abs/2601.16210"><img src="https://img.shields.io/badge/arXiv-2601.16210-b31b1b.svg?style=for-the-badge"></a>
  <a href="https://github.com/PLAN-Lab/PyraTok"><img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github"></a>
  <a href="https://huggingface.co/onkarsus13/PyraTok"><img src="https://img.shields.io/badge/Model-HuggingFace-orange?style=for-the-badge&logo=huggingface"></a>
</div>

PyraTok is a language-aligned pyramidal video tokenizer designed for both video understanding and generation.  
This repository includes model code, inference scripts, and finetuning scripts with Accelerate.

<img width="1310" height="380" alt="pyratok_overview" src="https://github.com/user-attachments/assets/198f469b-d6ee-498b-9cfa-22594fbc9c1e" />


## Highlights
- Language-aligned pyramidal quantization (`LaPQ`) for semantically meaningful video tokens.
- End-to-end VAE-style reconstruction with optional text conditioning.
- Sliding-window inference over full videos at frame level.
- Finetuning pipeline with multi-GPU support via `accelerate`.

## Model Weights
- Hugging Face model page: `https://huggingface.co/onkarsus13/PyraTok`

Download with script:
```bash
python download.py
```

Or directly:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='onkarsus13/PyraTok', local_dir='./checkpoints', local_dir_use_symlinks=False)"
```

## Installation
### 1) Clone and enter the repo
```bash
git clone <your-repo-url>
cd CVPR25-PyraTok
```

### 2) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Format
Training/inference manifest supports `.json` or `.jsonl`.

Each row must contain:
- `path`: relative or absolute video path
- `caption`: text instruction/caption

Example `.json`:
```json
[
  {"path": "videos/clip_0001.mp4", "caption": "a person is walking in a park"},
  {"path": "videos/clip_0002.mp4", "caption": "two people talking indoors"}
]
```

Example `.jsonl`:
```json
{"path": "videos/clip_0001.mp4", "caption": "a person is walking in a park"}
{"path": "videos/clip_0002.mp4", "caption": "two people talking indoors"}
```

## Inference
Run with explicit inputs:
```bash
python infer.py /abs/path/to/video.mp4 "your prompt here"
```

Outputs are written under `./reconstructions/`:
- `*_input.mp4`
- `*_recon.mp4`
- `*_sbs.mp4`
- `reconstruction_metadata.json`

### Inference settings (edit in `infer.py`)
- `vae_model_path`: path to downloaded/finetuned VAE checkpoint.
- `qwen_model_path`: path to text encoder checkpoint.
- `use_text_condition`: enable/disable language conditioning.
- `num_frames`: sliding window size.
- `window_stride`: sliding step (`None` means no overlap, stride = `num_frames`).
- `height`, `width`, `fps`, `device`, `dtype`.

## Finetuning (Complete Workflow)
This project uses `finetune.py` as the Accelerate entrypoint, which calls `fine_tune.py`.

### 1) Prepare data
Update these paths in `fine_tune.py` (`TrainConfig`):
- `video_base_path`
- `fallback_video_base_paths`
- `train_manifest_path`
- `pyratok_pretrained_path`
- `qwen_model_path`
- `output_dir`

### 2) Configure training hyperparameters
In `fine_tune.py`, set:
- Core training: `num_epochs`, `max_steps`, `learning_rate`, `weight_decay`, `grad_clip_norm`
- Batch/layout: `batch_size`, `gradient_accumulation_steps`, `num_workers`, `num_frames`
- Precision/distributed: `mixed_precision`, `offload_text_encoder_to_cpu`
- LaPQ params: `lapq_num_codes`, `lapq_num_quantizers`, `lapq_codebook_dim`, etc.

### 3) Configure Accelerate (first run)
```bash
accelerate config
```

### 4) Launch training
Single command:
```bash
accelerate launch --mixed_precision bf16 finetune.py
```

Or use provided script:
```bash
bash train.sh
```

`train.sh` example:
```bash
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision bf16 finetune.py
```

### 5) Training outputs
Saved in `output_dir`:
- periodic checkpoints: `checkpoint_step_XXXXXXX.pt`
- diffusers-format VAE folders: `vae_checkpoint_step_XXXXXXX/`
- dumped config: `hardcoded_train_config.json`

## Notes and Troubleshooting
- Video decoding tries `torchvision`, then `imageio`, then `opencv`.
- Unreadable videos are skipped during dataset loading instead of crashing training.
- If you see missing decoder/backend errors, install both `torchvision` and `opencv-python` (or `imageio-ffmpeg`).
- `fine_tune.py` currently uses hardcoded config via `TrainConfig`, so editing that dataclass is the primary way to change runs.

## Contact

While installing or finetuning, if you find any issues, please contact to ```onkarsus13@gmail.com```.

## Citation
:star: If you find this work useful, please cite our [paper](https://arxiv.org/abs/2601.16210)

```bibtex
@inproceedings{susladkar2026pyratok,
  title={PyraTok: Language-Aligned Pyramidal Tokenizer for Video Understanding and Generation},
  author={Susladkar, Onkar and Prakash, Tushar and Juvekar, Adheesh and Nguyen, Kiet A. and Jang, Dong-Hwan and Dhillon, Inderjit S. and Lourentzou, Ismini},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
