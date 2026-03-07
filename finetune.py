"""
Accelerate entrypoint for language-guided PyraTok training.

Launch:
    accelerate launch --mixed_precision bf16 /data/home/onkar/PyraTok/finetune.py
"""

try:
    from .fine_tune import train
except ImportError:
    from fine_tune import train


if __name__ == "__main__":
    train()

