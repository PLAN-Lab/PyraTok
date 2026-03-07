NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --mixed_precision bf16 finetune.py
