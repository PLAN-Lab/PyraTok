from huggingface_hub import snapshot_download

model_id = "onkarsus13/PyraTok"  # replace with your model, e.g. "meta-llama/Llama-2-7b-hf"
target_dir = "/data/onkar/models_pyratok"

snapshot_download(
    repo_id=model_id,
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # store actual files, not symlinks
    # folder="vae"
)
print(f"Downloaded to: {target_dir}")