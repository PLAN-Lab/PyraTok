# Simple-PyTorch-LoRA

_A Light-weight, Library-agnostic LoRA Implementation for PyTorch_  
Support for `Linear`, `Conv1d`, `Conv2d`, and `Conv3d` layers. Compatible with any PyTorch-based library (e.g., `torchvision`, `timm`, `transformers`).

---

## 📦 Features

- **Seamless Integration**: Plug-and-play LoRA adapters for existing models without modifying base weights.
- **Broad Coverage**: Works on linear layers and 1D/2D/3D convolutions.
- **Zero-Dependency**: Depends only on `torch` and standard Python libs.
- **Configurable**: Easily adjust rank (`r`), scaling factor (`alpha`), dropout, and target module patterns.
- **Merge for Inference**: Merge learned adapters into base weights for fast, standard inference.

---

## 🚀 Installation

Install via `pip`:

```bash
pip install simple-pytorch-lora
```

_or install from source:_

```bash
git clone https://github.com/YourUsername/Simple-PyTorch-LoRA.git
cd Simple-PyTorch-LoRA
pip install -e .
```

---

## 💡 Quick Start

```python
import torch
from simple_pytorch_lora import LoRAConfig, apply_lora, merge_lora
from transformers import AutoModelForCausalLM

# Load a pretrained model
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Configure LoRA
config = LoRAConfig(
    r=8,              # low-rank
    alpha=32,         # scaling
    dropout=0.1,      # optional adapter dropout
    target_modules=["attn", "conv"]  # apply on attention & convs
)

# Inject LoRA adapters
apply_lora(model, config)

# Train only LoRA parameters...
# optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# After training, merge for inference:
merge_lora(model)

# Save or export
model.save_pretrained("./lora_merged_model")
```

---

## 📚 Examples

- **Vision**: Accelerate fine-tuning of ResNet/VGG in `torchvision` by injecting LoRA into all conv layers.
- **Text**: Fine-tune large language models (GPT, BERT, T5) via LoRA on `transformers`.
- **Audio**: Adapt 1D convolutional front-ends for speech tasks with minimal new parameters.

See [examples/](./examples) for Jupyter notebooks and scripts.

---

## ⚙️ API Reference

### `LoRAConfig`  
Configuration dataclass:

| Field           | Type         | Default | Description                                |
| --------------- | ------------ | ------- | ------------------------------------------ |
| `r`             | `int`        | 4       | Rank of low-rank decomposition             |
| `alpha`         | `int`        | 16      | Scaling factor                             |
| `dropout`       | `float`      | 0.0     | Dropout probability for adapter inputs     |
| `target_modules`| `List[str]`  | None    | Substrings to match module names (e.g., `q_proj`, `conv`) |

### `apply_lora(model, config)`  
Recursively wraps target layers in LoRA modules.  

**Args**:
- `model` (`nn.Module`): PyTorch model to adapt.
- `config` (`LoRAConfig`): Adapter configuration.

### `merge_lora(model)`  
Merges adapter weights back into the original modules for inference.

**Args**:
- `model` (`nn.Module`): Model with active LoRA wrappers.

---

## 🤝 Contributing

We welcome contributions!  

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Make your changes & add tests/examples  
4. Open a Pull Request  

Please follow the [Code of Conduct](./CODE_OF_CONDUCT.md) and [Contributing Guidelines](./CONTRIBUTING.md).

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).

---

## 📢 Roadmap

- [ ] Support for LoKR (Low-Kernel Rank) layers
- [ ] Official integration with `timm` and `torchmetrics`
- [ ] Additional tutorials and benchmark scripts

---

> Made with ❤️ by [Your Name](https://github.com/YourUsername)
