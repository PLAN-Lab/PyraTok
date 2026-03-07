import torch
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM


def tokenize_prompt(
    tokenizer: Qwen2TokenizerFast,
    prompt: str | list[str],
    max_sequence_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = [prompt] if isinstance(prompt, str) else prompt
    all_input_ids = []
    all_attention_masks = []

    for single_prompt in prompts:
        chat_template = getattr(tokenizer, "chat_template", None)
        if chat_template:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        else:
            # Fallback for tokenizers without a configured chat template.
            text = single_prompt
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )
        all_input_ids.append(tokens["input_ids"])
        all_attention_masks.append(tokens["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    return input_ids, attention_mask


def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    out_ids = []
    for _ in range(bsz):
        t = torch.arange(1)
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(seq_len)
        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)
    return torch.stack(out_ids)


def _get_qwen3_prompt_embeds(
    text_encoder: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None or dtype is None:
        ref_param = next(text_encoder.parameters())
        if device is None:
            device = ref_param.device
        if dtype is None:
            dtype = ref_param.dtype

    output = text_encoder(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        output_hidden_states=True,
        use_cache=False,
    )

    # Use only the final hidden state.
    prompt_embeds = output.hidden_states[-4].to(dtype=dtype, device=device)
    return prompt_embeds


def encode_prompt(
    text_encoder: Qwen3ForCausalLM,
    tokenizer: Qwen2TokenizerFast,
    prompt: str | list[str] | None,
    device: torch.device | None = None,
    prompt_embeds: torch.Tensor | None = None,
    max_sequence_length: int = 512,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prompt is None:
        prompt = ""

    if prompt_embeds is None:
        input_ids, attention_mask = tokenize_prompt(
            tokenizer=tokenizer,
            prompt=prompt,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = _get_qwen3_prompt_embeds(
            text_encoder=text_encoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            dtype=dtype,
            device=device,
        )

    if device is None:
        device = prompt_embeds.device

    
    return prompt_embeds
