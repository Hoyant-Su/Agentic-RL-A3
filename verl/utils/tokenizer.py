
import warnings
from verl.utils.bash_coding import BASH_CODING_TAG_TOKENS, is_bash_coding_enabled

__all__ = [
    "hf_tokenizer",
    "hf_processor",
    "register_bash_coding_tag_tokens",
    "resize_model_embeddings_to_tokenizer",
]

def set_pad_token_id(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(
            f"tokenizer.pad_token_id is None. Now set to {tokenizer.eos_token_id}",
            stacklevel=1,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(
            f"tokenizer.pad_token is None. Now set to {tokenizer.eos_token}",
            stacklevel=1,
        )

def register_bash_coding_tag_tokens(tokenizer):
    if not is_bash_coding_enabled():
        return 0
    additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    missing = [token for token in BASH_CODING_TAG_TOKENS if token not in additional]
    if not missing:
        return 0
    return int(tokenizer.add_special_tokens({"additional_special_tokens": missing}))

def resize_model_embeddings_to_tokenizer(model, tokenizer):
    if not is_bash_coding_enabled():
        return 0
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        return 0
    current_size = int(embeddings.num_embeddings)
    target_size = int(len(tokenizer))
    if target_size <= current_size:
        return 0
    model.resize_token_embeddings(target_size)
    return target_size - current_size

def hf_tokenizer(name_or_path, correct_pad_token=True, correct_gemma2=True, **kwargs):
    from transformers import AutoTokenizer

    if (
        correct_gemma2
        and isinstance(name_or_path, str)
        and "gemma-2-2b-it" in name_or_path
    ):
        warnings.warn(
            "Found gemma-2-2b-it tokenizer. Set eos_token and eos_token_id to <end_of_turn> and 107.",
            stacklevel=1,
        )
        kwargs["eos_token"] = "<end_of_turn>"
        kwargs["eos_token_id"] = 107
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    register_bash_coding_tag_tokens(tokenizer)
    if correct_pad_token:
        set_pad_token_id(tokenizer)
    return tokenizer

def hf_processor(name_or_path, **kwargs):
    from transformers import AutoProcessor

    try:
        processor = AutoProcessor.from_pretrained(name_or_path, **kwargs)
    except Exception:
        processor = None
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None
    return processor
