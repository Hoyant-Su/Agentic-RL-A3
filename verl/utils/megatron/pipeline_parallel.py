import torch
from megatron.core import parallel_state as mpu
from .sequence_parallel import pad_to_sequence_parallel

def compute_transformers_input_shapes(batches, meta_info):
    from flash_attn.bert_padding import unpad_input

    input_shapes = []
    for model_inputs in batches:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        input_ids_rmpad = unpad_input(input_ids.unsqueeze(dim=-1), attention_mask)[0]
        if meta_info["sequence_parallel"]:
            input_ids_rmpad = pad_to_sequence_parallel(input_ids_rmpad)
            input_shapes.append(
                torch.Size(
                    [
                        input_ids_rmpad.shape[0]
                        // mpu.get_tensor_model_parallel_world_size(),
                        1,
                        meta_info["hidden_size"],
                    ]
                )
            )
        else:
            input_shapes.append(
                torch.Size([input_ids_rmpad.shape[0], 1, meta_info["hidden_size"]])
            )
    return input_shapes

def make_batch_generator(batches, vpp_size):
    if vpp_size > 1:
        batch_generator = [batches] * vpp_size
        batch_generator = [iter(b) for b in batch_generator]
    else:
        batch_generator = iter(batches)
    return batch_generator
