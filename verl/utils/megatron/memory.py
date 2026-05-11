import torch

class MemoryBuffer:
    def __init__(self, numel, numel_padded, dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(
            self.numel_padded,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

    def zero(self):
        self.data.zero_()

    def get(self, shape, start_index):
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor
