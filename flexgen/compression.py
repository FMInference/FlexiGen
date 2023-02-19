import dataclasses

import torch
import numpy as np

from flexgen.pytorch_backend import (TorchTensor, TorchDevice,
    DeviceType, general_copy, fix_recursive_import)
from flexgen.utils import np_dtype_to_torch_dtype


@dataclasses.dataclass
class CompressionConfig:
    """Group-wise quantization."""
    num_bits: int
    group_size: int
    group_dim: int
    symmetric: bool
    enabled: bool = True


class TorchCompressedDevice:
    """Manage tensors stored in a compressed format."""

    def __init__(self, base_device):
        self.name = "compressed"
        self.device_type = DeviceType.COMPRESSED
        self.base_device = base_device

        self.data_decompress_workspace = None
        self.workspace_pt = 0

    def allocate(self, shape, dtype, comp_config, pin_memory=None, name=None):
        """Allocate a compressed TorchTensor. Round up the shape to group boundary."""
        assert comp_config.num_bits == 4 and dtype == np.float16

        group_size, group_dim = comp_config.group_size, comp_config.group_dim

        # Round up
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])

        data = self.base_device.allocate(data_shape, np.uint8, pin_memory=pin_memory)
        scale = self.base_device.allocate(scale_shape, np.float16, pin_memory=pin_memory)

        return TorchTensor(shape, np_dtype_to_torch_dtype[dtype],
                           (data, scale, comp_config), self, name=name)

    def init_cache_one_gpu_batch(self, config, task, policy):
        num_head, hidden_size, prompt_len, gen_len, gpu_batch_size = (
            config.n_head, config.input_dim, task.prompt_len, task.gen_len,
            policy.gpu_batch_size)
        shape = (prompt_len + gen_len - 1, gpu_batch_size * num_head, hidden_size // num_head)
        # NOTE: disable pin_memory due to high memory overhead
        pin_memory = False
        k_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        v_cache = self.allocate(shape, np.float16,
            comp_config=policy.comp_cache_config, pin_memory=pin_memory)
        return k_cache, v_cache

    def init_attention_compute_workspace(self, config, task, policy):
        if self.base_device.device_type != DeviceType.CPU:
            return  # Only CPU requires this fp32 workspace

        b = policy.gpu_batch_size
        n_head = config.n_head
        head_dim = config.input_dim // n_head
        max_seq_len = task.prompt_len + task.gen_len - 1
        shape = (max_seq_len, b * n_head, head_dim)

        group_size, group_dim = (
            policy.comp_cache_config.group_size, policy.comp_cache_config.group_dim)
        num_groups = (shape[group_dim] + group_size - 1) // group_size
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])

        self.data_decompress_workspace = [
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
            torch.empty(*new_shape, dtype=torch.float32,
                device=self.base_device.dev),
        ]

    def compress(self, tensor, comp_config):
        """Compress a torch.Tensor. Round up the shape to group boundary."""
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)
        assert num_bits == 4 and group_size % 2 == 0 and not symmetric

        if tensor.device.type == "cpu" and tensor.dtype == torch.float16:
            tensor = tensor.float()

        shape = tensor.shape
        num_groups = (shape[group_dim] + group_size - 1) // group_size

        # Pad
        new_shape = (shape[:group_dim] + (num_groups, group_size) +
                     shape[group_dim+1:])
        pad_len = (group_size - shape[group_dim] % group_size) % group_size
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            tensor = torch.cat([
                tensor,
                torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
                dim=group_dim)
        data = tensor.view(new_shape)

        # Quantize
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)
        data = data.clamp_(0, B).round_().to(torch.uint8)

        # Pack
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))
        data = torch.bitwise_or(
            data[left_indices].bitwise_left_shift(4), data[right_indices])

        # Reshape
        data_shape = (
            shape[:group_dim] + (num_groups * (group_size // 2),) + shape[group_dim+1:])
        scale_shape = (
            shape[:group_dim] + (num_groups, 2) + shape[group_dim+1:])
        data = data.view(data_shape)
        scale = torch.cat([scale, mn], dim=group_dim+1).view(scale_shape)

        data = TorchTensor.create_from_torch(data, self.base_device)
        scale = TorchTensor.create_from_torch(scale, self.base_device)

        return TorchTensor(shape, tensor.dtype,
                           (data, scale, comp_config), self)

    def decompress(self, tensor):
        data, scale, comp_config = tensor.data
        group_size, num_bits, group_dim, symmetric = (
            comp_config.group_size, comp_config.num_bits,
            comp_config.group_dim, comp_config.symmetric)

        group_size_c = group_size // 2
        shape = data.shape
        num_groups = (shape[group_dim] + group_size_c - 1) // group_size_c

        # Pad
        new_shape = (shape[:group_dim] + (num_groups, group_size_c) +
                     shape[group_dim+1:])
        pad_len = (group_size_c - shape[group_dim] % group_size_c) % group_size_c
        if pad_len != 0:
            pad_shape = shape[:group_dim] + (pad_len,) + shape[group_dim+1:]
            data = torch.cat([
                data,
                torch.zeros(pad_shape, dtype=data.dtype, device=data.device)],
                dim=group_dim)
        packed = data.data.view(new_shape)

        # Unpack
        if self.base_device.device_type == DeviceType.CPU:
            self.workspace_pt = (self.workspace_pt + 1) % len(
                self.data_decompress_workspace)
            data = self.data_decompress_workspace[
                self.workspace_pt][:shape[0]]
        else:
            new_shape = (shape[:group_dim] + (num_groups, group_size,) +
                         shape[group_dim+1:])
            data = torch.empty(new_shape, dtype=torch.float16, device=packed.device)
        left_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(0, data.shape[group_dim+1], 2),))
        right_indices = (
            tuple(slice(0, x) for x in data.shape[:group_dim+1]) +
            (slice(1, data.shape[group_dim+1], 2),))
        data[left_indices] = packed.bitwise_right_shift(4)
        data[right_indices] = packed.bitwise_and(0xF)

        # Dequantize
        scale, mn = scale.data.split(1, dim=group_dim + 1)
        data.div_(scale)
        data.add_(mn)

        # Reshape
        unpad_len = (group_size - tensor.shape[group_dim] % group_size) % group_size
        if unpad_len != 0:
            flatten_shape = (shape[:group_dim] + (num_groups * group_size,) +
                             shape[group_dim+1:])
            indices = [slice(0, x) for x in flatten_shape]
            indices[group_dim] = slice(0, flatten_shape[group_dim] - unpad_len)
            data = data.view(flatten_shape)[indices].contiguous()

        return data.view(tensor.shape)


def general_copy_compressed(dst, dst_indices, src, src_indices):
    assert (src.device.device_type == DeviceType.COMPRESSED and
            dst.device.device_type == DeviceType.COMPRESSED)

    src_data_indices, src_scale_indices = get_compressed_indices(
        src, src_indices, src.shape)

    dst_data_indices, dst_scale_indices = get_compressed_indices(
        dst, dst_indices, dst.shape)

    general_copy(dst.data[0], dst_data_indices, src.data[0], src_data_indices)
    general_copy(dst.data[1], dst_scale_indices, src.data[1], src_scale_indices)


def get_compressed_indices(tensor, indices, shape):
    comp_config = tensor.data[2]
    group_size, group_dim = comp_config.group_size, comp_config.group_dim
    assert comp_config.num_bits == 4

    if indices is None:
        indices = list(slice(0, x) for x in shape[:group_dim+1])
    else:
        indices = list(indices) + [slice(0, x) for x in shape[len(indices):]]
    assert indices[group_dim].start % group_size == 0

    data_indices = list(indices)
    data_indices[group_dim] = slice(
        indices[group_dim].start // 2, (indices[group_dim].stop + 1) // 2)

    scale_indices = indices
    scale_indices.insert(group_dim+1, slice(0, 2))
    scale_indices[group_dim] = slice(
        indices[group_dim].start // group_size,
        (indices[group_dim].stop + group_size - 1) // group_size)

    return data_indices, scale_indices


default_cache_config = CompressionConfig(
    num_bits=0, group_size=0, group_dim=0, symmetric=False, enabled=False)


def set_cache_compression_config(config):
    global default_cache_config
    default_cache_config = config


def get_cache_compression_config():
    return default_cache_config


def compress(tensor, config):
    """Simulate group-wise quantization."""
    if not config.enabled:
        return tensor

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)
    assert num_bits <= 8

    original_shape = tensor.shape
    num_groups = (original_shape[group_dim] + group_size - 1) // group_size
    new_shape = (original_shape[:group_dim] + (num_groups, group_size) +
                 original_shape[group_dim+1:])

    # Pad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len != 0:
        pad_shape = original_shape[:group_dim] + (pad_len,) + original_shape[group_dim+1:]
        tensor = torch.cat([
            tensor,
            torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)],
            dim=group_dim)
    data = tensor.view(new_shape)

    # Quantize
    if symmetric:
        B = 2 ** (num_bits - 1) - 1
        scale = B / torch.max(data.abs(), dim=group_dim + 1, keepdim=True)[0]
        data = data * scale
        data = data.clamp_(-B, B).round_().to(torch.int8)
        return data, scale, original_shape
    else:
        B = 2 ** num_bits - 1
        mn = torch.min(data, dim=group_dim + 1, keepdim=True)[0]
        mx = torch.max(data, dim=group_dim + 1, keepdim=True)[0]

        scale = B / (mx - mn)
        data = data - mn
        data.mul_(scale)

        data = data.clamp_(0, B).round_().to(torch.uint8)
        return data, mn, scale, original_shape


def decompress(packed_data, config):
    """Simulate group-wise dequantization."""
    if not config.enabled:
        return packed_data

    group_size, num_bits, group_dim, symmetric = (
        config.group_size, config.num_bits, config.group_dim, config.symmetric)

    # Dequantize
    if symmetric:
        data, scale, original_shape = packed_data
        data = data / scale
    else:
        data, mn, scale, original_shape = packed_data
        data = data / scale
        data.add_(mn)

    # Unpad
    pad_len = (group_size - original_shape[group_dim] % group_size) % group_size
    if pad_len:
        padded_original_shape = (
            original_shape[:group_dim] +
            (original_shape[group_dim] + pad_len,) +
            original_shape[group_dim+1:])
        data = data.reshape(padded_original_shape)
        indices = [slice(0, x) for x in original_shape]
        return data[indices].contiguous()
    else:
        return data.view(original_shape)


def compress_and_decompress(tensor, config):
    packed_data = compress(tensor, config)
    return decompress(packed_data, config)


def test_simulated_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (64, 64, 64), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed_data = compress(a, config)
    b = decompress(packed_data, config)
    print(a[0])
    print(b[0])


def test_real_compression():
    torch.manual_seed(0)
    a = torch.normal(0, 1, (32, 1, 1), dtype=torch.float16).cuda()

    config = CompressionConfig(
        num_bits=4, group_size=32, group_dim=0, symmetric=False)
    dev = TorchDevice("cuda:0", 0, 0).compressed_device
    packed = dev.compress(a, config)
    b = dev.decompress(packed)

    print(a.flatten())
    print(b.flatten())


if __name__ == "__main__":
    fix_recursive_import()
    #test_simulated_compression()
    test_real_compression()
