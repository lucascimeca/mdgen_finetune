from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask

import numpy as np
import torch
import torch.nn.init as init
import random

import torch.nn
import torch.nn.functional as F
import torch.nn as nn

from huggingface_hub import hf_hub_download, HfApi

from rtb_utils.simple_io import save_dict_to_file, load_dict_from_file


def get_batch(name, seqres, tps, no_frames, data_dir, suffix):
    arr = np.lib.format.open_memmap(f'{data_dir}/{name}{suffix}.npy', 'r')

    if not tps:  # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)

    if no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }

    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask,  # (L,)
    }


def create_batches(ids, batch_size):
    for i in range(0, len(ids), batch_size):
        yield ids[i:i + batch_size]


def safe_reinit(module):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
    else:
        for param in module.parameters(recurse=False):
            if param.dim() >= 2:
                init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif param.dim() == 1:
                init.zeros_(param)
    for child in module.children():
        safe_reinit(child)




def seed_experiment(seed):
    """Set the seed for reproducibility in PyTorch runs.

    Args:
        seed (int): The seed number.
    """
    # Set the seed for Python's 'random' module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If using CUDA (PyTorch with GPU support)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def maybe_detach(x, t, times_to_detach):
    return x.detach() if t in times_to_detach else x


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(T.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def print_gpu_memory(total_memory=None, memory_allocated=None, memory_free=None):
    current_memory_free, current_total_memory = torch.cuda.mem_get_info()
    current_memory_allocated = current_total_memory - current_memory_free

    if total_memory is None or memory_allocated is None or memory_free is None:
        print()
        print(f"Total memory: {current_total_memory / (1024 ** 3):.4} GB")
        print(f"Used memory: {current_memory_allocated / (1024 ** 3):.4f} GB")
        print(f"Free memory: {current_memory_free / (1024 ** 3):.4f} GB")

    else:
        print(f"Total memory change: {(current_total_memory - total_memory) / (1024 ** 3):.4f} GB")
        print(f"Used memory change: {(current_memory_allocated - memory_allocated) / (1024 ** 3):.4f} GB")
        print(f"Free memory change: {(current_memory_free - memory_free) / (1024 ** 3):.4f} GB")
    return current_total_memory, current_memory_allocated, current_memory_free


def get_gpu_memory():
    current_total_memory = T.cuda.get_device_properties(0).total_memory
    current_memory_allocated = T.cuda.memory_allocated()
    current_memory_free = T.cuda.memory_reserved() - T.cuda.memory_allocated()
    return current_total_memory, current_memory_allocated, current_memory_free


class NoContext:
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def check_gradients(model):
    # Initialize a list to store the gradients
    gradients = []

    # Extract the gradients for each parameter and store them in the list
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(torch.abs(param.grad).view(-1))  # Flatten the gradient tensor

    if len(gradients) > 0:
        # Concatenate all gradients into a single tensor
        all_gradients = torch.cat(gradients)

        # Compute the mean of the gradients
        return torch.mean(all_gradients)

    else:
        return 0.


def check_model_exists(model_name):
    """Check if the model exists in the Hugging Face model hub."""
    api = HfApi()
    try:
        model_info = api.model_info(model_name)
        print(f"Model '{model_name}' exists on Hugging Face Hub.")
        return True
    except Exception as e:
        print(f"Model '{model_name}' does not exist or has been deleted. Error: {e}")
        return False


def load_model_if_exists(model_class, model_name):
    """Load the model only if it exists in the Hugging Face model hub."""
    if check_model_exists(model_name):
        model = model_class.from_pretrained(model_name)
        return model
    else:
        raise ValueError(f"Model '{model_name}' does not exist on Hugging Face Hub.")


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True
    return model



class Logger:

    def __init__(self, args=None):
        self.args = args
        self.logs = {}

    def log(self, results):
        for key, item in results.items():
            if key not in self.logs.keys():
                self.logs[key] = []
            else:
                if isinstance(results[key], torch.Tensor):
                    self.logs[key].append(results[key].cpu().mean().item())
                else:
                    self.logs[key].append(results[key])
    def print(self, it=None):
        it = it if it is not None else ''

        txt = f"it {it}: "
        parameters = []
        for key, value in self.logs.items():
            if len(value) > 0 and isinstance(value[0], dict):
                ks = {}
                for i in range(min(10, len(value))):
                    for k, v in value[-i].items():
                        if k not in ks.keys():
                            ks[k] = []
                        ks[k].append(v)
                for k in ks.keys():
                    parameters.append(f"{key}_{k}: {np.mean(ks[k][-10:]):.4f}")
            else:
                parameters.append(f"{key}: {np.mean(value[-10:]):.4f}")
        txt += ", ".join(parameters)
        print(txt)
        try:
            print(f"it {it}: " + ", ".join([f"{key}: {np.mean(value[-10:]):.4f}" for key, value in self.logs.items()]))
        except Exception as e:
            print(f"it {it}: " + ", ".join([f"{key}: {value[-10:]}" for key, value in self.logs.items()]))
            print(e)

    def save(self):
        save_dict_to_file(data=self.logs, path=self.args.save_folder, filename='run_logs', format='json', replace=True)

    def save_args(self):
        save_dict_to_file(data=self.args.__dict__, path=self.args.save_folder, filename='run_args', format='json', replace=True)

    def load(self, path):
        self.args = load_dict_from_file(f"{path}/run_args.json")
        self.logs = load_dict_from_file(f"{path}/run_logs.json")


def cycle(dl):
    while True:
        for data in dl:
            yield data


def flatten_logs(d: dict, parent_keys=None) -> dict:
    """
    Flattens a nested dict so that for each leaf value the key becomes:
      parent1.parent2/.../parentN/leaf
    where the join between parent keys is '.', and the final separator before
    the leaf key is '/'.

    Examples:
        {"loss": 0.5}
          → {"loss": 0.5}
        {"kpis": {"k1": 1.0, "k2": 2.0}}
          → {"kpis/k1": 1.0, "kpis/k2": 2.0}
        {"outer": {"inner": {"x": 42}}}
          → {"outer.inner/x": 42}
    """
    flat = {}
    parent_keys = parent_keys or []

    for key, val in d.items():
        path = parent_keys + [key]
        if isinstance(val, dict):
            flat.update(flatten_logs(val, path))
        else:
            if len(path) == 1:
                flat_key = path[0]
            else:
                prefix = ".".join(path[:-1])
                flat_key = f"{prefix}/{path[-1]}"
            flat[flat_key] = val

    return flat