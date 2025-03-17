from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask

import numpy as np
import torch
import torch.nn.init as init


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