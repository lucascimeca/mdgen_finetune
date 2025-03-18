import glob

from openmm.app import PDBFile
from pdbfixer import PDBFixer
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb

import os
import time
import torch
import mdtraj
import tqdm
import numpy as np
import pandas as pd

from rtb_utils.pytorch_utils import get_batch


class MDGenSimulator:
    def __init__(self,
                 sim_ckpt,
                 data_dir,
                 peptide,
                 split='splits/4AA_test.csv',
                 suffix='',
                 pdb_id=None,
                 num_frames=1,
                 num_rollouts=100,
                 retain=300,
                 no_frames=False,
                 tps=False,
                 xtc=False,
                 out_dir=".",
                 device=None):
        """
        Initialize the MDGenSimulator.

        Parameters:
            sim_ckpt (str): Path to the simulation checkpoint.
            data_dir (str): Directory containing the data.
            split (str): CSV file with simulation splits (default 'splits/4AA_test.csv').
            suffix (str): Suffix for the data file names.
            pdb_id (list): List of pdb IDs to simulate. If empty, the first available simulation is used.
            num_frames (int): Number of frames to simulate.
            num_rollouts (int): Number of rollouts per simulation.
            no_frames (bool): If True, simulation is run without frames.
            tps (bool): If True, keep all frames from the data.
            xtc (bool): If True, output an additional xtc trajectory file.
            out_dir (str): Directory where results are saved.
            device: Torch device to run simulation on. If None, auto-detect.
        """
        self.sim_ckpt = sim_ckpt
        self.data_dir = data_dir
        self.split = split
        self.suffix = suffix
        self.pdb_id = pdb_id if pdb_id is not None else []
        self.num_frames = num_frames
        self.num_rollouts = num_rollouts
        self.no_frames = no_frames
        self.tps = tps
        self.xtc = xtc
        self.out_dir = out_dir
        self.peptide = peptide

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        os.makedirs(self.out_dir, exist_ok=True)

        # Load the simulation model.
        self.model = NewMDGenWrapper.load_from_checkpoint(self.sim_ckpt)
        self.model.eval().to(self.device)

        # Load the split file.
        self.df = pd.read_csv(self.split, index_col='name')

        item = self._get_batch(retain=retain)
        self.batch = next(iter(torch.utils.data.DataLoader([item])))
        self.batch = tensor_tree_map(lambda x: x.to(self.device), self.batch)
        self.dims = self.model.get_dims(self.batch)

        self.target_dist = None

    def _get_batch(self, retain=300):
        """
        Prepare a batch for simulation.

        Returns:
            dict: A batch dictionary.
        """
        file_path = os.path.join(self.data_dir, f"{self.peptide}{self.suffix}.npy")
        arr = np.lib.format.open_memmap(file_path, 'r')

        idxes = np.random.randint(0, len(arr), size=retain)
        if not self.tps:  # if tps flag is not set, use only the first frame
            self.batch_arr = np.copy(arr[idxes]).astype(np.float32)
            arr = np.copy(arr[0:1]).astype(np.float32)
        else:
            self.batch_arr = arr

        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres_tensor = torch.tensor([restype_order[c] for c in self.peptide])
        atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres_tensor[None])).float()
        L = len(seqres_tensor)
        mask = torch.ones(L)

        if self.no_frames:
            return {
                'atom37': atom37,
                'seqres': seqres_tensor,
                'mask': restype_atom37_mask[seqres_tensor],
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, seqres_tensor[None])
        return {
            'torsions': torsions,
            'torsion_mask': torsion_mask[0],
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres_tensor,
            'mask': mask,
        }

    def get_cond_args(self, device):
        item = get_batch(self.peptide,
                         self.peptide,
                         tps=self.tps,
                         no_frames=self.model.args.no_frames,
                         data_dir=self.data_dir,
                         suffix=self.suffix)
        batch = next(iter(torch.utils.data.DataLoader([item])))
        prep = self.model.prep_batch(batch)
        cond_args = prep['model_kwargs']
        for k, v in cond_args.items():
            cond_args[k] = v.to(device)
        return cond_args

    def _rollout(self, batch, zs0=None):
        """
        Perform one rollout of the simulation.

        Parameters:
            batch (dict): The simulation batch.

        Returns:
            tuple: A tuple (atom14, updated_batch)
        """
        if self.no_frames:
            expanded_batch = {
                'atom37': batch['atom37'].expand(-1, self.num_frames, -1, -1, -1),
                'seqres': batch['seqres'],
                'mask': batch['mask'],
            }
        else:
            expanded_batch = {
                'torsions': batch['torsions'].expand(-1, self.num_frames, -1, -1, -1),
                'torsion_mask': batch['torsion_mask'],
                'trans': batch['trans'].expand(-1, self.num_frames, -1, -1),
                'rots': batch['rots'].expand(-1, self.num_frames, -1, -1, -1),
                'seqres': batch['seqres'],
                'mask': batch['mask'],
            }
        atom14, _ = self.model.inference(expanded_batch, zs0=zs0)
        new_batch = {**batch}

        if self.no_frames:
            new_batch['atom37'] = torch.from_numpy(
                atom14_to_atom37(atom14[:, -1].cpu(), batch['seqres'][0].cpu())
            ).to(self.device)[:, None].float()
        else:
            frames = atom14_to_frames(atom14[:, -1])
            new_batch['trans'] = frames._trans[None]
            new_batch['rots'] = frames._rots._rot_mats[None]
            atom37 = atom14_to_atom37(atom14[0, -1].cpu(), batch['seqres'][0].cpu())
            torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
            new_batch['torsions'] = torsions[None, None].to(self.device)

        return atom14, new_batch

    def fix_and_save_pdbs(self, frames):

        for i in range(len(frames)):
            pdb_path = os.path.join(self.out_dir, f"{self.peptide}_{i}.pdb")
            atom14_to_pdb(frames[i].unsqueeze(0).cpu().numpy(), self.batch['seqres'][0].cpu().numpy(), pdb_path)

            fixer = PDBFixer(filename=pdb_path)
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            with open(pdb_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f, True)

        # Now, if xtc conversion is requested, load all individual pdb files and join them.
        if self.xtc:
            # Use glob to find all pdb files for this peptide.
            pdb_paths = sorted(glob.glob(os.path.join(self.out_dir, f"{self.peptide}_*.pdb")))
            # Load each pdb file into a trajectory.
            traj_list = [mdtraj.load(p) for p in pdb_paths]
            # Join the trajectories into a single trajectory.
            traj = mdtraj.join(traj_list)

            # Superpose the trajectory.
            traj.superpose(traj)

            # Save the trajectory as an xtc file.
            xtc_path = os.path.join(self.out_dir, f"{self.peptide}.xtc")
            traj.save(xtc_path)

    def sample(self, zs0=None):
        """
        Run one simulation and generate the corresponding pdb file.

        Parameters:
            name (str, optional): The simulation identifier. If not provided, the first valid
                                  simulation from the split file is used.
        """
        """
          Run the simulation for a given name and residue sequence, and write output files.

          Parameters:
              name (str): Name of peptide.
              seqres (str): Residue sequence string.
          """

        all_atom14 = []
        start_time = time.time()
        for _ in tqdm.trange(self.num_rollouts, desc=f"Rollouts for {self.peptide}"):
            atom14, batch = self._rollout(self.batch, zs0=zs0)
            all_atom14.append(atom14)
        elapsed = time.time() - start_time
        print(f"Simulation for {self.peptide} took {elapsed:.2f} seconds.")

        all_atom14 = torch.cat(all_atom14, 1)

        self.fix_and_save_pdbs(frames=all_atom14.squeeze(1))

        return all_atom14[0].cpu().numpy(), self.batch['seqres'][0].cpu().numpy(), self.out_dir