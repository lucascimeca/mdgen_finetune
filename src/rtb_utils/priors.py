import glob
from itertools import chain, repeat

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
from mdgen.dataset import MDGenDataset
from rtb_utils.simple_io import *

import os
import time
import torch
import mdtraj
import tqdm
import numpy as np
import pandas as pd

from rtb_utils.pytorch_utils import get_batch, cycle


class MDGenSimulator:
    def __init__(self,
                 sim_ckpt,
                 data_dir,
                 peptide=None,
                 split='splits/4AA_test.csv',
                 suffix='',
                 pdb_id=None,
                 num_frames=1,
                 num_rollouts=100,
                 no_frames=False,
                 tps=False,
                 xtc=True,
                 out_dir=".",
                 config=None,
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
        self.config = config

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
        self.model = NewMDGenWrapper.load_from_checkpoint(self.sim_ckpt, weights_only=False)
        self.model.eval().to(self.device)

        # Load the split file.
        self.df = pd.read_csv(self.split, index_col='name')

        self.target_dist_path = self.config.save_folder + "../target_dist.pt"
        if file_exists(self.target_dist_path):
            self.target_dist = load_dict_from_file(self.target_dist_path)
        else:

            self.target_dist = {}

        trainset = MDGenDataset(self.config,
                                split=self.split,
                                peptide=self.peptide,
                                data_dir=self.data_dir)

        self.train_loader = iter(cycle(torch.utils.data.DataLoader(
            trainset,
            batch_size=int(self.config.batch_size // self.config.vargrad_sample_n0),
            num_workers=0,
            shuffle=True,
            drop_last=False,
        )))

        self.batch = self._get_batch()
        self.dims = self.model.get_dims(self.batch)

    def _get_batch(self, device='cpu', multi_peptide=True, size=None):
        """
        Prepare a batch for simulation.

        Returns:
            dict: A batch dictionary.
        """

        # its = size % 8
        batch = next(self.train_loader)

        if not self.config.vargrad or not multi_peptide:

            if size is None:
                multiplier = self.config.batch_size
            else:
                multiplier = size

            for k, v in batch.items():
                if k == 'name':
                    batch[k] = [v[0]] * multiplier
                else:
                    batch[k] = v[0].to(device).unsqueeze(0).repeat_interleave(repeats=multiplier, dim=0)
        else:

            if size is None:
                multiplier = self.config.vargrad_sample_n0
            else:
                multiplier = size // batch['rots'].shape[0]

            for k, v in batch.items():
                if k == 'name':
                    batch[k] = list(chain.from_iterable(repeat(x, multiplier) for x in v))
                else:
                    batch[k] = v.to(device).repeat_interleave(repeats=multiplier, dim=0)

        return batch

    def get_cond_args(self, device, multi_peptide=True, size=None):
        # print(f"HERE SIZE {size}")
        batch = self._get_batch(device, multi_peptide=multi_peptide, size=size)

        prep = self.model.prep_batch(batch)
        cond_args = prep['model_kwargs']
        for k, v in cond_args.items():
            cond_args[k] = v.to(device)
        cond_args['peptide'] = batch['name']

        # DEBUG shape prints
        # [print(f"{k}: {v.shape}") for k, v in cond_args.items() if isinstance(v, torch.Tensor)]
        return cond_args, batch

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

    def fix_and_save_pdbs(self, frames, peptide):

        paths = []
        torsions = []
        for i in range(len(frames)):
            pdb_path = os.path.join(self.out_dir, f"{peptide}_{i}.pdb")
            pos37 = atom14_to_pdb(frames[i].unsqueeze(0).cpu().numpy(), self.batch['seqres'][0].cpu().numpy(), pdb_path)

            fixer = PDBFixer(filename=pdb_path)
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()

            torsions.append(atom37_to_torsions(pos37, aatype=self.batch['seqres'][0].cpu().numpy())[0].numpy())

            paths.append(pdb_path)
            with open(pdb_path, 'w') as f:
                PDBFile.writeFile(fixer.topology, fixer.positions, f, True)

        # Now, if xtc conversion is requested, load all individual pdb files and join them.
        if self.xtc:
            # Use glob to find all pdb files for this peptide.
            pdb_paths = sorted(glob.glob(os.path.join(self.out_dir, f"{peptide}_*.pdb")))
            # Load each pdb file into a trajectory.
            traj_list = [mdtraj.load(p) for p in pdb_paths]
            # Join the trajectories into a single trajectory.
            traj = mdtraj.join(traj_list)

            # Superpose the trajectory.
            traj.superpose(traj)

            # Save the trajectory as an xtc file.
            xtc_path = os.path.join(self.out_dir, f"{peptide}.xtc")
            traj.save(xtc_path)

        if torsions:
            torsions_path = os.path.join(self.out_dir, f"{peptide}_torsions.npy")
            np.save(torsions_path, np.stack(torsions))
        return paths

    def sample(self, batch, zs0=None):
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

        peptides = np.array(batch['name'])
        unique_peptides = np.unique(batch['name'])
        all_paths = []
        for peptide in unique_peptides:
            start_time = time.time()
            all_atom14 = []
            idx = np.nonzero(peptides == peptide)[0]
            custom_batch = {k: v[idx] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            for _ in tqdm.trange(self.num_rollouts, desc=f"Rollouts for {peptide}"):
                atom14, _ = self._rollout(custom_batch, zs0=zs0[idx])
                all_atom14.append(atom14)
            elapsed = time.time() - start_time
            print(f"Simulation for {peptide} took {elapsed:.2f} seconds.")

            all_atom14 = torch.cat(all_atom14, 1)
            paths = self.fix_and_save_pdbs(frames=all_atom14.squeeze(1), peptide=peptide)
            all_paths.extend(paths)

        return all_atom14[0].cpu().numpy(), self.batch['seqres'][0].cpu().numpy(), self.out_dir, all_paths