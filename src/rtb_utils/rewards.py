import glob
import os
from time import time

from openmm.app import PDBFile, Modeller, Simulation, PME, HBonds, Topology, ForceField
from openmm.app.element import Element
from openmm.unit import nanometer, bar, kelvin, picosecond, femtosecond
from openmm import LangevinMiddleIntegrator, Platform, Vec3, MonteCarloBarostat
from tqdm import tqdm

from mdgen.utils import atom14_to_pdb

from mdgen.geometry import atom14_to_atom37
from mdgen.utils import create_full_prot, prots_to_pdb

import torch
import torch.nn as nn
import numpy as np
import mdtraj
import concurrent.futures
from openmm import Platform

def choose_device():
    """
    Return 'CUDA' if the CUDA platform is available,
    otherwise return 'CPU'.
    """
    n_platforms = Platform.getNumPlatforms()
    available_platforms = [
        Platform.getPlatform(i).getName() for i in range(n_platforms)
    ]
    if "CUDA" in available_platforms:
        return "CUDA"
    return "CPU"

DEVICE = choose_device()


class Amber14Reward(nn.Module):
    def __init__(self, device='CPU', implicit=False, friction_coeff=1.0,
                 dt=2.0 * femtosecond, *args, **kwargs):
        """
        Initializes the reward function with the appropriate force field and simulation parameters.

        Parameters:
            device (str or torch.device): 'CPU' or 'CUDA' (default 'CPU'); if a torch.device is passed,
                                           it will be converted to a string.
            implicit (bool): Whether to use an implicit solvent model.
            friction_coeff (float): Friction coefficient (in 1/ps units).
            dt: Timestep for the integrator.
        """
        super().__init__(*args, **kwargs)
        self.device = str(device).upper()
        self.implicit = implicit
        self.friction_coeff = friction_coeff
        self.dt = dt

        # Select force field based on solvent model.
        if self.implicit:
            self.forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        else:
            self.forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        self.platform = Platform.getPlatformByName(self.device)
        self.integrator = LangevinMiddleIntegrator(350 * kelvin, friction_coeff / picosecond, dt)

    def forward(self, sequence, data_path, tmp_dir='../samples/'):
        """
        Compute energies for a batch of conformations.

        Parameters:
            confs: a tuple with three elements:
                - confs[0]: NumPy array of positions with shape (B, T, L, 3)
                - confs[1]: encoded sequence (e.g. [13, 10, 1, 8]) -- assumed same for all
                - confs[2]: a template PDB file path to build the system (e.g., for FLRH)

        Returns:
            A list of potential energy values (one per conformation).
        """
        # Create the base PDB and modeller using the provided template PDB file.

        t0 = time()
        pdb = PDBFile(f'{data_path}/4AA_sims/FLRH/{sequence}.pdb')
        traj = mdtraj.load(f'{tmp_dir}{sequence}.xtc',
                           top=f'{tmp_dir}{sequence}_0.pdb')

        energies = []

        # Depending on how your topology is ordered, you might need to reshape or re-order the positions.
        # Here, we assume that the full set of positions for the protein is a flattened (T*L, 3) array.
        print("computing energies")
        for i in tqdm(range(len(traj.xyz)), total=len(traj.xyz)):

            modeller = Modeller(pdb.topology, traj.xyz[i])

            modeller.addHydrogens(self.forcefield, pH=7)

            # Build the system (using implicit or explicit solvent as needed).
            if self.implicit:
                system = self.forcefield.createSystem(modeller.topology, constraints=HBonds)
            else:
                modeller.addSolvent(self.forcefield, padding=1.0 * nanometer)
                system = self.forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=PME,
                    nonbondedCutoff=1.0 * nanometer,
                    constraints=HBonds
                )

            # Create the integrator and simulation.
            dt = 2 * femtosecond
            integrator = LangevinMiddleIntegrator(350 * kelvin, self.friction_coeff / picosecond, dt)
            simulation = Simulation(modeller.topology, system, integrator,
                                    platform=Platform.getPlatformByName(DEVICE))
            # Set the initial positions from the template (or modeller) – will be overwritten.
            simulation.context.setPositions(modeller.positions)

            # If using explicit solvent, add a barostat (and reinitialize if needed).
            if not self.implicit:
                system.addForce(MonteCarloBarostat(1 * bar, 350 * kelvin))
                simulation.context.reinitialize(preserveState=True)

            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy()
            energies.append(energy._value)
        print("removing pdb files")
        for f in glob.glob(os.path.join(tmp_dir, "*.pdb")) + glob.glob(os.path.join(tmp_dir, "*.xtc")):
            os.remove(f)

        t1 = time()
        print(f"elapsed {t1 - t0}")

        return -torch.FloatTensor(energies)

    # def forward(self, sequence, data_path, tmp_dir='../samples/'):
    #     t0 = time()
    #     pdb = PDBFile(f'{data_path}/4AA_sims/FLRH/{sequence}.pdb')
    #     traj = mdtraj.load(f'{tmp_dir}{sequence}.xtc', top=f'{tmp_dir}{sequence}_0.pdb')
    #
    #     # Precompute all hydrogens-added positions
    #     all_positions = []
    #     base_topology = None
    #     for i in range(len(traj.xyz)):
    #         modeller = Modeller(pdb.topology, traj.xyz[i])
    #         modeller.addHydrogens(self.forcefield, pH=7)
    #         if i == 0:
    #             base_topology = modeller.topology
    #         all_positions.append(modeller.positions)
    #
    #     # Configure platform
    #     platform = Platform.getPlatformByName(DEVICE)
    #     if DEVICE == 'CUDA':
    #         platform.setPropertyDefaultValue('Precision', 'mixed')
    #
    #     # Initialize simulation based on solvent type
    #     if self.implicit:
    #         system = self.forcefield.createSystem(base_topology, constraints=HBonds)
    #         integrator = LangevinMiddleIntegrator(350 * kelvin, self.friction_coeff / picosecond, 2 * femtosecond)
    #         simulation = Simulation(base_topology, system, integrator, platform=platform)
    #         simulation.context.setPositions(all_positions[0])
    #         _ = simulation.context.getState(getEnergy=True)  # Warmup
    #
    #     energies = []
    #     for i in range(len(traj.xyz)):
    #         if self.implicit:
    #             simulation.context.setPositions(all_positions[i])
    #             state = simulation.context.getState(getEnergy=True)
    #             energies.append(state.getPotentialEnergy()._value)
    #         else:
    #             # Explicit solvent: create fresh simulation per frame
    #             modeller = Modeller(base_topology, all_positions[i])
    #             modeller.addSolvent(self.forcefield, padding=1.0 * nanometer)
    #             system = self.forcefield.createSystem(
    #                 modeller.topology,
    #                 nonbondedMethod=PME,
    #                 nonbondedCutoff=1.0 * nanometer,
    #                 constraints=HBonds
    #             )
    #             integrator = LangevinMiddleIntegrator(350 * kelvin, self.friction_coeff / picosecond, 2 * femtosecond)
    #             simulation = Simulation(
    #                 modeller.topology,
    #                 system,
    #                 integrator,
    #                 platform=platform
    #             )
    #             simulation.context.setPositions(modeller.positions)
    #             system.addForce(MonteCarloBarostat(1 * bar, 350 * kelvin))
    #             simulation.context.reinitialize(preserveState=True)
    #             state = simulation.context.getState(getEnergy=True)
    #             energies.append(state.getPotentialEnergy()._value)
    #
    #     t1 = time()
    #     print(f"elapsed {t1 - t0}")
    #     return energies

# 91.64365983009338