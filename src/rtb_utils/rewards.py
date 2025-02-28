from simtk.openmm import Platform, LangevinMiddleIntegrator, MonteCarloBarostat
from simtk.openmm.app import ForceField, Simulation, PDBFile, Modeller, PME, HBonds
from simtk.unit import kelvin, picosecond, femtosecond, nanometer, bar


class Amber14Reward:
    def __init__(self, device='CPU', implicit=False, friction_coeff=1.0, dt=2.0 * femtosecond):
        """
        Initializes the reward function with the appropriate forcefield and simulation parameters.

        Parameters:
            device (str or torch.device): 'CPU' or 'CUDA' (default 'CPU'); if a torch.device is passed,
                                           it will be converted to a string.
            implicit (bool): Whether to use an implicit solvent model.
            friction_coeff (float): Friction coefficient (in 1/ps units).
            dt: Timestep for the integrator.
        """
        # Convert device to string (upper case to match OpenMM's expected names)
        self.device = str(device).upper()
        self.implicit = implicit
        self.friction_coeff = friction_coeff
        self.dt = dt

        # Select forcefield based on solvent model.
        if self.implicit:
            self.forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
        else:
            self.forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

        self.platform = Platform.getPlatformByName(self.device)
        # Use LangevinMiddleIntegrator at 350 K.
        self.integrator = LangevinMiddleIntegrator(350 * kelvin, friction_coeff / picosecond, dt)

    def compute_reward(self, pdb):
        """
        Given a conformation in the form of an OpenMM PDBFile (with .topology and .positions),
        create a system (adding hydrogens and solvent as needed), minimize the energy,
        and return the computed potential energy.

        Parameters:
            pdb (PDBFile): An OpenMM PDBFile with topology and positions.

        Returns:
            potential_energy: The potential energy (as an OpenMM Quantity).
        """
        # Create a modeller from the provided PDB.
        modeller = Modeller(pdb.topology, pdb.positions)
        # Add hydrogens at pH 7.
        modeller.addHydrogens(self.forcefield, pH=7)

        # Create system: use explicit solvent if not implicit.
        if self.implicit:
            system = self.forcefield.createSystem(modeller.topology, constraints=HBonds)
        else:
            modeller.addSolvent(self.forcefield, padding=1.0 * nanometer)
            system = self.forcefield.createSystem(modeller.topology,
                                                  nonbondedMethod=PME,
                                                  nonbondedCutoff=1.0 * nanometer,
                                                  constraints=HBonds)

        # Create the simulation.
        simulation = Simulation(modeller.topology, system, self.integrator, self.platform)
        simulation.context.setPositions(modeller.positions)

        # Minimize the energy.
        simulation.minimizeEnergy()

        # Get the minimized potential energy.
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()

        return potential_energy
