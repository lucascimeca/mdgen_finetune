from openmm.app import PDBFile, Simulation, ForceField, NoCutoff
from openmm import LangevinIntegrator, Platform, NonbondedForce
from openmm.unit import kelvin, picosecond, femtosecond, nanometer

class Amber14Reward:
    def __init__(self, device):
        self.device = device
        # Use AMBER14 force field files (note the filename change from ff14SB)
        self.forcefield = ForceField('amber/protein.AMBER14.xml', 'amber/tip3p_standard.xml')
        self.platform = Platform.getPlatformByName('CPU')  # or use 'CUDA' if available
        self.integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 2 * femtosecond)

    def compute_reward(self, pdb):
        """
        Given a conformation in the form of an OpenMM PDBFile (with .topology and .positions),
        create a system, adjust nonbonded parameters, and return the computed potential energy.
        """
        system = self.forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff)
        # Modify nonbonded force parameters (smaller cutoff)
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setNonbondedMethod(NonbondedForce.NoCutoff)
                force.setCutoffDistance(1.0 * nanometer)

        # Create a simulation instance
        simulation = Simulation(pdb.topology, system, self.integrator, self.platform)
        simulation.context.setPositions(pdb.positions)
        state = simulation.context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()

        return potential_energy