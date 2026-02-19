"""OpenMM simulation helpers."""

import logging
from dataclasses import dataclass
from typing import Literal

import openmm as mm
from openmm import unit
from openmm.app import (
    PME,
    CutoffNonPeriodic,
    ForceField,
    HBonds,
    Modeller,
    Simulation,
)

from structopt.config import OptimizationConfig
from structopt.ligand import register_gaff_template

LOGGER = logging.getLogger(__name__)


@dataclass
class SimulationState:
    topology: object
    positions: unit.Quantity
    potential_energy_kj_mol: float


WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}


def _platform_for_device(
    device: Literal["auto", "cpu", "cuda", "opencl"],
) -> tuple[mm.Platform | None, dict]:
    if device == "auto":
        return None, {}
    if device == "cpu":
        return mm.Platform.getPlatformByName("CPU"), {}
    if device == "cuda":
        return mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"}
    if device == "opencl":
        return mm.Platform.getPlatformByName("OpenCL"), {"Precision": "mixed"}
    return None, {}


def _create_simulation(
    system: mm.System,
    modeller: Modeller,
    config: OptimizationConfig,
) -> Simulation:
    integrator = mm.LangevinMiddleIntegrator(
        config.temperature_k * unit.kelvin,
        config.friction_per_ps / unit.picosecond,
        config.timestep_fs * unit.femtoseconds,
    )
    if config.random_seed is not None:
        integrator.setRandomNumberSeed(config.random_seed)
        LOGGER.info("Using deterministic random seed: %d", config.random_seed)

    platform, properties = _platform_for_device(config.device)
    if platform is None:
        sim = Simulation(modeller.topology, system, integrator)
    else:
        sim = Simulation(modeller.topology, system, integrator, platform, properties)

    sim.context.setPositions(modeller.positions)
    return sim


def run_minimization(config: OptimizationConfig, modeller: Modeller) -> SimulationState:
    ff_files = ["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"]
    if config.minimize_solvent == "implicit":
        if config.implicit_solvent == "gbn2":
            ff_files.append("implicit/gbn2.xml")
        else:
            ff_files.append("implicit/obc2.xml")

    ff = ForceField(*ff_files)
    ligand_registered = register_gaff_template(ff, config, modeller.topology, modeller.positions)
    if ligand_registered:
        LOGGER.info("Adding any remaining hydrogens using registered force field templates")
        modeller.addHydrogens(ff, pH=config.ph)

    if config.minimize_solvent == "explicit":
        LOGGER.info(
            "Adding solvent for minimization (padding=%.3f nm, ionic_strength=%.3f M)",
            config.solvent_padding_nm,
            config.ionic_strength_molar,
        )
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=config.solvent_padding_nm * unit.nanometer,
            ionicStrength=config.ionic_strength_molar * unit.molar,
            neutralize=True,
        )
        LOGGER.info(
            "Creating minimization system with explicit solvent and cutoff=%.3f nm",
            config.nonbonded_cutoff_nm,
        )
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
            constraints=HBonds,
        )
    else:
        LOGGER.info(
            "Creating minimization system with implicit solvent=%s and cutoff=%.3f nm",
            config.implicit_solvent,
            config.nonbonded_cutoff_nm,
        )
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
            constraints=HBonds,
        )

    sim = _create_simulation(system, modeller, config)
    LOGGER.info("Running energy minimization (max_iterations=%d)", config.minimize_max_iter)
    sim.minimizeEnergy(maxIterations=config.minimize_max_iter)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    return SimulationState(
        topology=modeller.topology,
        positions=state.getPositions(),
        potential_energy_kj_mol=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
    )


def run_refinement_npt(config: OptimizationConfig, modeller: Modeller) -> SimulationState:
    ff = ForceField("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    ligand_registered = register_gaff_template(ff, config, modeller.topology, modeller.positions)
    if ligand_registered:
        LOGGER.info("Adding any remaining hydrogens using registered force field templates")
        modeller.addHydrogens(ff, pH=config.ph)

    if any(res.name in WATER_RESNAMES for res in modeller.topology.residues()):
        LOGGER.info(
            "Detected existing solvent in topology; skipping solvent addition for refinement"
        )
    else:
        LOGGER.info(
            "Adding solvent for NPT refinement (padding=%.3f nm, ionic_strength=%.3f M)",
            config.solvent_padding_nm,
            config.ionic_strength_molar,
        )
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=config.solvent_padding_nm * unit.nanometer,
            ionicStrength=config.ionic_strength_molar * unit.molar,
            neutralize=True,
        )
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
        constraints=HBonds,
        hydrogenMass=1.5*unit.amu,
    )
    system.addForce(
        mm.MonteCarloBarostat(
            config.pressure_bar * unit.bar,
            config.temperature_k * unit.kelvin,
        )
    )

    sim = _create_simulation(system, modeller, config)
    # Re-relax the exact NPT system (restraints/barostat/constraints) before dynamics.
    LOGGER.info(
        "Running pre-refinement minimization (max_iterations=%d)", config.minimize_max_iter
    )
    sim.minimizeEnergy(maxIterations=config.minimize_max_iter)

    if config.random_seed is not None:
        sim.context.setVelocitiesToTemperature(
            config.temperature_k * unit.kelvin, config.random_seed
        )
    else:
        sim.context.setVelocitiesToTemperature(config.temperature_k * unit.kelvin)

    if config.equil_steps > 0:
        LOGGER.info("Running equilibration steps: %d", config.equil_steps)
        sim.step(config.equil_steps)
    if config.npt_steps > 0:
        LOGGER.info("Running NPT production steps: %d", config.npt_steps)
        sim.step(config.npt_steps)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    return SimulationState(
        topology=modeller.topology,
        positions=state.getPositions(),
        potential_energy_kj_mol=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
    )
