"""End-to-end optimization pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path

from structopt.config import OptimizationConfig
from structopt.io import build_default_output_path, detect_input_format, write_structure
from structopt.prep import prepare_structure, validate_input_exists

LOGGER = logging.getLogger(__name__)

WATER_RESNAMES = {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}
ION_RESNAMES = {
    "NA",
    "NA+",
    "K",
    "K+",
    "CL",
    "CL-",
    "LI",
    "LI+",
    "CS",
    "CS+",
    "RB",
    "RB+",
    "MG",
    "MG2+",
    "CA",
    "CA2+",
    "ZN",
    "ZN2+",
    "FE",
    "FE2+",
    "FE3+",
}
SOLVENT_ION_RESNAMES = WATER_RESNAMES | ION_RESNAMES


@dataclass
class OptimizationResult:
    output_path: Path
    final_energy_kj_mol: float
    minimized_energy_kj_mol: float | None = None
    refined_energy_kj_mol: float | None = None
    post_refined_energy_kj_mol: float | None = None


def _modeller_from_state(state: object) -> object:
    from openmm.app import Modeller

    return Modeller(state.topology, state.positions)


def _strip_solvent_and_ions(topology: object, positions: object) -> tuple[object, object]:
    from openmm.app import Modeller

    modeller = Modeller(topology, positions)
    residues_to_strip = [
        residue
        for residue in modeller.topology.residues()
        if residue.name.upper() in SOLVENT_ION_RESNAMES
    ]
    if residues_to_strip:
        LOGGER.info(
            "Stripping %d solvent/ion residues from output structure", len(residues_to_strip)
        )
        modeller.delete(residues_to_strip)
    return modeller.topology, modeller.positions


def run_optimization(config: OptimizationConfig) -> OptimizationResult:
    LOGGER.info("Validating input path: %s", config.input_path)
    validate_input_exists(config.input_path)
    output_format = detect_input_format(config.input_path)
    output_path = config.output_path or build_default_output_path(
        config.input_path,
        config.mode,
        output_format,
    )

    from structopt.sim import SimulationState, run_minimization, run_refinement_npt

    LOGGER.info(
        "Optimization settings: mode=%s, output=%s, ligand=%s, ligand_sdf=%s",
        config.mode,
        output_path,
        config.ligand_name,
        config.ligand_sdf,
    )
    modeller = prepare_structure(config)
    minimized = None
    refined = None

    if config.mode in {"minimize", "both"}:
        LOGGER.info("Running minimization stage")
        minimized = run_minimization(config, modeller)
        modeller = _modeller_from_state(minimized)
        if (
            config.mode == "both"
            and config.minimize_solvent == "explicit"
            and config.refine_solvent == "implicit"
        ):
            LOGGER.info("Stripping explicit solvent/ions before implicit-solvent refinement")
            dry_top, dry_pos = _strip_solvent_and_ions(minimized.topology, minimized.positions)
            modeller = _modeller_from_state(
                SimulationState(
                    topology=dry_top,
                    positions=dry_pos,
                    potential_energy_kj_mol=minimized.potential_energy_kj_mol,
                )
            )

    if config.mode in {"refine", "both"}:
        LOGGER.info("Running refinement stage")
        refined = run_refinement_npt(config, modeller)
        final_state = refined
    elif minimized is not None:
        final_state = minimized
    else:
        raise RuntimeError("No simulation state produced. Check optimization mode.")

    # After MD refinement, run a final energy minimization in the same solvent as the
    # initial minimization.  Strip any solvent/ions from the refined state first so
    # run_minimization always starts from a clean (dry) structure and can re-add solvent
    # as required by minimize_solvent.
    post_minimized = None
    if config.mode in {"refine", "both"}:
        LOGGER.info(
            "Running post-refinement minimization (minimize_solvent=%s)", config.minimize_solvent
        )
        dry_top, dry_pos = _strip_solvent_and_ions(final_state.topology, final_state.positions)
        dry_modeller = _modeller_from_state(
            SimulationState(
                topology=dry_top,
                positions=dry_pos,
                potential_energy_kj_mol=final_state.potential_energy_kj_mol,
            )
        )
        post_minimized = run_minimization(config, dry_modeller)
        final_state = post_minimized

    output_topology, output_positions = _strip_solvent_and_ions(
        final_state.topology,
        final_state.positions,
    )
    write_structure(output_path, output_topology, output_positions, output_format)
    LOGGER.info("Wrote optimized structure to: %s", output_path)

    return OptimizationResult(
        output_path=output_path,
        final_energy_kj_mol=final_state.potential_energy_kj_mol,
        minimized_energy_kj_mol=minimized.potential_energy_kj_mol if minimized else None,
        refined_energy_kj_mol=refined.potential_energy_kj_mol if refined else None,
        post_refined_energy_kj_mol=post_minimized.potential_energy_kj_mol
        if post_minimized
        else None,
    )
