"""End-to-end optimization pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path

from structopt.config import OptimizationConfig
from structopt.io import build_default_output_path, detect_input_format, write_structure
from structopt.prep import prepare_structure, validate_input_exists

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    output_path: Path
    final_energy_kj_mol: float
    minimized_energy_kj_mol: float | None = None
    refined_energy_kj_mol: float | None = None


def _modeller_from_state(state: object) -> object:
    from openmm.app import Modeller

    return Modeller(state.topology, state.positions)


def run_optimization(config: OptimizationConfig) -> OptimizationResult:
    LOGGER.info("Validating input path: %s", config.input_path)
    validate_input_exists(config.input_path)
    output_format = detect_input_format(config.input_path)
    output_path = config.output_path or build_default_output_path(
        config.input_path,
        config.mode,
        output_format,
    )

    from structopt.sim import run_minimization, run_refinement_npt

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

    if config.mode in {"refine", "both"}:
        LOGGER.info("Running refinement stage")
        refined = run_refinement_npt(config, modeller)
        final_state = refined
    elif minimized is not None:
        final_state = minimized
    else:
        raise RuntimeError("No simulation state produced. Check optimization mode.")

    write_structure(output_path, final_state.topology, final_state.positions, output_format)
    LOGGER.info("Wrote optimized structure to: %s", output_path)

    return OptimizationResult(
        output_path=output_path,
        final_energy_kj_mol=final_state.potential_energy_kj_mol,
        minimized_energy_kj_mol=minimized.potential_energy_kj_mol if minimized else None,
        refined_energy_kj_mol=refined.potential_energy_kj_mol if refined else None,
    )
