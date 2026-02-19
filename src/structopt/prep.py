"""Structure preparation using PDBFixer."""

import logging
from pathlib import Path

from structopt.config import OptimizationConfig

LOGGER = logging.getLogger(__name__)


def prepare_structure(config: OptimizationConfig) -> object:
    """Run PDBFixer-based cleanup and return a Modeller object."""
    from openmm.app import Modeller
    from pdbfixer import PDBFixer

    LOGGER.info("Loading structure into PDBFixer: %s", config.input_path)
    fixer = PDBFixer(filename=str(config.input_path))
    LOGGER.debug("Finding missing residues and atoms")
    fixer.findMissingResidues()
    if not config.add_missing_residues:
        LOGGER.info("Skipping missing residue insertion")
        fixer.missingResidues = {}

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    LOGGER.info("Adding missing hydrogens at pH %.2f", config.ph)
    fixer.addMissingHydrogens(pH=config.ph)

    # Keep heterogens by default so ligand residues remain in the model.
    if config.strip_heterogens:
        LOGGER.info("Removing heterogens (keep_water=%s)", config.keep_water)
        fixer.removeHeterogens(keepWater=config.keep_water)
    else:
        LOGGER.info("Keeping heterogens to preserve ligand residues")

    return Modeller(fixer.topology, fixer.positions)


def validate_input_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input structure does not exist: {path}")
