"""Configuration models for structure optimization."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class OptimizationConfig(BaseModel):
    """Validated configuration for a single optimization run."""

    input_path: Path
    output_path: Path | None = None
    mode: Literal["minimize", "refine", "both"] = "both"
    ph: float = Field(default=7.2, ge=0.0, le=14.0)

    ligand_name: str | None = None
    ligand_sdf: Path | None = None
    gaff_forcefield: str = "gaff-2.11"
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    implicit_solvent: Literal["gbn2", "obc2"] = "gbn2"
    minimize_max_iter: int = Field(default=10000, ge=1)

    temperature_k: float = Field(default=300.0, gt=0.0)
    pressure_bar: float = Field(default=1.0, gt=0.0)
    timestep_fs: float = Field(default=2.0, gt=0.0)
    friction_per_ps: float = Field(default=1.0, gt=0.0)
    equil_steps: int = Field(default=5000, ge=0)
    npt_steps: int = Field(default=50000, ge=0)
    report_interval: int = Field(default=100, ge=1)
    restraint_k_kcal_per_a2: float = Field(default=1.0, ge=0.0)
    random_seed: int | None = None

    solvent_padding_nm: float = Field(default=1.0, ge=0.1)
    ionic_strength_molar: float = Field(default=0.15, ge=0.0)
    nonbonded_cutoff_nm: float = Field(default=1.0, ge=0.5)

    device: Literal["auto", "cpu", "cuda", "opencl"] = "auto"

    add_missing_residues: bool = False
    strip_heterogens: bool = False
    keep_water: bool = True

    @model_validator(mode="after")
    def check_input(self) -> "OptimizationConfig":
        suffix = self.input_path.suffix.lower()
        if suffix not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("input_path must be one of: .pdb, .cif, .mmcif")
        if self.ligand_sdf is not None:
            if self.ligand_sdf.suffix.lower() != ".sdf":
                raise ValueError("ligand_sdf must point to an .sdf file")
            if not self.ligand_sdf.exists():
                raise ValueError(f"ligand_sdf does not exist: {self.ligand_sdf}")
        if self.mode == "refine" and self.npt_steps == 0 and self.equil_steps == 0:
            raise ValueError("refine mode requires equil_steps > 0 or npt_steps > 0")
        return self
