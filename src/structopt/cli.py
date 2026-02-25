"""Typer CLI for StructOpt."""

import logging
from pathlib import Path
from typing import Annotated

import typer

from structopt.config import OptimizationConfig
from structopt.pipeline import run_optimization

app = typer.Typer(add_completion=False, help="Quick structure optimization with PDBFixer + OpenMM.")
LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@app.callback()
def root() -> None:
    """StructOpt command group."""


@app.command()
def optimize(
    input_path: Annotated[
        Path,
        typer.Argument(..., exists=True, readable=True, help="Input .pdb/.cif/.mmcif"),
    ],
    mode: Annotated[str, typer.Option(help="Optimization mode: minimize, refine, both.")] = "both",
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output structure path.")
    ] = None,
    ph: Annotated[float, typer.Option(help="pH used to add hydrogens.")] = 7.2,
    ligand_name: Annotated[str, typer.Option(help="Ligand residue name in the topology.")] = "LIG1",
    ligand_sdf: Annotated[
        Path | None,
        typer.Option(help="Optional ligand SDF file to help GAFF parametrization."),
    ] = None,
    log_level: Annotated[
        str,
        typer.Option(help="Logging verbosity: debug, info, warning, error."),
    ] = "info",
    minimize_solvent: Annotated[
        str, typer.Option(help="Minimization solvent model: explicit or implicit.")
    ] = "explicit",
    refine_solvent: Annotated[
        str, typer.Option(help="Refinement (short MD) solvent model: explicit or implicit.")
    ] = "explicit",
    implicit_solvent: Annotated[
        str, typer.Option(help="Implicit solvent model for minimization.")
    ] = "gbn2",
    minimize_max_iter: Annotated[int, typer.Option(help="Maximum minimization iterations.")] = 5000,
    temperature: Annotated[float, typer.Option(help="Simulation temperature (K).")] = 300.0,
    pressure: Annotated[float, typer.Option(help="Pressure for NPT refinement (bar).")] = 1.0,
    timestep_fs: Annotated[float, typer.Option(help="Integrator time step (fs).")] = 2.0,
    friction_ps: Annotated[float, typer.Option(help="Langevin friction coefficient (1/ps).")] = 1.0,
    equil_steps: Annotated[
        int, typer.Option(help="Equilibration steps before production NPT.")
    ] = 5000,
    npt_steps: Annotated[int, typer.Option(help="Production NPT steps.")] = 50000,
    report_interval: Annotated[int, typer.Option(help="Reporter interval.")] = 1000,
    restraint_k: Annotated[
        float, typer.Option(help="Backbone restraint strength (kcal/mol/A^2).")
    ] = 1.0,
    solvent_padding_nm: Annotated[
        float, typer.Option(help="Solvent padding for NPT refinement (nm).")
    ] = 1.0,
    ionic_strength_molar: Annotated[float, typer.Option(help="Ionic strength (M).")] = 0.15,
    nonbonded_cutoff_nm: Annotated[float, typer.Option(help="Nonbonded cutoff (nm).")] = 1.0,
    device: Annotated[str, typer.Option(help="Compute device: auto, cpu, cuda, opencl.")] = "auto",
    random_seed: Annotated[int | None, typer.Option(help="Optional RNG seed.")] = None,
) -> None:
    """Optimize a structure from PDB/mmCIF input."""
    _configure_logging(log_level)
    LOGGER.info("Starting optimization for input: %s", input_path)
    try:
        config = OptimizationConfig(
            input_path=input_path,
            output_path=output,
            mode=mode,
            ph=ph,
            ligand_name=ligand_name,
            ligand_sdf=ligand_sdf,
            log_level=log_level.lower(),
            minimize_solvent=minimize_solvent,
            refine_solvent=refine_solvent,
            implicit_solvent=implicit_solvent,
            minimize_max_iter=minimize_max_iter,
            temperature_k=temperature,
            pressure_bar=pressure,
            timestep_fs=timestep_fs,
            friction_per_ps=friction_ps,
            equil_steps=equil_steps,
            npt_steps=npt_steps,
            report_interval=report_interval,
            restraint_k_kcal_per_a2=restraint_k,
            solvent_padding_nm=solvent_padding_nm,
            ionic_strength_molar=ionic_strength_molar,
            nonbonded_cutoff_nm=nonbonded_cutoff_nm,
            device=device,
            random_seed=random_seed,
        )
        result = run_optimization(config)
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Output: {result.output_path}", fg=typer.colors.GREEN)
    typer.echo(f"Final energy (kJ/mol): {result.final_energy_kj_mol:.3f}")
    if result.minimized_energy_kj_mol is not None:
        typer.echo(f"Minimized energy (kJ/mol): {result.minimized_energy_kj_mol:.3f}")
    if result.refined_energy_kj_mol is not None:
        typer.echo(f"Refined energy (kJ/mol): {result.refined_energy_kj_mol:.3f}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
