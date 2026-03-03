"""Typer CLI for StructOpt."""

import logging
from glob import glob
from pathlib import Path
from typing import Annotated

import typer

from structopt.config import OptimizationConfig
from structopt.io import build_default_output_path, detect_input_format
from structopt.pipeline import run_optimization

app = typer.Typer(add_completion=False, help="Quick structure optimization with PDBFixer + OpenMM.")
LOGGER = logging.getLogger(__name__)
_ALLOWED_INPUT_SUFFIXES = {".pdb", ".cif", ".mmcif"}
_WILDCARD_CHARS = {"*", "?", "["}


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _is_structure_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _ALLOWED_INPUT_SUFFIXES


def _expand_input_spec(spec: str) -> list[Path]:
    if any(char in spec for char in _WILDCARD_CHARS):
        return [
            Path(match)
            for match in sorted(glob(spec))
            if _is_structure_file(Path(match))
        ]

    candidate = Path(spec)
    if not candidate.exists():
        raise ValueError(f"Input path does not exist: {spec}")
    if candidate.is_dir():
        return [path for path in sorted(candidate.iterdir()) if _is_structure_file(path)]
    if _is_structure_file(candidate):
        return [candidate]
    raise ValueError(
        f"Unsupported input format for {candidate}. Allowed: .pdb, .cif, .mmcif"
    )


def _resolve_input_paths(input_specs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for spec in input_specs:
        matches = _expand_input_spec(spec)
        if not matches:
            raise ValueError(
                f"No supported structure files found for input spec: {spec} "
                "(allowed: .pdb, .cif, .mmcif)"
            )
        for path in matches:
            canonical = path.resolve()
            if canonical not in seen:
                seen.add(canonical)
                resolved.append(path)
    return resolved


def _build_output_path_for_input(input_path: Path, output: Path | None, mode: str) -> Path | None:
    if output is None:
        return None
    if output.exists() and output.is_file():
        return output
    if output.suffix:
        return output

    output_format = detect_input_format(input_path)
    default_name = build_default_output_path(input_path, mode, output_format).name
    return output / default_name


@app.callback()
def root() -> None:
    """StructOpt command group."""


@app.command()
def optimize(
    input_specs: Annotated[
        list[str],
        typer.Argument(
            ...,
            metavar="INPUT",
            help=(
                "One or more inputs: file path(s), a directory, or wildcard pattern(s) "
                "for .pdb/.cif/.mmcif files."
            ),
        ),
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
    nonbonded_cutoff_nm: Annotated[float, typer.Option(help="Nonbonded cutoff (nm).")]= 1.0,
    device: Annotated[str, typer.Option(help="Compute device: auto, cpu, cuda, opencl.")] = "auto",
    random_seed: Annotated[int | None, typer.Option(help="Optional RNG seed.")] = None,
) -> None:
    """Optimize one or more structures from PDB/mmCIF input."""
    _configure_logging(log_level)
    try:
        input_paths = _resolve_input_paths(input_specs)
        if len(input_paths) > 1 and output is not None and output.suffix:
            raise ValueError(
                "For batch optimization, --output must be omitted or point to a directory."
            )

        for input_path in input_paths:
            current_output = _build_output_path_for_input(input_path, output, mode)
            LOGGER.info("Starting optimization for input: %s", input_path)
            config = OptimizationConfig(
                input_path=input_path,
                output_path=current_output,
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

            typer.secho(f"Input:  {input_path}", fg=typer.colors.BLUE)
            typer.secho(f"Output: {result.output_path}", fg=typer.colors.GREEN)
            typer.echo(f"Final energy (kJ/mol): {result.final_energy_kj_mol:.3f}")
            if result.minimized_energy_kj_mol is not None:
                typer.echo(f"Minimized energy (kJ/mol): {result.minimized_energy_kj_mol:.3f}")
            if result.refined_energy_kj_mol is not None:
                typer.echo(f"Refined energy (kJ/mol): {result.refined_energy_kj_mol:.3f}")
            if result.post_refined_energy_kj_mol is not None:
                typer.echo(f"Post-refined energy (kJ/mol): {result.post_refined_energy_kj_mol:.3f}")
    except Exception as exc:  # noqa: BLE001
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def main() -> None:
    app()


if __name__ == "__main__":
    main()
