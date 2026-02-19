"""I/O helpers for structure files."""

from pathlib import Path
from typing import Literal

InputFormat = Literal["pdb", "cif"]


def detect_input_format(path: Path) -> InputFormat:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return "pdb"
    if suffix in {".cif", ".mmcif"}:
        return "cif"
    raise ValueError(f"Unsupported input format: {path.suffix}")


def load_topology_positions(path: Path) -> tuple[object, object, InputFormat]:
    fmt = detect_input_format(path)
    from openmm.app import PDBFile, PDBxFile

    if fmt == "pdb":
        pdb = PDBFile(str(path))
        return pdb.topology, pdb.positions, fmt
    cif = PDBxFile(str(path))
    return cif.topology, cif.positions, fmt


def build_default_output_path(input_path: Path, mode: str, output_format: InputFormat) -> Path:
    suffix = ".pdb" if output_format == "pdb" else ".cif"
    stem = input_path.stem
    if mode == "minimize":
        return input_path.with_name(f"{stem}_minimized{suffix}")
    if mode == "refine":
        return input_path.with_name(f"{stem}_refined{suffix}")
    return input_path.with_name(f"{stem}_optimized{suffix}")


def write_structure(path: Path, topology: object, positions: object, fmt: InputFormat) -> None:
    from openmm.app import PDBFile, PDBxFile

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if fmt == "pdb":
            PDBFile.writeFile(topology, positions, handle, keepIds=True)
        else:
            PDBxFile.writeFile(topology, positions, handle, keepIds=True)
