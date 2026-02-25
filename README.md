# StructOpt

`structopt` is a Python package and CLI to quickly clean and optimize protein structures from
predicted or experimental sources (`.pdb`, `.cif`, `.mmcif`).

It combines:
- `PDBFixer` for structure cleanup.
- OpenMM minimization (implicit solvent by default).
- Optional short MD refinement.
- Optional GAFF small-molecule handling (for ligands like `LIG1`).

## Install (uv)

```bash
uv sync
```

To include dev tools:

```bash
uv sync --group dev
```

For GAFF ligand support, install `openmmforcefields`, `rdkit`, and AmberTools
in the active environment.
If your package index cannot resolve those, use mamba:

```bash
mamba env create -f environment.mamba.yml
mamba activate structopt
pip install -e .
```

## CLI

Run the command below to see all the possible options you can have for your optimisation
```bash
uv run structopt optimize --help
```


Example:
```bash
uv run structopt optimize tests/data/geraniol_model_0.cif
```

Useful options:

```bash
uv run structopt optimize tests/data/geraniol_model_0.cif \
  --mode both \
  --ph 7.2 \
  --ligand-name LIG1 \
  --npt-steps 50000 \
  --temperature 300 \
  --pressure 1.0
```

If GAFF ligand parametrization cannot be inferred from the structure alone, provide an SDF:

```bash
uv run structopt optimize tests/data/geraniol_model_0.cif \
  --ligand-name LIG1 \
  --ligand-sdf path/to/ligand.sdf
```

## Notes

- Default cleanup adds hydrogens at pH `7.2`.
- Minimization uses explicit by default or possibly implicit solvent (`gbn2` or `obc2`).
- MD refinement uses short explicit-solvent NPT by default.
- For GAFF workflows, install AmberTools in your environment.
