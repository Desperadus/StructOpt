from pathlib import Path

import pytest

from structopt.config import OptimizationConfig
from structopt.pipeline import run_optimization


def _deps_available() -> bool:
    try:
        import openmm  # noqa: F401
        import pdbfixer  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.slow
def test_pipeline_smoke_geraniol(tmp_path):
    if not _deps_available():
        pytest.skip("OpenMM/PDBFixer not installed in test environment")

    inp = Path("tests/data/geraniol_model_0.cif")
    out = tmp_path / "geraniol_optimized.cif"
    cfg = OptimizationConfig(
        input_path=inp,
        output_path=out,
        mode="minimize",
        strip_heterogens=True,
        minimize_max_iter=50,
    )
    result = run_optimization(cfg)
    assert result.output_path.exists()
    text = result.output_path.read_text(encoding="utf-8")
    assert "ATOM" in text or "HETATM" in text


@pytest.mark.slow
def test_pipeline_smoke_obp5_both_mode_short_refinement(tmp_path):
    if not _deps_available():
        pytest.skip("OpenMM/PDBFixer not installed in test environment")

    inp = Path("tests/data/OBP5_model_0.cif")
    out = tmp_path / "obp5_optimized.cif"
    cfg = OptimizationConfig(
        input_path=inp,
        output_path=out,
        mode="both",
        minimize_max_iter=200,
        equil_steps=50,
        npt_steps=0,
    )
    result = run_optimization(cfg)
    assert result.output_path.exists()
    assert result.minimized_energy_kj_mol is not None
    assert result.refined_energy_kj_mol is not None
