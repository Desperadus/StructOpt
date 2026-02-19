from pathlib import Path

import pytest

from structopt.config import OptimizationConfig


def test_config_defaults():
    cfg = OptimizationConfig(input_path=Path("tests/data/OBP5_model_0.cif"))
    assert cfg.ph == 7.2
    assert cfg.mode == "both"
    assert cfg.ligand_name == "LIG1"
    assert cfg.minimize_solvent == "explicit"


def test_bad_extension_fails():
    with pytest.raises(ValueError):
        OptimizationConfig(input_path=Path("bad.xyz"))


def test_refine_requires_steps():
    with pytest.raises(ValueError):
        OptimizationConfig(
            input_path=Path("tests/data/OBP5_model_0.cif"),
            mode="refine",
            equil_steps=0,
            npt_steps=0,
        )


def test_ligand_sdf_must_exist(tmp_path):
    missing = tmp_path / "missing.sdf"
    with pytest.raises(ValueError):
        OptimizationConfig(
            input_path=Path("tests/data/OBP5_model_0.cif"),
            ligand_sdf=missing,
        )


def test_ligand_sdf_must_be_sdf(tmp_path):
    bad = tmp_path / "ligand.mol2"
    bad.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        OptimizationConfig(
            input_path=Path("tests/data/OBP5_model_0.cif"),
            ligand_sdf=bad,
        )
