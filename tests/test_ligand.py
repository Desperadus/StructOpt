from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from structopt.config import OptimizationConfig
from structopt.ligand import register_gaff_template


class _Residue:
    def __init__(self, name: str):
        self.name = name


class _Topology:
    def __init__(self, residue_names: list[str]):
        self._residue_names = residue_names

    def residues(self):
        return (_Residue(name) for name in self._residue_names)


class _ForceField:
    def __init__(self):
        self.generator = None

    def registerTemplateGenerator(self, generator):
        self.generator = generator


def _install_fake_gaff(monkeypatch, recorder: dict) -> None:
    class _FakeGAFFTemplateGenerator:
        def __init__(self, **kwargs):
            recorder["kwargs"] = kwargs
            self.generator = object()

    fake_generators = types.ModuleType("openmmforcefields.generators")
    fake_generators.GAFFTemplateGenerator = _FakeGAFFTemplateGenerator

    fake_openmmforcefields = types.ModuleType("openmmforcefields")
    fake_openmmforcefields.generators = fake_generators

    monkeypatch.setitem(sys.modules, "openmmforcefields", fake_openmmforcefields)
    monkeypatch.setitem(sys.modules, "openmmforcefields.generators", fake_generators)


def test_register_gaff_template_skips_when_ligand_name_absent():
    cfg = OptimizationConfig(input_path=Path("tests/data/OBP5_model_0.cif"), ligand_name="LIG1")
    ff = _ForceField()
    registered = register_gaff_template(ff, cfg, _Topology(["ALA", "GLY"]))
    assert registered is False
    assert ff.generator is None


def test_register_gaff_template_requires_sdf_for_detected_ligand():
    cfg = OptimizationConfig(input_path=Path("tests/data/OBP5_model_0.cif"), ligand_name="LIG1")
    ff = _ForceField()
    with pytest.raises(RuntimeError, match="automatic extraction"):
        register_gaff_template(ff, cfg, _Topology(["ALA", "LIG1"]))


def test_register_gaff_template_uses_loaded_sdf_molecules(monkeypatch, tmp_path):
    sdf = tmp_path / "ligand.sdf"
    sdf.write_text("fake", encoding="utf-8")
    cfg = OptimizationConfig(
        input_path=Path("tests/data/OBP5_model_0.cif"),
        ligand_name="LIG1",
        ligand_sdf=sdf,
    )
    ff = _ForceField()

    recorder: dict[str, object] = {}
    _install_fake_gaff(monkeypatch, recorder)
    monkeypatch.setattr(
        "structopt.ligand._load_openff_molecules_from_sdf",
        lambda path: ["mol-from-sdf", path],
    )

    registered = register_gaff_template(ff, cfg, _Topology(["LIG1"]))

    assert registered is True
    assert ff.generator is not None
    assert recorder["kwargs"]["forcefield"] == "gaff-2.11"
    assert recorder["kwargs"]["molecules"] == ["mol-from-sdf", str(sdf)]


def test_register_gaff_template_auto_extracts_ligand_sdf(monkeypatch, tmp_path):
    cfg = OptimizationConfig(input_path=Path("tests/data/OBP5_model_0.cif"), ligand_name="LIG1")
    ff = _ForceField()
    extracted_sdf = tmp_path / "auto_ligand.sdf"
    extracted_sdf.write_text("fake", encoding="utf-8")

    recorder: dict[str, object] = {}
    _install_fake_gaff(monkeypatch, recorder)

    monkeypatch.setattr(
        "structopt.ligand._extract_ligand_sdf_from_topology",
        lambda **_: extracted_sdf,
    )
    monkeypatch.setattr(
        "structopt.ligand._load_openff_molecules_from_sdf",
        lambda path: ["auto-mol", path],
    )

    registered = register_gaff_template(ff, cfg, _Topology(["LIG1"]), positions=object())

    assert registered is True
    assert cfg.ligand_sdf == extracted_sdf
    assert recorder["kwargs"]["molecules"] == ["auto-mol", str(extracted_sdf)]
