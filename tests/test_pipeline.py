from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from structopt.config import OptimizationConfig
from structopt.pipeline import _strip_solvent_and_ions, run_optimization


class _FakeResidue:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeTopology:
    def __init__(self, residues: list[_FakeResidue]) -> None:
        self._residues = residues

    def residues(self):
        return iter(self._residues)


class _FakeModeller:
    def __init__(self, topology: _FakeTopology, positions: object) -> None:
        self.topology = topology
        self.positions = positions

    def delete(self, residues_to_delete: list[_FakeResidue]) -> None:
        self.topology._residues = [
            residue for residue in self.topology._residues if residue not in residues_to_delete
        ]


def test_strip_solvent_and_ions_removes_water_and_ions(monkeypatch):
    openmm_module = ModuleType("openmm")
    openmm_app_module = ModuleType("openmm.app")
    openmm_app_module.Modeller = _FakeModeller
    openmm_module.app = openmm_app_module
    monkeypatch.setitem(sys.modules, "openmm", openmm_module)
    monkeypatch.setitem(sys.modules, "openmm.app", openmm_app_module)

    topology = _FakeTopology(
        [
            _FakeResidue("ALA"),
            _FakeResidue("HOH"),
            _FakeResidue("NA"),
            _FakeResidue("cl-"),
            _FakeResidue("LIG"),
        ]
    )
    positions = object()

    stripped_topology, stripped_positions = _strip_solvent_and_ions(topology, positions)

    assert [residue.name for residue in stripped_topology.residues()] == ["ALA", "LIG"]
    assert stripped_positions is positions


def test_run_optimization_writes_stripped_output(monkeypatch, tmp_path):
    cfg = OptimizationConfig(
        input_path=Path("tests/data/OBP5_model_0.cif"),
        output_path=tmp_path / "optimized.cif",
        mode="minimize",
    )

    fake_state = type(
        "State",
        (),
        {
            "topology": "solvated-topology",
            "positions": "solvated-positions",
            "potential_energy_kj_mol": -12.3,
        },
    )()

    fake_sim = ModuleType("structopt.sim")
    fake_sim.run_minimization = lambda _cfg, _modeller: fake_state
    fake_sim.run_refinement_npt = lambda _cfg, _modeller: (_cfg, _modeller)
    monkeypatch.setitem(sys.modules, "structopt.sim", fake_sim)

    monkeypatch.setattr("structopt.pipeline.validate_input_exists", lambda _path: None)
    monkeypatch.setattr("structopt.pipeline.detect_input_format", lambda _path: "cif")
    monkeypatch.setattr("structopt.pipeline.prepare_structure", lambda _cfg: "prepared-modeller")
    monkeypatch.setattr(
        "structopt.pipeline._strip_solvent_and_ions",
        lambda _topology, _positions: ("dry-topology", "dry-positions"),
    )

    written = {}

    def _fake_write(path, topology, positions, fmt):
        written["path"] = path
        written["topology"] = topology
        written["positions"] = positions
        written["fmt"] = fmt

    monkeypatch.setattr("structopt.pipeline.write_structure", _fake_write)

    result = run_optimization(cfg)

    assert result.output_path == cfg.output_path
    assert result.final_energy_kj_mol == -12.3
    assert written["path"] == cfg.output_path
    assert written["topology"] == "dry-topology"
    assert written["positions"] == "dry-positions"
    assert written["fmt"] == "cif"
