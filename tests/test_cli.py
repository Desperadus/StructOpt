from pathlib import Path

from typer.testing import CliRunner

from structopt.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "optimize" in result.stdout


def test_cli_optimize_parsing(monkeypatch):
    called = {}

    def fake_run(config):
        called["cfg"] = config
        return type(
            "Result",
            (),
            {
                "output_path": Path("out.cif"),
                "final_energy_kj_mol": -1.0,
                "minimized_energy_kj_mol": -2.0,
                "refined_energy_kj_mol": -1.0,
                "post_refined_energy_kj_mol": None,
            },
        )()

    monkeypatch.setattr("structopt.cli.run_optimization", fake_run)
    result = runner.invoke(
        app,
        [
            "optimize",
            "tests/data/OBP5_model_0.cif",
            "--mode",
            "minimize",
            "--ph",
            "6.8",
            "--ligand-name",
            "LIG1",
        ],
    )
    assert result.exit_code == 0
    assert "Output:" in result.stdout
    assert called["cfg"].mode == "minimize"
    assert called["cfg"].ph == 6.8
    assert called["cfg"].minimize_solvent == "explicit"
    assert called["cfg"].refine_solvent == "explicit"


def test_cli_refine_solvent_implicit(monkeypatch):
    called = {}

    def fake_run(config):
        called["cfg"] = config
        return type(
            "Result",
            (),
            {
                "output_path": Path("out.cif"),
                "final_energy_kj_mol": -1.0,
                "minimized_energy_kj_mol": None,
                "refined_energy_kj_mol": None,
                "post_refined_energy_kj_mol": None,
            },
        )()

    monkeypatch.setattr("structopt.cli.run_optimization", fake_run)
    result = runner.invoke(
        app,
        [
            "optimize",
            "tests/data/OBP5_model_0.cif",
            "--mode",
            "minimize",
            "--refine-solvent",
            "implicit",
        ],
    )
    assert result.exit_code == 0
    assert called["cfg"].refine_solvent == "implicit"


def test_cli_remove_h_flag(monkeypatch):
    called = {}

    def fake_run(config):
        called["cfg"] = config
        return type(
            "Result",
            (),
            {
                "output_path": Path("out.cif"),
                "final_energy_kj_mol": -1.0,
                "minimized_energy_kj_mol": -2.0,
                "refined_energy_kj_mol": -1.0,
                "post_refined_energy_kj_mol": None,
            },
        )()

    monkeypatch.setattr("structopt.cli.run_optimization", fake_run)
    result = runner.invoke(app, ["optimize", "tests/data/OBP5_model_0.cif", "--removeH"])

    assert result.exit_code == 0
    assert called["cfg"].remove_h is True


def test_cli_optimize_batch_from_directory(monkeypatch, tmp_path):
    (tmp_path / "a.cif").write_text("x", encoding="utf-8")
    (tmp_path / "b.pdb").write_text("x", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("x", encoding="utf-8")
    called = []

    def fake_run(config):
        called.append(config.input_path.name)
        return type(
            "Result",
            (),
            {
                "output_path": Path(f"out_{config.input_path.name}.cif"),
                "final_energy_kj_mol": -1.0,
                "minimized_energy_kj_mol": -2.0,
                "refined_energy_kj_mol": -1.0,
                "post_refined_energy_kj_mol": None,
            },
        )()

    monkeypatch.setattr("structopt.cli.run_optimization", fake_run)
    result = runner.invoke(app, ["optimize", str(tmp_path)])

    assert result.exit_code == 0
    assert called == ["a.cif", "b.pdb"]


def test_cli_optimize_batch_from_wildcard(monkeypatch, tmp_path):
    (tmp_path / "a.cif").write_text("x", encoding="utf-8")
    (tmp_path / "b.cif").write_text("x", encoding="utf-8")
    called = []

    def fake_run(config):
        called.append(config.input_path.name)
        return type(
            "Result",
            (),
            {
                "output_path": Path(f"out_{config.input_path.name}.cif"),
                "final_energy_kj_mol": -1.0,
                "minimized_energy_kj_mol": -2.0,
                "refined_energy_kj_mol": -1.0,
                "post_refined_energy_kj_mol": None,
            },
        )()

    monkeypatch.setattr("structopt.cli.run_optimization", fake_run)
    result = runner.invoke(app, ["optimize", str(tmp_path / "*.cif")])

    assert result.exit_code == 0
    assert called == ["a.cif", "b.cif"]


def test_cli_optimize_batch_rejects_single_output_file(monkeypatch):
    monkeypatch.setattr("structopt.cli.run_optimization", lambda _cfg: None)
    result = runner.invoke(
        app,
        [
            "optimize",
            "tests/data/OBP5_model_0.cif",
            "tests/data/geraniol_model_0.cif",
            "--output",
            "out.cif",
        ],
    )

    assert result.exit_code == 1
    assert (
        "For batch optimization, --output must be omitted or point to a directory."
        in result.stderr
    )
