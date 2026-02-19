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
