from pathlib import Path

from structopt.io import build_default_output_path, detect_input_format


def test_detect_input_format():
    assert detect_input_format(Path("x.pdb")) == "pdb"
    assert detect_input_format(Path("x.cif")) == "cif"
    assert detect_input_format(Path("x.mmcif")) == "cif"


def test_default_output_path_suffix():
    path = Path("tests/data/OBP5_model_0.cif")
    out = build_default_output_path(path, mode="both", output_format="cif")
    assert out.name.endswith("_optimized.cif")
