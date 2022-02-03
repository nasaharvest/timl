from shutil import copytree
from pathlib import Path

from src.timl import Learner


TIML_modelname = "TIML_lstm_2011"


def test_load(tmp_path):

    copytree(Path(__file__).parent / "test_data", tmp_path / "data")

    # check everything loads okay
    learner = Learner.load_from_folder(tmp_path / "data", model_name=TIML_modelname)
    assert learner.encoder is not None
