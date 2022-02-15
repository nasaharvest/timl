from shutil import copytree
from pathlib import Path

import pytest

from src.timl import Learner


TIML_modelname = "TIML_lstm_2011"


def test_load(tmp_path):

    copytree(Path(__file__).parent / "test_data", tmp_path / "data")

    # check everything loads okay
    learner = Learner.load_from_folder(tmp_path / "data", model_name=TIML_modelname)
    assert learner.encoder is not None


@pytest.mark.integration
def test_load_all():

    DATA_FOLDER = Path(__file__).parent.parent / "data"

    for model in ["lstm", "cnn"]:
        model_folder = f"yield_{model}_timl"
        for year in [2011, 2012, 2013, 2014, 2015]:
            model_name = f"TIML_{model}_{year}"

            learner = Learner.load_from_folder(
                DATA_FOLDER,
                model_name=model_name,
                model_folder=DATA_FOLDER / model_folder / model_name,
            )
            assert learner.encoder is not None
