import numpy as np
import sys
import torch
from pathlib import Path

# Needed for running pytest without python -m pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from dl.lstm import Classifier
from dl.encoder import TaskEncoder
from inference import Inference

test_tif = Path(__file__).parent / "test_data/373-croplands_2016-02-07_2017-02-01.tif"
model_state_dict_path = Path(__file__).parent / "test_data/timl/state_dict.pth"
encoder_state_dict_path = (
    Path(__file__).parent / "test_data/timl/encoder_state_dict.pth"
)


# TODO this might make sense to put into deeplearning.py with more parameters
def load_model(
    task_info: np.ndarray,
    input_size: int,
    num_timesteps: int,
    model_state_dict_path: Path = model_state_dict_path,
    encoder_state_dict_path: Path = encoder_state_dict_path,
):
    model = Classifier(
        input_size=input_size,
        classifier_base_layers=1,
        num_classification_layers=2,
        classifier_dropout=0.2,
        classifier_vector_size=128,
    )
    model.load_state_dict(torch.load(model_state_dict_path))

    model.eval()

    encoder = TaskEncoder(
        input_size=task_info.shape[0],
        encoder_vector_sizes=[32, 64, 128],
        encoder_dropout=0.2,
        num_bands=input_size,
        num_hidden_layers=2,
        hidden_vector_size=128,
        num_timesteps=num_timesteps,
    )
    encoder.load_state_dict(torch.load(encoder_state_dict_path))

    encoder.eval()

    model.update_embeddings(encoder(task_info))

    return model


def test_ckpt_jit_predictions_match(tmp_path):

    input_size, num_timesteps = 18, 12
    task_info = torch.rand(13)

    model = load_model(
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
    )

    model.save("timl", tmp_path)

    jit_model = torch.jit.load(tmp_path / f"timl.pt")
    jit_model.eval()

    x = torch.rand(5, num_timesteps, input_size)

    with torch.no_grad():
        y_from_jit_model = jit_model(x).numpy()
        y_from_state_dict_model = model(x).numpy()

    assert np.allclose(y_from_jit_model, y_from_state_dict_model)


def test_ckpt_inference_from_file():
    input_size, num_timesteps = 18, 12
    task_info = torch.rand(13)

    model = load_model(
        model_state_dict_path=model_state_dict_path,
        encoder_state_dict_path=encoder_state_dict_path,
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
    )

    inference = Inference(model=model)
    xr_predictions = inference.run(local_path=test_tif)

    # Check size
    assert xr_predictions.dims["lat"] == 17
    assert xr_predictions.dims["lon"] == 29

    # Check all predictions between 0 and 1
    assert xr_predictions.min() >= 0
    assert xr_predictions.max() <= 1


def test_jit_inference_from_file(tmp_path):
    input_size, num_timesteps = 18, 12
    task_info = torch.rand(13)

    model = load_model(
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
    )
    model.save("timl", tmp_path)
    jit_model = torch.jit.load(tmp_path / f"timl.pt")
    jit_model.eval()

    inference = Inference(model=jit_model)
    xr_predictions = inference.run(local_path=test_tif)

    # Check size
    assert xr_predictions.dims["lat"] == 17
    assert xr_predictions.dims["lon"] == 29

    # Check all predictions between 0 and 1
    assert xr_predictions.min() >= 0
    assert xr_predictions.max() <= 1
