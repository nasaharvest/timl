import numpy as np
import sys
import torch
from pathlib import Path
from cropharvest.inference import Inference

# Needed for running pytest without python -m pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from dl.timl import load_timl_model

test_tif = Path(__file__).parent / "test_data/373-croplands_2016-02-07_2017-02-01.tif"
model_state_dict_path = Path(__file__).parent / "test_data/timl/state_dict.pth"
encoder_state_dict_path = (
    Path(__file__).parent / "test_data/timl/encoder_state_dict.pth"
)


def test_ckpt_jit_predictions_match(tmp_path):

    input_size, num_timesteps = 18, 12
    task_info = torch.rand(13)

    model = load_timl_model(
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
        model_state_dict_path=model_state_dict_path,
        encoder_state_dict_path=encoder_state_dict_path,
    )
    model.eval()

    normalizing_dict = {"mean": [1, 2, 3], "std": [4, 5, 6]}
    model.normalizing_dict = normalizing_dict

    model.save("timl", tmp_path)

    jit_model = torch.jit.load(tmp_path / f"timl.pt")
    jit_model.eval()

    assert jit_model.normalizing_dict is not None
    for key, val in normalizing_dict.items():
        assert jit_model.normalizing_dict[key] == val

    x = torch.rand(5, num_timesteps, input_size)

    with torch.no_grad():
        y_from_jit_model = jit_model(x).numpy()
        y_from_state_dict_model = model(x).numpy()

    assert np.allclose(y_from_jit_model, y_from_state_dict_model)


def test_ckpt_inference_from_file():
    input_size, num_timesteps = 18, 12
    task_info = torch.rand(13)

    model = load_timl_model(
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
        model_state_dict_path=model_state_dict_path,
        encoder_state_dict_path=encoder_state_dict_path,
    )
    model.eval()

    inference = Inference(model=model, normalizing_dict=None)
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

    model = load_timl_model(
        task_info=task_info,
        input_size=input_size,
        num_timesteps=num_timesteps,
        model_state_dict_path=model_state_dict_path,
        encoder_state_dict_path=encoder_state_dict_path,
    )
    model.eval()
    model.save("timl", tmp_path)
    jit_model = torch.jit.load(tmp_path / f"timl.pt")
    jit_model.eval()

    inference = Inference(model=jit_model, normalizing_dict=None)
    xr_predictions = inference.run(local_path=test_tif)

    # Check size
    assert xr_predictions.dims["lat"] == 17
    assert xr_predictions.dims["lon"] == 29

    # Check all predictions between 0 and 1
    assert xr_predictions.min() >= 0
    assert xr_predictions.max() <= 1
