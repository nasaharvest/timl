import torch
import numpy as np
from pathlib import Path

from dl.lstm import Classifier
from dl.encoder import TaskEncoder


def test_jit(tmp_path):

    input_size, num_timesteps, input_task_info_size = 18, 12, 13

    # train a model
    model = Classifier(
        input_size=input_size,
        classifier_base_layers=1,
        num_classification_layers=2,
        classifier_dropout=0.2,
        classifier_vector_size=128,
    )
    model.load_state_dict(
        torch.load(Path(__file__).parent / "test_data/timl/state_dict.pth")
    )

    model.eval()

    encoder = TaskEncoder(
        input_size=input_task_info_size,
        encoder_vector_sizes=[32, 64, 128],
        encoder_dropout=0.2,
        num_bands=input_size,
        num_hidden_layers=2,
        hidden_vector_size=128,
        num_timesteps=num_timesteps,
    )
    encoder.load_state_dict(
        torch.load(Path(__file__).parent / "test_data/timl/encoder_state_dict.pth")
    )

    encoder.eval()

    task_info = torch.rand(input_task_info_size)

    model.update_embeddings(encoder(task_info))

    model.save("timl", tmp_path)

    jit_model = torch.jit.load(tmp_path / f"timl.pt")
    jit_model.eval()

    x = torch.rand(5, num_timesteps, input_size)

    with torch.no_grad():
        y_from_jit_model = jit_model(x).numpy()
        y_from_state_dict_model = model(x).numpy()

    assert np.allclose(y_from_jit_model, y_from_state_dict_model)
