import torch

from dl.encoder import MMAMLEncoder


def test_mmaml_encoder():

    batch_size, bands, timesteps = 10, 18, 12
    encoder = MMAMLEncoder(
        num_bands=bands,
        num_hidden_layers=2,
        encoder_hidden_vector_size=128,
        classifier_hidden_vector_size=128,
        encoder_dropout=0.2,
        num_timesteps=timesteps,
    )
    x = torch.ones((batch_size, timesteps, bands))
    y = torch.ones(batch_size)

    with torch.no_grad():
        _ = encoder(x, y)
