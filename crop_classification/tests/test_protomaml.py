import torch


from dl.protolstm import ProtoClassifier
from dl.encoder import TaskEncoder


def test_proto_classifier():

    input_size, num_timesteps = 18, 12
    x_in = torch.ones(10, num_timesteps, input_size).float()
    x_support = torch.ones(5, num_timesteps, input_size).float()
    y = torch.tensor([0, 1, 1, 0, 0])

    model = ProtoClassifier(input_size=input_size)

    output = model(x_in, x_support, y)
    assert len(output) == 10


def test_proto_classifier_with_encoder():

    input_size, num_timesteps = 18, 12
    x_in = torch.ones(10, num_timesteps, input_size).float()
    x_support = torch.ones(5, num_timesteps, input_size).float()
    y = torch.tensor([0, 1, 1, 0, 0])

    model = ProtoClassifier(input_size=input_size, classifier_vector_size=128)

    encoder = TaskEncoder(
        input_size=8,
        encoder_vector_sizes=[8],
        num_bands=input_size,
        num_hidden_layers=0,
        hidden_vector_size=128,
        encoder_dropout=0.2,
        num_timesteps=num_timesteps,
        num_channels_per_group=4,
    )

    task_encodings = encoder(torch.ones(8).float())
    model.update_embeddings(task_encodings)
    output = model(x_in, x_support, y)
    assert len(output) == 10
