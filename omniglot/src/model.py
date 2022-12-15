import torch
from torch import nn
from functools import reduce
import operator

from typing import List, Tuple, Optional

from .data import ALPHABETS

from .config import (
    NUM_WAYS,
    HIDDEN_SIZE,
    NUM_LAYERS,
    ENCODER_OUTPUT_SHAPES,
    ENCODER_DROPOUT,
    ENCODER_VECTOR_SIZES,
    ENCODER_NUM_CHANNELS_PER_GROUP,
)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class ConvBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        max_pool=False,
        max_pool_factor=1.0,
    ):
        super().__init__()
        stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(
                kernel_size=stride,
                stride=stride,
                ceil_mode=False,
            )
            stride = (1, 1)
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(
            out_channels,
            affine=True,
        )
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=1,
            bias=True,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class OmniglotCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        output_size = NUM_WAYS

        self.hidden_size = HIDDEN_SIZE

        layers = [
            ConvBlock(in_channels=1, out_channels=HIDDEN_SIZE, kernel_size=(3, 3))
        ]
        for _ in range(NUM_LAYERS - 1):
            layers.append(
                ConvBlock(
                    in_channels=HIDDEN_SIZE,
                    out_channels=HIDDEN_SIZE,
                    kernel_size=(3, 3),
                )
            )

        self.layers = nn.ModuleList(layers)
        self.classifier = torch.nn.Linear(HIDDEN_SIZE, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)
        self.embeddings: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None

    def update_embeddings(
        self, embeddings: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> None:

        if self.embeddings is not None:
            for embedding in self.embeddings:
                del embedding[0]
                del embedding[1]
                del embedding
            del self.embeddings
        self.embeddings = embeddings

    def forward(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx > 1:
                if self.embeddings is not None:
                    gamma, beta = (
                        self.embeddings[0][idx - 2],
                        self.embeddings[1][idx - 2],
                    )
                    x = (x * gamma) + beta

        x = x.mean(dim=[2, 3])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TaskEncoder(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        num_channels_per_group = [
            ENCODER_NUM_CHANNELS_PER_GROUP for _ in ENCODER_VECTOR_SIZES
        ]

        for idx, vector_size in enumerate(ENCODER_VECTOR_SIZES):
            assert vector_size % num_channels_per_group[idx] == 0

        encoder_layers: List[nn.Module] = []
        for i in range(len(ENCODER_VECTOR_SIZES)):
            encoder_layers.append(
                nn.Linear(
                    in_features=len(ALPHABETS)
                    if i == 0
                    else ENCODER_VECTOR_SIZES[i - 1],
                    out_features=ENCODER_VECTOR_SIZES[i],
                )
            )
            encoder_layers.append(nn.GELU())
            encoder_layers.append(
                nn.GroupNorm(
                    num_channels=ENCODER_VECTOR_SIZES[i],
                    num_groups=ENCODER_VECTOR_SIZES[i] // num_channels_per_group[i],
                )
            )
            encoder_layers.append(nn.Dropout(p=ENCODER_DROPOUT))

        self.initial_encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.initial_encoder(x.unsqueeze(0)).squeeze(0)


class MMAMLEncoder(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        # the lstm here isn't for the timesteps; its for the examples!
        self.initial_encoder = nn.LSTM(
            # we add the target to each timestep
            input_size=(28 * 28 + 1),
            hidden_size=ENCODER_VECTOR_SIZES[-1],
            dropout=ENCODER_DROPOUT,
            batch_first=True,
        )

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        x, y = x
        with torch.no_grad():
            x = x.view(x.size(0), -1)
            x = torch.cat((x, y.expand(1, -1).T), dim=-1)
            x = torch.unsqueeze(x.reshape(x.shape[0], -1), 0)
        x = self.initial_encoder(x)[1][0]
        return x.squeeze(1)


class EncoderHead(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.gamma_layer_names: List[str] = []
        self.beta_layer_names: List[str] = []
        for i in range(len(ENCODER_OUTPUT_SHAPES)):
            gamma_layer = nn.Sequential(
                LinearNd(
                    in_features=ENCODER_VECTOR_SIZES[-1],
                    out_shape=ENCODER_OUTPUT_SHAPES[i],
                ),
                nn.GELU(),
            )
            beta_layer = nn.Sequential(
                LinearNd(
                    in_features=ENCODER_VECTOR_SIZES[-1],
                    out_shape=ENCODER_OUTPUT_SHAPES[i],
                ),
                nn.GELU(),
            )

            gamma_layer_name = f"task_embedding_{i}_gamma"
            beta_layer_name = f"task_embedding_{i}_beta"
            self.__setattr__(gamma_layer_name, gamma_layer)
            self.__setattr__(beta_layer_name, beta_layer)
            self.gamma_layer_names.append(gamma_layer_name)
            self.beta_layer_names.append(beta_layer_name)

        self.dropout = nn.Dropout(p=ENCODER_DROPOUT)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        gamma_outputs: List[torch.Tensor] = []
        beta_outputs: List[torch.Tensor] = []
        for layer_name in self.gamma_layer_names:
            gamma_outputs.append(self.dropout(self.__getattr__(layer_name)(x)))
        for layer_name in self.beta_layer_names:
            beta_outputs.append(self.dropout(self.__getattr__(layer_name)(x)))
        return (gamma_outputs, beta_outputs)


class LinearNd(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: Tuple[int, ...],
        sum_from: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sum_from = sum_from
        self.out_features = self.prod(out_shape)
        self.in_features = in_features
        self.out_shape = out_shape

        self.linear = nn.Linear(in_features, self.out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.sum_from is not None:
            x = (torch.ones_like(x) * self.sum_from) + torch.sigmoid(x)
        if len(self.out_shape) > 1:
            x = x.reshape(self.out_shape)
        return x

    @staticmethod
    def prod(iterable):
        return int(reduce(operator.mul, iterable, 1))


class Encoder(nn.Module):
    def __init__(self, method: str = "timl") -> None:
        super().__init__()
        if method == "mmaml":
            self.initial_encoder = MMAMLEncoder()
        else:
            self.initial_encoder = TaskEncoder()

        self.head = EncoderHead()

    def forward(self, x):
        return self.head(self.initial_encoder(x))
