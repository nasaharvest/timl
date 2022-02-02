from torch import nn
import torch

from typing import List, Tuple, Union, Optional


class TaskEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        encoder_vector_sizes: List[int],
        num_bands: int,
        num_hidden_layers: int,
        hidden_vector_size: int,
        encoder_dropout: float,
        num_timesteps: int,
        num_channels_per_group: Union[int, List[int]] = 16,
    ) -> None:
        super().__init__()

        if isinstance(num_channels_per_group, int):
            num_channels_per_group = [
                num_channels_per_group for _ in encoder_vector_sizes
            ]

        for idx, vector_size in enumerate(encoder_vector_sizes):
            assert vector_size % num_channels_per_group[idx] == 0

        encoder_layers: List[nn.Module] = []
        for i in range(len(encoder_vector_sizes)):
            encoder_layers.append(
                nn.Linear(
                    in_features=input_size if i == 0 else encoder_vector_sizes[i - 1],
                    out_features=encoder_vector_sizes[i],
                )
            )
            encoder_layers.append(nn.GELU())
            encoder_layers.append(
                nn.GroupNorm(
                    num_channels=encoder_vector_sizes[i],
                    num_groups=encoder_vector_sizes[i] // num_channels_per_group[i],
                )
            )
            encoder_layers.append(nn.Dropout(p=encoder_dropout))

        self.initial_encoder = nn.Sequential(*encoder_layers)

        self.gamma_layer_names: List[str] = []
        self.beta_layer_names: List[str] = []
        for i in range(num_hidden_layers + 1):
            # these will want outputs of shape [hidden_vector_size, hidden_vector_size]
            out_features = num_bands if i == 0 else hidden_vector_size
            if i == 0:
                # the nonlinearity is captured in the linear3d layer
                gamma_layer = Linear3d(
                    in_features=encoder_vector_sizes[-1],
                    out_shape=(num_timesteps, out_features),
                    sum_from=1,
                )
                beta_layer = Linear3d(
                    in_features=encoder_vector_sizes[-1],
                    out_shape=(num_timesteps, out_features),
                    sum_from=0,
                )
            else:
                gamma_layer = nn.Sequential(
                    nn.Linear(
                        in_features=encoder_vector_sizes[-1], out_features=out_features
                    ),
                    nn.GELU() if i > 0 else nn.Sigmoid(),
                )
                beta_layer = nn.Sequential(
                    nn.Linear(
                        in_features=encoder_vector_sizes[-1], out_features=out_features
                    ),
                    nn.GELU(),
                )

            gamma_layer_name = f"task_embedding_{i}_gamma"
            beta_layer_name = f"task_embedding_{i}_beta"
            self.__setattr__(gamma_layer_name, gamma_layer)
            self.__setattr__(beta_layer_name, beta_layer)
            self.gamma_layer_names.append(gamma_layer_name)
            self.beta_layer_names.append(beta_layer_name)

        self.dropout = nn.Dropout(p=encoder_dropout)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        x = self.initial_encoder(x.unsqueeze(0)).squeeze(0)

        gamma_outputs: List[torch.Tensor] = []
        beta_outputs: List[torch.Tensor] = []
        for layer_name in self.gamma_layer_names:
            gamma_outputs.append(self.dropout(self.__getattr__(layer_name)(x)))
        for layer_name in self.beta_layer_names:
            beta_outputs.append(self.dropout(self.__getattr__(layer_name)(x)))
        return (gamma_outputs, beta_outputs)


class Linear3d(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_shape: Tuple[int, int],
        sum_from: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.sum_from = sum_from
        self.out_features = out_shape[0] * out_shape[1]
        self.in_features = in_features
        self.out_shape = out_shape

        self.linear = nn.Linear(in_features, self.out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.sum_from is not None:
            x = (torch.ones_like(x) * self.sum_from) + torch.sigmoid(x)
        return x.reshape(self.out_shape)
