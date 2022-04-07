from .lstm import UnrolledLSTM


from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict, Tuple, Optional, List


class ProtoClassifier(nn.Module):
    r"""
    An LSTM based model to predict the presence of cropland in a pixel.
    :param input_size: The number of input bands passed to the model. The
        input vector is expected to be of shape [batch_size, timesteps, bands]
    """

    def __init__(
        self,
        input_size: int,
        classifier_vector_size: int = 128,
        classifier_dropout: float = 0.2,
        classifier_base_layers: int = 1,
    ) -> None:
        super().__init__()

        self.base = nn.ModuleList(
            [
                UnrolledLSTM(
                    input_size=input_size if i == 0 else classifier_vector_size,
                    hidden_size=classifier_vector_size,
                    dropout=classifier_dropout,
                    batch_first=True,
                )
                for i in range(classifier_base_layers)
            ]
        )

        self.batchnorm = nn.BatchNorm1d(
            num_features=classifier_vector_size, affine=False
        )

        self.embeddings: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None
        self.normalizing_dict: Optional[Dict[str, List[float]]] = None

    def update_embeddings(
        self, embeddings: Tuple[List[torch.Tensor], List[torch.Tensor]]
    ) -> None:

        assert len(embeddings[0]) == len(embeddings[1]) == 1, (
            f"Expected 1 embedding for each block."
            f"Got {len(embeddings[0])} embeddings"
        )
        self.embeddings = embeddings

    @staticmethod
    def _compute_prototypes(
        embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes class prototypes over the last dimension of embeddings.
        Args:
        embeddings: Tensor of examples of shape [num_examples, embedding_size].
        labels: Tensor of one-hot encoded labels of shape [num_examples,
            num_classes].
        Returns:
        prototypes: Tensor of class prototypes of shape [num_classes,
        embedding_size].
        """
        # Sums each class' embeddings. [num classes, embedding size].
        # we assume only the binary case is considered
        labels = F.one_hot(labels.long(), num_classes=2)
        class_sums = torch.sum(
            torch.unsqueeze(labels, 2) * torch.unsqueeze(embeddings, 1), 0
        )

        # The prototype of each class is the averaged embedding of its examples.
        class_num_images = torch.sum(torch.unsqueeze(labels, 2), 0)
        return class_sums / class_num_images

    @staticmethod
    def _proto_weights_and_bias(
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = (2 * prototypes).T
        bias = torch.square(torch.norm(prototypes, dim=1))
        return weights, bias

    def _make_embeddings_for_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.embeddings is not None:
            gamma, beta = self.embeddings[0][0], self.embeddings[1][0]
            x = (x * gamma) + beta

        for _, lstm in enumerate(self.base):
            x, (hn, _) = lstm(x)
            x = x[:, 0, :, :]

        return self.batchnorm(hn[-1, :, :])

    def forward(
        self, x: torch.Tensor, support_x: torch.Tensor, support_y: torch.Tensor
    ) -> torch.Tensor:
        x = self._make_embeddings_for_features(x)

        support_x = self._make_embeddings_for_features(support_x)
        prototypes = self._compute_prototypes(support_x, support_y)
        proto_w, proto_b = self._proto_weights_and_bias(prototypes)

        # keep the 1st dimension of the predictions (i.e. the predictions
        # about the positive class) so that we can treat this as binary
        # cross entropy
        return torch.softmax(torch.matmul(x, proto_w) + proto_b, 1)[:, 1]

    def save(self, model_name: str, savepath: Path):
        self.eval()
        sm = torch.jit.script(self)
        model_path = savepath / f"{model_name}.pt"
        if model_path.exists():
            model_path.unlink()
        sm.save(model_path)
