from tqdm import tqdm
from torch.optim import SGD
import torch
from torch import nn

from cropharvest.utils import sample_with_memory
from cropharvest.datasets import CropHarvest

from .lstm import Classifier
from .utils import concatenate_task_info

from typing import Optional, List


def train(
    classifier: Classifier,
    dataset: CropHarvest,
    sample_size: Optional[int],
    num_grad_steps: int = 250,
    learning_rate: float = 0.001,
    k: int = 10,
    task_info_to_concatenate: Optional[torch.tensor] = None,
    protomaml: bool = False,
) -> Classifier:
    r"""
    Train the classifier on the dataset.

    :param classifier: The classifier to train
    :param dataset: The dataset to train the classifier on
    :param sample_size: The number of training samples to use. If None, all training data
        in the dataset will be used
    :param num_grad_steps: The number of gradient steps to train for
    :param learning rate: The learning rate to use with the SGD optimizer
    :param k: A batch will have size k*2, with k positive and k negative examples
    """

    opt = SGD(classifier.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss(reduction="mean")

    input_dict = {}

    if sample_size is not None:
        train_batch_total = dataset.sample(sample_size // 2, deterministic=True)
        state: List[int] = []
        if protomaml:
            input_dict.update(
                {
                    "support_x": torch.from_numpy(train_batch_total[0]).float(),
                    "support_y": torch.from_numpy(train_batch_total[1]).float(),
                }
            )

    for i in tqdm(range(num_grad_steps)):
        if i != 0:
            classifier.train()
            opt.zero_grad()

        if sample_size is not None:
            assert train_batch_total is not None
            indices, state = sample_with_memory(
                list(range(train_batch_total[0].shape[0])), k * 2, state
            )
            train_x, train_y = (
                train_batch_total[0][indices],
                train_batch_total[1][indices],
            )
        else:
            train_x, train_y = dataset.sample(k, deterministic=False)
            if protomaml:
                support_x, support_y = dataset.sample(k, deterministic=False)
                input_dict.update(
                    {
                        "support_x": torch.from_numpy(support_x).float(),
                        "support_y": torch.from_numpy(support_y).float(),
                    }
                )

        train_x_t = torch.from_numpy(train_x).float()
        if task_info_to_concatenate is not None:
            train_x_t = concatenate_task_info(train_x_t, task_info_to_concatenate)
        preds = classifier(train_x_t, **input_dict).squeeze(dim=1)
        loss = loss_fn(preds, torch.from_numpy(train_y).float())

        loss.backward()
        opt.step()
    return classifier
