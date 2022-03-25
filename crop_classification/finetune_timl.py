from pathlib import Path
import torch
from torch import nn
import numpy as np

from cropharvest.datasets import Task, CropHarvest
from cropharvest.countries import get_country_bbox, BBox
from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.engineer import TestInstance

from dl import load_timl_model, TrainDataLoader, train
from config import DL_TIML

from typing import Union, Tuple


def _construct_datasets(
    data_folder: Path,
    country: Union[str, BBox],
    target_label: str,
    balance_negative_crops: bool = False,
    val_ratio: float = 0.2,
) -> Tuple[CropHarvest, CropHarvest]:
    if isinstance(country, BBox):
        assert (
            country.name is not None
        ), "country BBox requires a `name` attribute to save the model"
        country_bbox = country
    else:
        # get the largest polygon for the country name, since this will
        # likely be the primary polygon
        country_bbox = max(get_country_bbox(country), key=lambda c: c.area)

    task = Task(
        country_bbox, target_label, balance_negative_crops=balance_negative_crops
    )

    kwargs = {"root": data_folder, "task": task, "val_ratio": val_ratio}

    return CropHarvest(**kwargs, is_val=False), CropHarvest(**kwargs, is_val=True)


def _evaluate_on_dataset(model: nn.Module, dataset: CropHarvest):

    # construct a test instance
    x, y = dataset.as_array(num_samples=-1)
    test_instance = TestInstance(x=x, y=y, lats=np.ones_like(y), lons=np.ones_like(y))
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(x)).numpy()

    return test_instance.evaluate_predictions(preds)


def main(
    data_folder: Path,
    country: Union[str, BBox],
    target_label: str,
    balance_negative_crops: bool = False,
    val_ratio: float = 0.2,
):

    train_dataset, val_dataset = _construct_datasets(
        data_folder, country, target_label, balance_negative_crops, val_ratio
    )

    task_info = TrainDataLoader.task_to_task_info(train_dataset.task)
    num_timesteps = train_dataset[0][0].shape[0]

    uses_encoder = (data_folder / DL_TIML / "encoder_state_dict.pth").exists()

    model = load_timl_model(
        task_info=task_info,
        input_size=train_dataset.num_bands
        if uses_encoder
        else train_dataset.num_bands + task_info.shape[0],
        num_timesteps=num_timesteps,
        model_state_dict_path=data_folder / DL_TIML / "state_dict.pth",
        encoder_state_dict_path=data_folder / DL_TIML / "encoder_state_dict.pth",
        normalizing_dict=train_dataset.normalizing_dict,
    )

    model = train(
        model,
        train_dataset,
        None,
        task_info_to_concatenate=None,
    )
    model.eval()

    print(_evaluate_on_dataset(model, val_dataset))

    assert train_dataset.task.bounding_box.name is not None
    model_name = (
        f"{train_dataset.task.bounding_box.name}_{train_dataset.task.target_label}.pt"
    )
    model.save(model_name, data_folder / DL_TIML)


if __name__ == "__main__":

    main(DATAFOLDER_PATH, "Malawi", "maize")
