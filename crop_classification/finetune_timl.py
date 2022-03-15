from pathlib import Path

from cropharvest.datasets import Task, CropHarvest
from cropharvest.countries import get_country_bbox, BBox
from cropharvest.utils import DATAFOLDER_PATH

from dl import load_timl_model, TrainDataLoader, train
from config import DL_TIML

from typing import Union


def _construct_dataset(
    data_folder: Path,
    country: Union[str, BBox],
    target_label: str,
    balance_negative_crops: bool = False,
) -> CropHarvest:
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

    return CropHarvest(data_folder, task)


def main(
    data_folder: Path,
    country: Union[str, BBox],
    target_label: str,
    balance_negative_crops: bool = False,
):

    dataset = _construct_dataset(
        data_folder, country, target_label, balance_negative_crops
    )

    task_info = TrainDataLoader.task_to_task_info(dataset.task)
    num_timesteps = dataset[0][0].shape[0]

    uses_encoder = (data_folder / DL_TIML / "encoder_state_dict.pth").exists()

    model = load_timl_model(
        task_info=task_info,
        input_size=dataset.num_bands
        if uses_encoder
        else dataset.num_bands + task_info.shape[0],
        num_timesteps=num_timesteps,
        model_state_dict_path=data_folder / DL_TIML / "state_dict.pth",
        encoder_state_dict_path=data_folder / DL_TIML / "encoder_state_dict.pth",
    )

    model = train(
        model,
        dataset,
        None,
        task_info_to_concatenate=None,
    )
    model.eval()
    assert dataset.task.bounding_box.name is not None
    model_name = f"{dataset.task.bounding_box.name}_{dataset.task.target_label}.pt"
    model.save(model_name, data_folder / DL_TIML)


if __name__ == "__main__":

    main(DATAFOLDER_PATH, "Kenya", "maize")
