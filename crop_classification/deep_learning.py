import torch

from pathlib import Path
import json

from cropharvest.datasets import CropHarvest
from cropharvest.utils import DATAFOLDER_PATH
from cropharvest.engineer import TestInstance

from config import (
    SHUFFLE_SEEDS,
    DATASET_TO_SIZES,
    DL_TIML,
)

from dl import (
    train,
    train_timl_model,
    TrainDataLoader,
    concatenate_task_info,
    load_timl_model,
)

from typing import Optional


def evaluate_model(
    data_folder: Path = DATAFOLDER_PATH, zero_shot: bool = False
) -> None:

    evaluation_datasets = CropHarvest.create_benchmark_datasets(data_folder)
    results_folder = data_folder / DL_TIML
    results_folder.mkdir(exist_ok=True)

    for dataset in evaluation_datasets:

        sample_sizes = DATASET_TO_SIZES[dataset.id]

        if zero_shot:
            shuffle_seeds = [0]
            sample_sizes = [0]
        else:
            shuffle_seeds = SHUFFLE_SEEDS

        for seed in shuffle_seeds:
            dataset.shuffle(seed)
            task_info = TrainDataLoader.task_to_task_info(dataset.task)
            num_timesteps = dataset[0][0].shape[0]
            for sample_size in sample_sizes:
                print(
                    f"Running {DL_TIML} for {dataset}, seed: {seed} with size {sample_size}"
                )

                json_suffix = f"{'zero_shot_' if zero_shot else ''}{dataset.id}_{sample_size}_{seed}.json"
                nc_suffix = f"{'zero_shot_' if zero_shot else ''}{dataset.id}_{sample_size}_{seed}.nc"

                uses_encoder = (
                    data_folder / DL_TIML / "encoder_state_dict.pth"
                ).exists()

                x, y = dataset.as_array(num_samples=10)

                model = load_timl_model(
                    task_info=task_info,
                    train_x=x,
                    train_y=y,
                    input_size=dataset.num_bands
                    if uses_encoder
                    else dataset.num_bands + task_info.shape[0],
                    num_timesteps=num_timesteps,
                    model_state_dict_path=data_folder / DL_TIML / "state_dict.pth",
                    encoder_state_dict_path=data_folder
                    / DL_TIML
                    / "encoder_state_dict.pth",
                )

                # train a model
                task_info_to_concatenate: Optional[torch.tensor] = None
                if not uses_encoder:
                    task_info_to_concatenate = torch.from_numpy(task_info).float()

                if not zero_shot:
                    dataset.reset_sampled_indices()
                    model = train(
                        model,
                        dataset,
                        sample_size,
                        task_info_to_concatenate=task_info_to_concatenate,
                    )
                    model.eval()

                for test_id, test_instance in dataset.test_data(max_size=10000):

                    results_json = results_folder / f"{test_id}_{json_suffix}"
                    results_nc = results_folder / f"{test_id}_{nc_suffix}"

                    if results_json.exists():
                        print(f"Results already saved for {results_json} - skipping")

                    test_x = torch.from_numpy(test_instance.x).float()
                    with torch.no_grad():
                        if task_info_to_concatenate is not None:
                            test_x = concatenate_task_info(
                                test_x, task_info_to_concatenate
                            )
                        preds = model(test_x).squeeze(dim=1).numpy()
                    results = test_instance.evaluate_predictions(preds)

                    with Path(results_json).open("w") as f:
                        json.dump(results, f)

                    ds = test_instance.to_xarray(preds)
                    ds.to_netcdf(results_nc)
                # finally, we want to get results when all the test instances are considered
                # together
                all_nc_files = list(results_folder.glob(f"*_{nc_suffix}"))
                combined_instance, combined_preds = TestInstance.load_from_nc(
                    all_nc_files
                )

                combined_results = combined_instance.evaluate_predictions(
                    combined_preds
                )

                with (results_folder / f"combined_{json_suffix}").open("w") as f:
                    json.dump(combined_results, f)


if __name__ == "__main__":

    data_folder = DATAFOLDER_PATH
    checkpoint = True

    # we start by making the state_dicts necessary for the pretrained models
    if checkpoint and (data_folder / DL_TIML / "state_dict.pth").exists():
        pass
    else:
        train_timl_model(data_folder, model_name=DL_TIML)

    evaluate_model(data_folder, zero_shot=True)
