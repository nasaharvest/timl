from pathlib import Path
import pandas as pd
import json
import dill
import warnings
from random import shuffle

import torch
from torch import nn
from torch import optim

import learn2learn as l2l
from tqdm import tqdm
import numpy as np
import subprocess

from .lstm import Classifier as LSTMClassifier
from .cnn import ConvNet
from .encoder import TaskEncoder

from .data import CropYieldDataset
from .utils import sample_with_memory

from typing import cast, Dict, Tuple, Optional, List, Union


class Learner:
    def __init__(
        self,
        root,
        model_name: str,
        model_kwargs: Dict,
        model_type: str,
        k: int = 10,
        update_val_size: int = 8,
        val_size: float = 0.1,
        encoder_vector_sizes: Union[List[int], int] = 128,
        encoder_dropout: float = 0.2,
        concatenate_task_info: bool = False,
        min_test_year: int = 2011,
        sampling_buffer: Optional[float] = 0.1,
        num_encoder_channels_per_group: Union[int, List[int]] = 16,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        max_val_tasks: int = 50,
        add_awareness: bool = True,
    ) -> None:

        assert model_name.endswith(str(min_test_year))
        assert model_type in ["lstm", "cnn"]
        include_year = False
        if model_type == "lstm":
            include_year = True

        self.min_total_k = k + update_val_size
        self.concatenate_task_info = concatenate_task_info
        self.device = device

        self.model_info: Dict = {
            "k": k,
            "update_val_size": update_val_size,
            "model_kwargs": model_kwargs,
            "model_type": model_type,
            "val_size": val_size,
            "max_val_tasks": max_val_tasks,
            "encoder_dropout": encoder_dropout,
            "encoder_vector_sizes": encoder_vector_sizes,
            "concatenate_task_info": concatenate_task_info,
            "include_year": include_year,
            "num_encoder_channels_per_group": num_encoder_channels_per_group,
            "add_awareness": add_awareness,
            "git-describe": subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8"),
        }

        self.root = Path(root)

        self.train_dl = CropYieldDataset(
            root_dir=str(self.root),
            is_test=False,
            concatenate_task_info=self.concatenate_task_info,
            min_test_year=min_test_year,
            sampling_buffer=sampling_buffer,
            model_type=model_type,
            include_year=include_year,
            device=device,
        )
        self.test_dl = CropYieldDataset(
            root_dir=str(self.root),
            is_test=True,
            concatenate_task_info=self.concatenate_task_info,
            min_test_year=min_test_year,
            sampling_buffer=False,
            model_type=model_type,
            include_year=include_year,
            device=device,
        )
        self.test_dl.update_normalizing_dicts(
            self.train_dl.hist_norm_dict,
            self.train_dl.task_norm_dict,
        )

        self.model_info["input_shape"] = self.train_dl.input_shape
        self.model_info["input_task_info_size"] = self.train_dl.task_info_size

        if concatenate_task_info:
            assert model_type == "lstm"
            input_size = self.train_dl.num_bands + self.train_dl.task_info_size
        else:
            input_size = self.train_dl.num_bands
        self.model_info["input_size"] = input_size

        if model_type == "lstm":
            self.model = LSTMClassifier(input_size=input_size, **model_kwargs).to(device)
        else:
            self.model = ConvNet(in_channels=input_size, **model_kwargs).to(device)

        self.encoder: Optional[nn.Module] = None
        self.concatenate_task_info = concatenate_task_info
        self.add_awareness = add_awareness
        if isinstance(encoder_vector_sizes, int):
            encoder_vector_sizes = [encoder_vector_sizes]
        if self.add_awareness:
            if model_type == "lstm":
                self.encoder = TaskEncoder(
                    encoder_vector_sizes=encoder_vector_sizes,
                    input_size=self.train_dl.task_info_size,
                    raw_input_shape=self.train_dl.input_shape,
                    hidden_vector_shapes=[
                        (x,) for x in model_kwargs["classifier_vector_inout"][:-1]
                    ],
                    encoder_dropout=encoder_dropout,
                    num_channels_per_group=num_encoder_channels_per_group,
                ).to(device)
            elif model_type == "cnn":

                height, width = self.train_dl.input_shape[1], self.train_dl.input_shape[2]
                hidden_vector_shapes = []
                for idx, c in enumerate(model_kwargs["channel_sizes"]):
                    divisor = model_kwargs["stride_list"][idx]
                    if divisor == 2:
                        height //= 2
                        width //= 2
                    hidden_vector_shapes.append((c, int(height), int(width)))
                for dense_size in model_kwargs["dense_features"][:-1]:
                    hidden_vector_shapes.append((dense_size,))

                self.encoder = TaskEncoder(
                    encoder_vector_sizes=encoder_vector_sizes,
                    input_size=self.train_dl.task_info_size,
                    raw_input_shape=self.train_dl.input_shape,
                    hidden_vector_shapes=hidden_vector_shapes,
                    encoder_dropout=encoder_dropout,
                    num_channels_per_group=num_encoder_channels_per_group,
                ).to(device)

        self.loss = nn.MSELoss(reduction="mean")

        self.results_dict: Dict[str, List[float]] = {
            "meta_train": [],
            "meta_val": [],
        }

        self.train_info: Dict = {}
        self.maml: Optional[l2l.algorithms.MAML] = None

        self.model_folder = self.root / model_name
        self.model_folder.mkdir(exist_ok=True)

        model_increment = 0
        version_id = f"version_{model_increment}"

        while (self.model_folder / version_id).exists():
            model_increment += 1
            version_id = f"version_{model_increment}"

        self.version_folder = self.model_folder / version_id
        self.version_folder.mkdir()

        self.model_info["version_number"] = model_increment

        with (self.version_folder / "model_info.json").open("w") as f:  # type: ignore
            json.dump(self.model_info, f)  # type: ignore

    def fast_adapt(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        learner: nn.Module,
        k: int,
        val_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size = k
        data, labels = batch

        max_train_index = int(data.shape[0] - val_size)
        adaptation_data, adaptation_labels = data[:max_train_index], labels[:max_train_index]
        evaluation_data, evaluation_labels = data[max_train_index:], labels[max_train_index:]
        # Adapt the model
        num_adaptation_steps = len(adaptation_data) // batch_size  # should divide cleanly
        for i in range(num_adaptation_steps):
            x = adaptation_data[i * batch_size : (i + 1) * batch_size]
            y = adaptation_labels[i * batch_size : (i + 1) * batch_size]
            train_preds = learner(x).squeeze(dim=1)
            train_error = self.loss(train_preds, y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # currently getting a UserWarning from PyTorch here, but it can be ignored
                learner.adapt(train_error)

        # Evaluate the adapted model
        if len(evaluation_data) > 0:
            preds = learner(evaluation_data).squeeze(dim=1)
            valid_error = self.loss(preds, evaluation_labels)
            return train_error, valid_error
        return train_error, None

    def train(
        self,
        update_lr: float = 0.001,
        meta_lr: float = 0.001,
        min_meta_lr: float = 0.00001,
        max_adaptation_steps: int = 1,
        task_batch_size: int = 32,
        num_iterations: int = 1000,
        encoder_lr: float = 0.001,
        save_best_val: bool = True,
        checkpoint_every: int = 20,
        schedule: bool = False,
        task_removal_threshold: Optional[float] = 4,
        task_removal_lookback: Optional[int] = 20,
        task_noise_scale: float = 0.1,
    ) -> None:

        if task_removal_threshold is not None:
            assert task_removal_lookback is not None

        # for now, we will just overwrite this
        self.train_info = {
            "update_lr": update_lr,
            "meta_lr": meta_lr,
            "max_adaptation_steps": max_adaptation_steps,
            "num_iterations": num_iterations,
            "task_batch_size": task_batch_size,
            "task_removal_threshold": task_removal_threshold,
            "task_removal_lookback": task_removal_lookback,
            "task_noise_scale": task_noise_scale,
        }

        if not hasattr(self, "train_cache"):
            self.train_cache = {"best_val_score": np.inf, "removed_labels": []}

        k = self.model_info["k"]
        val_k = self.model_info["update_val_size"]

        with (self.version_folder / "train_info.json").open("w") as f:
            json.dump(self.train_info, f)

        self.maml = l2l.algorithms.MAML(self.model, lr=update_lr, first_order=False)
        opt = optim.Adam(self.maml.parameters(), meta_lr)
        scheduler = None
        if schedule:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=num_iterations, eta_min=min_meta_lr
            )

        encoder_opt: Optional[torch.optim.Optimizer] = None
        encoder_scheduler = None
        if self.add_awareness:
            assert self.encoder is not None
            encoder_opt = optim.Adam(self.encoder.parameters(), encoder_lr)
            if schedule:
                encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    encoder_opt, T_max=num_iterations, eta_min=min_meta_lr
                )

        labels, val_labels = self.train_dl.train_val_split(
            val_size=self.model_info["val_size"],
            min_k=k + val_k,
            max_val_tasks=self.model_info["max_val_tasks"],
        )
        print(f"Using {len(labels)} training and {len(val_labels)} validation tasks")
        self.val_results: Dict[str, Dict[str, List]] = {}
        self.train_results: Dict[str, Dict[str, List]] = {}

        for iteration_num in tqdm(range(num_iterations)):

            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0

            epoch_labels = [
                label for label in labels if label not in self.train_cache["removed_labels"]
            ]
            shuffle(epoch_labels)

            num_instances_in_batch = 0
            for idx, task_label in enumerate(tqdm(epoch_labels, leave=False)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone()

                # find the maximum number of adaptation steps this task can have,
                # based on the task_k
                task_k = self.train_dl.task_k(*task_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                task_info, _, batch = self.train_dl.sample(
                    *task_label, k=final_k, task_noise_scale=task_noise_scale
                )

                if self.add_awareness:
                    assert self.encoder is not None
                    task_encodings = self.encoder(task_info)
                    learner.module.update_embeddings(task_encodings)

                _, evaluation_error = self.fast_adapt(batch, learner, k=k, val_size=val_k)
                assert evaluation_error is not None

                task_label_str = f"{task_label[0]}_{task_label[1]}"
                if task_label_str not in self.train_results:
                    self.train_results[task_label_str] = {"batch_size": [], "loss": []}

                self.train_results[task_label_str]["batch_size"].append(len(batch[0]))
                self.train_results[task_label_str]["loss"].append(evaluation_error.item())

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()

                num_instances_in_batch += 1

                if ((idx % task_batch_size) == 0) or (idx == len(epoch_labels) - 1):

                    for p in self.maml.parameters():
                        p.grad.data.mul_(1.0 / num_instances_in_batch)
                    opt.step()

                    if encoder_opt is not None:
                        assert self.encoder is not None
                        for n, p in self.encoder.named_parameters():
                            try:
                                p.grad.data.mul_(1.0 / num_instances_in_batch)
                            except AttributeError:
                                print(f"{n} has no grad")
                        encoder_opt.step()

                    opt.zero_grad()
                    if encoder_opt is not None:
                        encoder_opt.zero_grad()
                    num_instances_in_batch = 0
            if schedule:
                assert scheduler is not None
                scheduler.step()
                if encoder_scheduler is not None:
                    encoder_scheduler.step()

            for val_label in val_labels:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone(first_order=True)

                task_k = self.train_dl.task_k(*val_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                val_task_info, _, val_batch = self.train_dl.sample(*val_label, k=final_k)

                with torch.no_grad():
                    if self.encoder is not None:
                        task_encodings = self.encoder(val_task_info)
                        learner.module.update_embeddings(task_encodings)

                _, val_error = self.fast_adapt(val_batch, learner, k=k, val_size=val_k)
                assert val_error is not None

                val_label_str = f"{val_label[0]}_{val_label[1]}"
                if val_label_str not in self.val_results:
                    self.val_results[val_label_str] = {"batch_size": [], "loss": []}

                self.val_results[val_label_str]["batch_size"].append(len(val_batch[0]))
                self.val_results[val_label_str]["loss"].append(val_error.item())

                # note that backwards is not called
                meta_valid_error += val_error.item()

            if task_removal_threshold is not None and (iteration_num >= task_removal_lookback):
                for label in epoch_labels:
                    mean_mse = np.mean(
                        self.train_results[f"{label[0]}_{label[1]}"]["loss"][
                            -cast(int, task_removal_lookback) :
                        ]
                    )
                    if mean_mse <= task_removal_threshold:
                        print(
                            f"Removing {label} - average MSE over the last {task_removal_lookback}: {mean_mse}"
                        )
                        self.train_cache["removed_labels"].append(label)

            # Print some metrics
            meta_epoch_size = len(epoch_labels)
            meta_val_size = len(val_labels)
            self.results_dict["meta_train"].append((meta_train_error / meta_epoch_size))
            self.results_dict["meta_val"].append((meta_valid_error / meta_val_size))

            if iteration_num > 0:
                mean_mt = np.mean(self.results_dict["meta_train"][-checkpoint_every:])
                mean_mv = np.mean(self.results_dict["meta_val"][-checkpoint_every:])

                if mean_mv <= self.train_cache["best_val_score"]:
                    self.train_cache["best_val_score"] = mean_mv
                    if save_best_val:
                        self.checkpoint(iteration=iteration_num)

                if iteration_num % checkpoint_every == 0:
                    print(f"Meta_train: {round(mean_mt, 3)}, meta_val: {round(mean_mv, 3)}, ")

        with (self.version_folder / "results.json").open("w") as rf:
            json.dump(self.results_dict, rf)

        with (self.version_folder / "val_results.json").open("w") as valf:
            json.dump(self.val_results, valf)

        with (self.version_folder / "train_results.json").open("w") as trainf:
            json.dump(self.train_results, trainf)

        with (self.version_folder / "final_model.pkl").open("wb") as mf:
            dill.dump(self, mf)

    def checkpoint(self, iteration: int) -> None:

        checkpoint_files = list(self.version_folder.glob("checkpoint*"))
        if len(checkpoint_files) > 0:
            for filepath in checkpoint_files:
                filepath.unlink()
        with (self.version_folder / f"checkpoint_iteration_{iteration}.pkl").open("wb") as f:
            dill.dump(self, f)

        classifier_files = list(self.version_folder.glob("classifier_state_dict_*"))
        if len(classifier_files) > 0:
            for filepath in classifier_files:
                filepath.unlink()
        torch.save(
            self.model.state_dict(),
            self.version_folder / f"classifier_state_dict_iteration_{iteration}.pth",
        )

        # save the encoder seperately, to make it easier to investigate
        if self.add_awareness:
            assert self.encoder is not None
            encoder_files = list(self.version_folder.glob("encoder_state_dict*"))
            if len(encoder_files) > 0:
                for filepath in encoder_files:
                    filepath.unlink()
            torch.save(
                self.encoder.state_dict(),
                self.version_folder / f"encoder_state_dict_iteration_{iteration}.pth",
            )

        # in addition, we will save a state_dict to be retrieved by the
        # benchmarking model
        torch.save(self.model.state_dict(), self.model_folder / "state_dict.pth")
        if self.encoder is not None:
            torch.save(self.encoder.state_dict(), self.model_folder / "encoder_state_dict.pth")

    def _finetune_and_run_for_test_task(
        self,
        state_fip: int,
        county_fip: int,
        finetuning_steps: int,
        learning_rate: float,
        zero_shot: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Finetune a trained (TI/MA)ML model and return a tuple of [True values, predictions]
        """
        # first, we check if there is training data to use
        train_k = self.model_info["k"]
        enough_samples_to_train = (
            len(self.train_dl.region_indices(state_fip, county_fip)) >= train_k
        )

        if zero_shot & enough_samples_to_train:
            return None

        if self.model_info["model_type"] == "lstm":
            classifier = LSTMClassifier(
                input_size=self.model_info["input_size"], **self.model_info["model_kwargs"]
            ).to(self.device)
        else:
            classifier = ConvNet(
                in_channels=self.model_info["input_size"], **self.model_info["model_kwargs"]
            ).to(self.device)

        classifier.load_state_dict(self.model.state_dict())

        task_info, test_years, (test_x, test_y) = self.test_dl.sample(state_fip, county_fip)
        latlons = self.train_dl.to_latlon(state_fip, county_fip)
        latlons_test = np.stack([latlons] * test_y.shape[0])
        assert len(np.unique(test_years)) == len(test_years)
        if self.encoder is not None:
            with torch.no_grad():
                self.encoder.eval()
                task_encodings = self.encoder(task_info)
                classifier.update_embeddings(task_encodings)

        if enough_samples_to_train:

            opt = optim.SGD(classifier.parameters(), lr=learning_rate)
            _, _, (all_train_x, all_train_y) = self.train_dl.sample(state_fip, county_fip, k=None)
            state: List[int] = []

            for _ in range(finetuning_steps):
                classifier.train()
                opt.zero_grad()

                indices, state = sample_with_memory(
                    list(range(all_train_x.shape[0])), train_k, state
                )

                train_x, train_y = all_train_x[indices], all_train_y[indices]

                preds = classifier(train_x).squeeze(dim=1)
                loss = self.loss(preds, train_y)
                loss.backward()
                opt.step()

        else:
            print(
                f"Skipping for {state_fip}, {county_fip}; k: {train_k} "
                f"for {len(self.train_dl.region_indices(state_fip, county_fip))} training samples"
            )

        # then, we evaluate it on the test data
        classifier.eval()
        with torch.no_grad():
            test_preds, test_h = classifier.forward(test_x, return_last_dense=True)
            test_preds = test_preds.squeeze(dim=1).cpu().numpy()
            test_h = test_h.cpu().numpy()

        return test_preds, test_y.cpu().numpy(), np.array(test_years), latlons_test, test_h

    def _test(
        self,
        finetuning_steps: int,
        learning_rate: float,
        max_regions: int,
        zero_shot: bool,
    ) -> Tuple[List[List[float]], np.ndarray, np.ndarray, np.ndarray, Dict]:
        region_tuples = self.test_dl.region_tuples

        test_loc, test_preds, test_true, test_years, test_latlon, test_h = [], [], [], [], [], []

        num_regions = 0
        for _, row in tqdm(region_tuples.iterrows(), total=len(region_tuples)):
            output = self._finetune_and_run_for_test_task(
                row.state_fip, row.county_fip, finetuning_steps, learning_rate, zero_shot
            )
            if output is None:
                continue
            test_output = output[0]
            test_preds.append(test_output[0])
            test_true.append(test_output[1])
            test_years.append(test_output[2])
            test_latlon.append(test_output[3])
            test_h.append(test_output[4])
            test_loc.extend([[row.state_fip, row.county_fip]] * len(test_output[0]))
            num_regions += 1
            if max_regions is not None:
                if num_regions >= max_regions:
                    break

        test_preds_np = np.concatenate(test_preds)
        test_true_np = np.concatenate(test_true)
        test_years_np = np.concatenate(test_years)

        print(f"Overall results: {self.test_dl.eval(test_preds_np, test_true_np)}\n")

        # Results broken down by year
        results_dict = {}
        for year in np.unique(test_years_np):
            [rmse, _], result_str = self.test_dl.eval(
                test_preds_np[test_years_np == year], test_true_np[test_years_np == year]
            )
            print(f"Results for {int(year)}: {result_str}\n")

            results_dict[year] = rmse

        return test_loc, test_preds_np, test_true_np, test_years_np, results_dict

    def test(
        self,
        finetuning_steps: int = 15,
        learning_rate: float = 0.001,
        num_runs: int = 10,
        max_regions: Optional[int] = None,
        save_vals: bool = True,
        zero_shot: bool = False,
    ):
        test_loc, test_preds, test_true, test_years = [], [], [], []
        overall_results_dict = {"run_number": [], "year": [], "RMSE": []}
        for i in range(num_runs):
            print(f"Running iteration {i}")
            locs, preds, true, years, results = self._test(
                finetuning_steps, learning_rate, max_regions, zero_shot
            )
            test_preds.append(preds)
            test_true.append(true)
            test_years.append(years)
            test_loc.extend(locs)

            for year, rmse in results.items():
                overall_results_dict["run_number"].append(i)
                overall_results_dict["year"].append(year)
                overall_results_dict["RMSE"].append(rmse)

        test_preds_np = np.concatenate(test_preds)
        test_true_np = np.concatenate(test_true)
        test_years_np = np.concatenate(test_years)
        test_loc_np = np.array(test_loc)
        overall_results, overall_results_str = self.test_dl.eval(test_preds_np, test_true_np)

        print(f"Overall overall results: {overall_results_str}\n")

        results = {"overall": overall_results}

        suffix = f"_finetuning_steps_{finetuning_steps}{'zero_shot' if zero_shot else ''}"

        # Results broken down by year
        for year in np.unique(test_years_np):
            results_tuple, result_str = self.test_dl.eval(
                test_preds_np[test_years_np == year], test_true_np[test_years_np == year]
            )
            print(f"Overall results for {int(year)}: {result_str}\n")
            results[int(year)] = results_tuple
        with (self.version_folder / f"test_results{suffix}.json").open("w") as rf:
            json.dump(results, rf)
        pd.DataFrame(data=overall_results_dict).to_csv(
            self.version_folder / f"test_results{suffix}.csv"
        )

        if save_vals:
            np.save(self.version_folder / f"test_preds{suffix}.npy", test_preds_np)
            np.save(self.version_folder / f"test_true{suffix}.npy", test_true_np)
            np.save(self.version_folder / f"test_years{suffix}.npy", test_years_np)
            np.save(self.version_folder / f"test_loc{suffix}.npy", test_loc_np)

    def to(self, device: torch.device) -> None:
        self.model.to(device)
        if self.encoder is not None:
            self.encoder.to(device)
        self.train_dl.to(device)
        self.test_dl.to(device)


def train_timl_model(
    root,
    model_name: str,
    model_kwargs: Dict,
    model_type: str,
    encoder_vector_sizes: Union[List[int], int],
    encoder_dropout: float,
    k: int = 10,
    update_val_size: int = 8,
    val_size: float = 0.1,
    update_lr: float = 0.001,
    meta_lr: float = 0.001,
    min_meta_lr: float = 0.00001,
    max_adaptation_steps: int = 1,
    task_batch_size: int = 32,
    num_iterations: int = 1000,
    concatenate_task_info: bool = False,
    save_best_val: bool = True,
    checkpoint_every: int = 20,
    schedule: bool = True,
    task_noise_scale: float = 0.1,
    min_test_year: int = 2011,
) -> Learner:
    r"""
    Initialize a classifier and pretrain it using model-agnostic meta-learning (MAML)

    :root: The path to the data
    :param classifier_vector_size: The LSTM hidden vector size to use
    :param classifier_dropout: The value for variational dropout between LSTM timesteps to us
    :param classifier_base_layers: The number of LSTM layers to use
    :param num_classification_layers: The number of linear classification layers to use on top
        of the LSTM base
    :param model_name: The model name. The model's weights will be saved at root / model_name.
    :param k: The number of positive and negative examples to use during inner loop training
    :param val_size: The number of positive and negative examples to use during outer loop training
    :param update_lr: The learning rate to use when learning a specific task
        (inner loop learning rate)
    :param meta_lr: The learning rate to use when updating the MAML model
        (outer loop learning rate)
    :param min_meta_lr: The minimum meta learning rate to use for the cosine
        annealing scheduler. Only used if `schedule == True`
    :param max_adaptation_steps: The maximum number of adaptation steps to be used
        per task. Each task will do as many adaptation steps as possible (up to this
        upper bound) given the number of unique positive and negative data instances
        it has
    :param task_batch_size: The number of tasks to batch before each outer loop update
    :param num_iterations: The number of iterations to train the meta-model for. One
        iteration is a complete pass over all the tasks
    :param encoder_vector_sizes: The size of the encoder's hidden layers (the default
        argument is a list, but this function should only get called once)
    :param encoder_dropout: The dropout to use between encoder layers
    :param concatenate_task_info: Whether to concatenate task info to the raw input. If
        True, no encoder is used
    :param save_best_val: Whether to save the model with the best validation score,
        as well as the final model
    :param checkpoint_every: The model prints out training statistics every
        `checkpoint_every` iteration
    :param save_train_tasks_results: Whether to save the results for the training
        tasks to a json object
    :param schedule: Whether to use cosine annealing on the meta learning rate during
        training
    :param task_noise_scale: The scale of the gaussian noise to add to the task
        information during training
    """
    model = Learner(
        root,
        model_name,
        model_kwargs,
        model_type,
        k,
        update_val_size,
        val_size,
        encoder_vector_sizes,
        encoder_dropout,
        concatenate_task_info,
        min_test_year,
    )

    model.train(
        update_lr=update_lr,
        meta_lr=meta_lr,
        min_meta_lr=min_meta_lr,
        max_adaptation_steps=max_adaptation_steps,
        task_batch_size=task_batch_size,
        num_iterations=num_iterations,
        save_best_val=save_best_val,
        checkpoint_every=checkpoint_every,
        schedule=schedule,
        task_noise_scale=task_noise_scale,
    )

    # load the best model, and return it
    checkpoints = list(model.version_folder.glob("checkpoint*"))
    assert len(checkpoints) == 1
    with checkpoints[0].open("rb") as f:
        best_model = dill.load(f)

    return best_model
