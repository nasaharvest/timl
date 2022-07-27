from encodings import normalize_encoding
from pathlib import Path
import json
import dill
import warnings
from random import shuffle, random
from collections import defaultdict

import torch
from torch import nn
from torch import optim

import learn2learn as l2l
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

from .config import (
    CLASSIFIER_DROPOUT,
    NUM_CLASSIFICATION_LAYERS,
    HIDDEN_VECTOR_SIZE,
    CLASSIFIER_BASE_LAYERS,
    ENCODER_VECTOR_SIZES,
    ENCODER_DROPOUT,
)
from .lstm import Classifier
from .encoder import TaskEncoder, MMAMLEncoder
from .datasets import TIMLCropHarvestLabels, TIMLCropHarvest, TIMLTask

from cropharvest import countries
from cropharvest.crops import to_one_hot
from cropharvest.config import TEST_DATASETS, TEST_REGIONS
from cropharvest.utils import NoDataForBoundingBoxError

from typing import cast, Dict, Tuple, Optional, List, DefaultDict, Union


class TrainDataLoader:
    def __init__(
        self,
        label_to_tasks: Dict[str, TIMLCropHarvest],
        concatenate_task_info: bool = False,
    ):
        self.label_to_tasks = label_to_tasks
        self.concatenate_task_info = concatenate_task_info

    @property
    def task_labels(self) -> List[str]:
        return list(self.label_to_tasks.keys())

    def task_k(self, label: str) -> int:
        return self.label_to_tasks[label].k

    @property
    def num_timesteps(self) -> int:
        # array has shape [timesteps, bands]
        return self.label_to_tasks[self.task_labels[0]][0][0].shape[0]

    @property
    def num_bands(self) -> int:
        return self.label_to_tasks[self.task_labels[0]].num_bands

    @property
    def task_info_size(self) -> int:
        return self.task_to_task_info(
            self.label_to_tasks[self.task_labels[0]].task
        ).shape[0]

    @staticmethod
    def task_to_task_info(task: TIMLTask, noise_scale: float = 0) -> np.ndarray:
        task_info = np.array(
            task.bounding_box.three_dimensional_points
            + to_one_hot(task.classification_label)
        )
        if noise_scale > 0:
            noise = np.random.normal(loc=0.0, scale=noise_scale, size=len(task_info))
            task_info += noise
        return task_info

    def sample_task(
        self, label: str, k: int, task_noise_scale: float = 0
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, y = self.label_to_tasks[label].sample(k)
        task_info = self.task_to_task_info(
            self.label_to_tasks[label].task, task_noise_scale
        )
        if self.concatenate_task_info:
            x = self._concatenate_task_info(x, task_info)
        return (
            torch.from_numpy(task_info).float(),
            (torch.from_numpy(x).float(), torch.from_numpy(y).float()),
        )

    @staticmethod
    def _concatenate_task_info(x: np.ndarray, task_info: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 3
        # x has shape [batches, timesteps, bands]. We stack the task info so that it
        # has this same shape, repeated for all batches and timesteps
        task_info_expanded = np.stack([np.stack([task_info] * x.shape[1])] * x.shape[0])
        # we then concatenate it to the bands dimension of x before returning it
        return np.concatenate([x, task_info_expanded], axis=-1)


class Learner:
    def __init__(
        self,
        root,
        model_name: str,
        classifier_vector_size: int,
        classifier_dropout: float,
        classifier_base_layers: int,
        num_classification_layers: int,
        k: int = 10,
        update_val_size: int = 8,
        val_size: float = 0.1,
        encoder_vector_sizes: Union[List[int], int] = 128,
        encoder_dropout: float = 0.2,
        concatenate_task_info: bool = False,
        num_encoder_channels_per_group: Union[int, List[int]] = 16,
        mmaml: bool = False,
        task_awareness: bool = True,
    ) -> None:

        assert not (mmaml and task_awareness), "Can't have both MMAML and TIML"
        # update val size needs to be divided by 2 since
        # k is multiplied by 2 (k = number of positive / negative vals)
        min_total_k = k + (update_val_size // 2)

        self.model_info: Dict = {
            "k": k,
            "update_val_size": update_val_size,
            "classifier_vector_size": classifier_vector_size,
            "classifier_dropout": classifier_dropout,
            "classifier_base_layers": classifier_base_layers,
            "num_classification_layers": num_classification_layers,
            "val_size": val_size,
            "encoder_dropout": encoder_dropout,
            "encoder_vector_sizes": encoder_vector_sizes,
            "concatenate_task_info": concatenate_task_info,
            "num_encoder_channels_per_group": num_encoder_channels_per_group,
            "mmaml": mmaml,
            "task_awareness": task_awareness
            # "git-describe": subprocess.check_output(["git", "describe", "--always"])
            # .strip()
            # .decode("utf-8"),
        }

        self.root = Path(root)
        self.train_tasks, self.val_tasks = {}, {}

        train_tasks, val_tasks = self._make_tasks(
            min_task_k=min_total_k, val_size=val_size
        )
        print(f"Using {len(train_tasks)} train tasks and {len(val_tasks)} val tasks")
        self.train_tasks.update(train_tasks)
        self.val_tasks.update(val_tasks)

        self.train_dl = TrainDataLoader(
            label_to_tasks=self.train_tasks, concatenate_task_info=concatenate_task_info
        )
        self.val_dl = TrainDataLoader(
            label_to_tasks=self.val_tasks, concatenate_task_info=concatenate_task_info
        )

        self.model_info["input_size"] = self.train_dl.num_bands
        self.model_info["num_timesteps"] = self.train_dl.num_timesteps
        self.model_info["input_task_info_size"] = self.train_dl.task_info_size

        if concatenate_task_info:
            input_size = self.train_dl.num_bands + self.train_dl.task_info_size
        else:
            input_size = self.train_dl.num_bands
        self.model = Classifier(
            input_size=input_size,
            classifier_base_layers=classifier_base_layers,
            classifier_dropout=classifier_dropout,
            classifier_vector_size=classifier_vector_size,
            num_classification_layers=num_classification_layers,
        )

        self.encoder: Optional[nn.Module] = None
        self.concatenate_task_info = concatenate_task_info
        self.task_awareness = task_awareness
        self.mmaml = mmaml
        if self.task_awareness:
            if isinstance(encoder_vector_sizes, int):
                encoder_vector_sizes = [encoder_vector_sizes]
            self.encoder = TaskEncoder(
                encoder_vector_sizes=encoder_vector_sizes,
                input_size=self.train_dl.task_info_size,
                num_bands=self.train_dl.num_bands,
                num_hidden_layers=num_classification_layers,
                hidden_vector_size=classifier_vector_size,
                encoder_dropout=encoder_dropout,
                num_timesteps=self.train_dl.num_timesteps,
                num_channels_per_group=num_encoder_channels_per_group,
            )
        elif self.mmaml:
            self.encoder = MMAMLEncoder(
                num_bands=input_size,
                num_hidden_layers=num_classification_layers,
                hidden_vector_size=classifier_vector_size,
                encoder_dropout=encoder_dropout,
                num_timesteps=self.train_dl.num_timesteps,
            )

        self.loss = nn.BCELoss(reduction="mean")

        self.results_dict: Dict[str, List[float]] = {
            "meta_train": [],
            "meta_val": [],
            "meta_val_auc": [],
            "meta_train_auc": [],
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
        calc_auc_roc: bool,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:

        batch_size = k * 2  # k is the number of positive, negative examples
        data, labels = batch

        max_train_index = int(data.shape[0] - val_size)
        adaptation_data, adaptation_labels = (
            data[:max_train_index],
            labels[:max_train_index],
        )
        evaluation_data, evaluation_labels = (
            data[max_train_index:],
            labels[max_train_index:],
        )

        train_auc_roc: Optional[float] = None
        valid_auc_roc: Optional[float] = None

        # Adapt the model
        num_adaptation_steps = (
            len(adaptation_data) // batch_size
        )  # should divide cleanly
        for i in range(num_adaptation_steps):
            x = adaptation_data[i * batch_size : (i + 1) * batch_size]
            y = adaptation_labels[i * batch_size : (i + 1) * batch_size]
            train_preds = learner(x).squeeze(dim=1)
            train_error = self.loss(train_preds, y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # currently getting a UserWarning from PyTorch here, but it can be ignored
                learner.adapt(train_error)
            if calc_auc_roc:
                train_auc_roc = roc_auc_score(
                    y.cpu().numpy(), train_preds.detach().cpu().numpy()
                )

        # Evaluate the adapted model
        if len(evaluation_data) > 0:
            preds = learner(evaluation_data).squeeze(dim=1)
            valid_error = self.loss(preds, evaluation_labels)
            if calc_auc_roc:
                valid_auc_roc = roc_auc_score(
                    evaluation_labels.cpu().numpy(), preds.detach().cpu().numpy()
                )
            else:
                valid_auc_roc = None
            return train_error, train_auc_roc, valid_error, valid_auc_roc
        return train_error, train_auc_roc, None, None

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
        task_noise_scale: float = 0.1,
        task_removal_threshold: Optional[float] = 0.95,
        task_removal_lookback: Optional[int] = 20,
    ) -> None:

        if task_removal_threshold is not None:
            assert task_removal_lookback is not None

        self.train_info = {
            "update_lr": update_lr,
            "meta_lr": meta_lr,
            "max_adaptation_steps": max_adaptation_steps,
            "num_iterations": num_iterations,
            "task_batch_size": task_batch_size,
            "task_noise_scale": task_noise_scale,
            "task_removal_threshold": task_removal_threshold,
            "task_removal_lookback": task_removal_lookback,
        }

        k = self.model_info["k"]
        update_val_size = self.model_info["update_val_size"]
        val_k = update_val_size // 2

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
        if (self.task_awareness) or (self.mmaml):
            assert self.encoder is not None
            encoder_opt = optim.Adam(self.encoder.parameters(), encoder_lr)
            if schedule:
                encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    encoder_opt, T_max=num_iterations, eta_min=min_meta_lr
                )

        best_val_score = np.inf

        val_labels = self.val_dl.task_labels
        self.val_results: Dict[str, Dict[str, List]] = {}
        self.train_results: Dict[str, Dict[str, List]] = {}

        labels = self.train_dl.task_labels
        removed_labels: List[str] = []

        for iteration_num in tqdm(range(num_iterations)):

            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0
            meta_train_auc = 0.0
            meta_valid_auc_roc = 0.0

            epoch_labels = [label for label in labels if label not in removed_labels]
            shuffle(epoch_labels)

            num_instances_in_batch = 0
            for idx, task_label in enumerate(epoch_labels):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone()

                # find the maximum number of adaptation steps this task can have,
                # based on the task_k
                task_k = self.train_dl.task_k(task_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                task_info, batch = self.train_dl.sample_task(
                    task_label, k=final_k, task_noise_scale=task_noise_scale
                )

                if self.task_awareness:
                    assert self.encoder is not None
                    task_encodings = self.encoder(task_info)
                    learner.module.update_embeddings(task_encodings)
                elif self.mmaml:
                    assert self.encoder is not None
                    task_encodings = self.encoder(*batch)
                    learner.module.update_embeddings(task_encodings)

                _, _, evaluation_error, train_auc = self.fast_adapt(
                    batch, learner, k=k, val_size=update_val_size, calc_auc_roc=True
                )
                assert evaluation_error is not None
                assert train_auc is not None

                if task_label not in self.train_results:
                    self.train_results[task_label] = {
                        "batch_size": [],
                        "AUC": [],
                        "loss": [],
                    }

                self.train_results[task_label]["batch_size"].append(len(batch[0]))
                self.train_results[task_label]["AUC"].append(train_auc)
                self.train_results[task_label]["loss"].append(evaluation_error.item())

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_auc += train_auc

                num_instances_in_batch += 1

                if ((idx % task_batch_size) == 0) or (idx == len(epoch_labels) - 1):

                    for p in self.maml.parameters():
                        p.grad.data.mul_(1.0 / num_instances_in_batch)
                    opt.step()

                    if encoder_opt is not None:
                        assert self.encoder is not None
                        for p in self.encoder.parameters():
                            p.grad.data.mul_(1.0 / num_instances_in_batch)
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
                    learner = self.maml.clone()

                task_k = self.val_dl.task_k(val_label)

                task_train_k = task_k - val_k
                num_steps = min(max_adaptation_steps, task_train_k // k)

                final_k = (k * num_steps) + val_k

                val_task_info, val_batch = self.val_dl.sample_task(val_label, k=final_k)

                with torch.no_grad():
                    if self.task_awareness:
                        task_encodings = self.encoder(val_task_info)
                        learner.module.update_embeddings(task_encodings)
                    elif self.mmaml:
                        task_encodings = self.encoder(*val_batch)
                        learner.module.update_embeddings(task_encodings)

                _, _, val_error, val_auc = self.fast_adapt(
                    val_batch, learner, k=k, val_size=update_val_size, calc_auc_roc=True
                )
                assert val_error is not None
                assert val_auc is not None

                if val_label not in self.val_results:
                    self.val_results[val_label] = {
                        "batch_size": [],
                        "AUC": [],
                        "loss": [],
                    }

                self.val_results[val_label]["batch_size"].append(len(val_batch[0]))
                self.val_results[val_label]["AUC"].append(val_auc)
                self.val_results[val_label]["loss"].append(val_error.item())

                # note that backwards is not called
                meta_valid_error += val_error.item()
                meta_valid_auc_roc += val_auc

            if task_removal_threshold is not None and (
                iteration_num >= task_removal_lookback
            ):
                for label in epoch_labels:
                    mean_auc = np.mean(
                        self.train_results[label]["AUC"][
                            -cast(int, task_removal_lookback) :
                        ]
                    )
                    if mean_auc >= task_removal_threshold:
                        print(
                            f"Removing {label} - average AUC over the last {task_removal_lookback}: {mean_auc}"
                        )
                        removed_labels.append(label)

            # Print some metrics
            meta_epoch_size = len(epoch_labels)
            meta_val_size = len(val_labels)
            self.results_dict["meta_train"].append((meta_train_error / meta_epoch_size))
            self.results_dict["meta_val"].append((meta_valid_error / meta_val_size))
            self.results_dict["meta_val_auc"].append(meta_valid_auc_roc / meta_val_size)
            self.results_dict["meta_train_auc"].append(meta_train_auc / meta_epoch_size)

            if iteration_num > 0:
                mean_mt = np.mean(self.results_dict["meta_train"][-checkpoint_every:])
                mean_mv = np.mean(self.results_dict["meta_val"][-checkpoint_every:])
                mean_mvauc = np.mean(
                    self.results_dict["meta_val_auc"][-checkpoint_every:]
                )
                mean_mtauc = np.mean(
                    self.results_dict["meta_train_auc"][-checkpoint_every:]
                )

                if mean_mv <= best_val_score:
                    best_val_score = mean_mv
                    if save_best_val:
                        self.checkpoint(iteration=iteration_num)

                if iteration_num % checkpoint_every == 0:
                    print(
                        f"Meta_train: {round(mean_mt, 3)}, meta_val: {round(mean_mv, 3)}, "
                        f"meta_train_auc: {round(mean_mtauc, 3)}, "
                        f"meta_val_auc: {round(mean_mvauc, 3)}"
                    )

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
        with (self.version_folder / f"checkpoint_iteration_{iteration}.pkl").open(
            "wb"
        ) as f:
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
        if (self.task_awareness) or (self.mmaml):
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
            torch.save(
                self.encoder.state_dict(), self.model_folder / "encoder_state_dict.pth"
            )

    def _make_tasks(
        self, min_task_k: int, val_size: float
    ) -> Tuple[Dict[str, TIMLCropHarvest], Dict[str, TIMLCropHarvest]]:
        labels = TIMLCropHarvestLabels(self.root)

        # remove any test regions, and collect the countries / crops
        test_countries_to_crops: DefaultDict[str, List[str]] = defaultdict(list)

        # reshuffle the test_regions dict so its a little easier to
        # manipulate in this function
        for identifier, _ in TEST_REGIONS.items():
            country, crop, _, _ = identifier.split("_")
            test_countries_to_crops[country].append(crop)

        label_to_task: Dict[str, TIMLCropHarvest] = {}

        countries_to_ignore = [
            country for country, _ in TEST_DATASETS.items() if crop is None
        ]

        for country in tqdm(countries.get_countries()):
            if country in countries_to_ignore:
                continue
            country_bboxes = countries.get_country_bbox(country)
            for _, country_bbox in enumerate(country_bboxes):

                try:
                    task = TIMLCropHarvest(
                        self.root,
                        TIMLTask(
                            bounding_box=country_bbox, target_label=None, normalize=True
                        ),
                    )
                except NoDataForBoundingBoxError:
                    continue

                if task.k >= min_task_k:
                    label_to_task[task.id] = task

                for label, classification_label in labels.classes_in_bbox(
                    country_bbox, return_classifications=True
                ):
                    if country in test_countries_to_crops:
                        if label in test_countries_to_crops[country]:
                            continue
                    try:
                        task = TIMLCropHarvest(
                            self.root,
                            TIMLTask(
                                bounding_box=country_bbox,
                                target_label=label,
                                balance_negative_crops=True,
                                normalize=True,
                                classification_label=classification_label,
                            ),
                        )
                    except NoDataForBoundingBoxError:
                        continue
                    if task.k >= min_task_k:
                        label_to_task[task.id] = task

        train_tasks, val_tasks = {}, {}
        for country, task in label_to_task.items():
            if random() < val_size:
                val_tasks[country] = task
            else:
                train_tasks[country] = task

        return train_tasks, val_tasks


def train_timl_model(
    root,
    model_name: str,
    encoder_vector_sizes: Union[List[int], int],
    classifier_vector_size: int = HIDDEN_VECTOR_SIZE,
    classifier_dropout: float = CLASSIFIER_DROPOUT,
    classifier_base_layers: int = CLASSIFIER_BASE_LAYERS,
    num_classification_layers: int = NUM_CLASSIFICATION_LAYERS,
    encoder_dropout: float = ENCODER_DROPOUT,
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
    task_awareness: bool = True,
    mmaml: bool = False,
) -> Classifier:
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
        classifier_vector_size,
        classifier_dropout,
        classifier_base_layers,
        num_classification_layers,
        k,
        update_val_size,
        val_size,
        encoder_vector_sizes,
        encoder_dropout,
        concatenate_task_info,
        task_awareness=task_awareness,
        mmaml=mmaml,
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

    return model.model


def _check_normalizing_dict(normalizing_dict: Optional[Dict]) -> Dict:
    if normalizing_dict is None:
        return None

    for expected_key in ["mean", "std"]:
        assert expected_key in normalizing_dict.keys()

    if isinstance(normalizing_dict["mean"], list):
        return normalizing_dict
    else:
        return {key: val.tolist() for key, val in normalizing_dict.items()}


def load_timl_model(
    task_info: np.ndarray,
    input_size: int,
    num_timesteps: int,
    model_state_dict_path: Path,
    encoder_state_dict_path: Optional[Path],
    normalizing_dict: Optional[Dict],
):
    """
    Load a trained TIML model
    """

    model = Classifier(
        input_size=input_size,
        classifier_base_layers=CLASSIFIER_BASE_LAYERS,
        num_classification_layers=NUM_CLASSIFICATION_LAYERS,
        classifier_dropout=CLASSIFIER_DROPOUT,
        classifier_vector_size=HIDDEN_VECTOR_SIZE,
    )
    model.load_state_dict(torch.load(model_state_dict_path))
    model.normalizing_dict = _check_normalizing_dict(normalizing_dict)

    if encoder_state_dict_path.exists():
        encoder = TaskEncoder(
            input_size=task_info.shape[0],
            encoder_vector_sizes=ENCODER_VECTOR_SIZES,
            encoder_dropout=ENCODER_DROPOUT,
            num_bands=input_size,
            num_hidden_layers=NUM_CLASSIFICATION_LAYERS,
            hidden_vector_size=HIDDEN_VECTOR_SIZE,
            num_timesteps=num_timesteps,
        )
        encoder.load_state_dict(torch.load(encoder_state_dict_path))
        encoder.eval()
        with torch.no_grad():
            task_embeddings = encoder(torch.from_numpy(task_info).float())
            model.update_embeddings(task_embeddings)

    return model
