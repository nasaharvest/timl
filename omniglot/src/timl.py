import json
import dill
import warnings
from random import shuffle
from pathlib import Path

import torch
from torch import nn
from torch import optim

import learn2learn as l2l
from tqdm import tqdm
import numpy as np

from .data import OmniglotAlphabet, ALPHABETS, VAL_TEST_SPLIT
from .model import OmniglotCNN, accuracy, Encoder

from typing import cast, Dict, Tuple, Optional, List


class Learner:
    def __init__(
        self,
        awareness: Optional[str],
        seed: int = 42,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:

        self.tasks = {
            alphabet: OmniglotAlphabet(alphabet, device=device)
            for alphabet in ALPHABETS
        }
        self.device = device
        self.model = OmniglotCNN().to(device)

        self.encoder: Optional[nn.Module] = None
        self.awareness = awareness
        if awareness:
            self.encoder = Encoder(method=awareness).to(device)

        self.loss = nn.CrossEntropyLoss(reduction="mean")

        self.results_dict: Dict[str, List[float]] = {
            "meta_train": [],
            "meta_val": [],
            "meta_val_acc": [],
            "meta_train_acc": [],
        }

        self.train_info: Dict = {}
        self.maml: Optional[l2l.algorithms.MAML] = None

        self.model_folder = (
            Path(__file__).parents[1]
            / "models"
            / (f"{awareness}_{seed}" if awareness else f"maml_{seed}")
        )
        self.model_folder.mkdir(exist_ok=True)

        model_increment = 0
        version_id = f"version_{model_increment}"

        while (self.model_folder / version_id).exists():
            model_increment += 1
            version_id = f"version_{model_increment}"

        self.version_folder = self.model_folder / version_id
        self.version_folder.mkdir()

    def fast_adapt(
        self,
        num_adaptation_steps: int,
        train_batch: Tuple[torch.Tensor, torch.Tensor],
        eval_batch: Tuple[torch.Tensor, torch.Tensor],
        learner: nn.Module,
    ) -> Tuple[torch.Tensor, Optional[float], Optional[torch.Tensor], Optional[float]]:

        for _ in range(num_adaptation_steps):
            train_preds = learner(train_batch[0])
            train_error = self.loss(train_preds, train_batch[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # currently getting a UserWarning from PyTorch here, but it can be ignored
                learner.adapt(train_error)

        learner.eval()
        preds = learner(eval_batch[0])
        valid_error = self.loss(preds, eval_batch[1])
        val_accuracy = accuracy(preds, eval_batch[1])
        return valid_error, val_accuracy

    def train(
        self,
        update_lr: float = 0.1,
        meta_lr: float = 0.001,
        min_meta_lr: float = 0.00001,
        num_adaptation_steps: int = 1,
        samples_per_way: int = 1,
        task_batch_size: int = 4,
        num_iterations: int = 60000,
        encoder_lr: float = 0.001,
        save_best_val: bool = True,
        checkpoint_every: int = 20,
        schedule: bool = False,
        task_removal_threshold: Optional[float] = 0.99,
        task_removal_lookback: Optional[int] = 100,
        task_noise: float = 0.1,
    ) -> None:

        if self.awareness != "timl":
            task_removal_threshold = task_removal_lookback = None

        if task_removal_threshold is not None:
            assert task_removal_lookback is not None

        self.train_info = {
            "update_lr": update_lr,
            "meta_lr": meta_lr,
            "num_adaptation_steps": num_adaptation_steps,
            "num_iterations": num_iterations,
            "task_batch_size": task_batch_size,
            "task_removal_threshold": task_removal_threshold,
            "task_removal_lookback": task_removal_lookback,
            "task_noise": task_noise,
        }

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
        if self.awareness is not None:
            assert self.encoder is not None
            encoder_opt = optim.Adam(self.encoder.parameters(), encoder_lr)
            if schedule:
                encoder_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    encoder_opt, T_max=num_iterations, eta_min=min_meta_lr
                )

        best_val_score = np.inf

        self.val_results: Dict[str, Dict[str, List]] = {}
        self.train_results: Dict[str, Dict[str, List]] = {}

        removed_labels: List[str] = []
        val_labels = [
            label for label in ALPHABETS if len(VAL_TEST_SPLIT[label]["val"]) > 0
        ]

        # this won't consider removed labels, so might
        # slightly underestimate but should be a good approximation
        iters_per_epoch = len(ALPHABETS) / task_batch_size
        num_epochs = int(num_iterations / iters_per_epoch)

        for iteration_num in tqdm(range(num_epochs)):

            opt.zero_grad()
            meta_train_error = 0.0
            meta_valid_error = 0.0
            meta_train_acc = 0.0
            meta_valid_acc = 0.0

            epoch_labels = [label for label in ALPHABETS if label not in removed_labels]
            shuffle(epoch_labels)

            num_instances_in_batch = 0
            for idx, task_label in enumerate(epoch_labels):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # currently getting a UserWarning from PyTorch here, but it can be ignored
                    learner = self.maml.clone()

                train_batch, eval_batch = self.tasks[task_label].sample_task(
                    mode="train",
                    train_samples_per_way=samples_per_way,
                    eval_samples_per_way=samples_per_way,
                )
                task_info = self.one_hot_from_idx(
                    self.tasks[task_label].idx, task_noise
                )

                if self.awareness is not None:
                    assert self.encoder is not None
                    if self.awareness == "timl":
                        task_encodings = self.encoder(task_info)
                    elif self.awareness == "mmaml":
                        task_encodings = self.encoder(train_batch)
                    learner.module.update_embeddings(task_encodings)

                train_error, train_acc = self.fast_adapt(
                    num_adaptation_steps, train_batch, eval_batch, learner
                )

                if task_label not in self.train_results:
                    self.train_results[task_label] = {
                        "accuracy": [],
                        "loss": [],
                    }

                self.train_results[task_label]["accuracy"].append(train_acc.item())
                self.train_results[task_label]["loss"].append(train_error.item())

                train_error.backward()
                meta_train_error += train_error.item()
                meta_train_acc += train_acc.item()

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

                val_train_batch, val_eval_batch = self.tasks[val_label].sample_task(
                    mode="val",
                    train_samples_per_way=samples_per_way,
                    eval_samples_per_way=samples_per_way,
                )
                val_task_info = self.one_hot_from_idx(
                    self.tasks[val_label].idx, task_noise
                )

                with torch.no_grad():
                    if self.awareness is not None:
                        self.encoder.eval()
                        assert self.encoder is not None
                        if self.awareness == "timl":
                            task_encodings = self.encoder(val_task_info)
                        elif self.awareness == "mmaml":
                            task_encodings = self.encoder(val_train_batch)
                        learner.module.update_embeddings(task_encodings)

                val_error, val_acc = self.fast_adapt(
                    num_adaptation_steps, val_train_batch, val_eval_batch, learner
                )

                if val_label not in self.val_results:
                    self.val_results[val_label] = {
                        "accuracy": [],
                        "loss": [],
                    }

                self.val_results[val_label]["accuracy"].append(val_acc.item())
                self.val_results[val_label]["loss"].append(val_error.item())

                # note that backwards is not called
                meta_valid_error += val_error.item()
                meta_valid_acc += val_acc.item()

            if task_removal_threshold is not None and (
                iteration_num >= task_removal_lookback
            ):
                for label in epoch_labels:
                    mean_acc = np.mean(
                        self.train_results[label]["accuracy"][
                            -cast(int, task_removal_lookback) :
                        ]
                    )
                    if mean_acc >= task_removal_threshold:
                        print(
                            f"Removing {label} - average accuracy over the last {task_removal_lookback}: {mean_acc}"
                        )
                        removed_labels.append(label)

            # Print some metrics
            meta_epoch_size = len(epoch_labels)
            self.results_dict["meta_train"].append((meta_train_error / meta_epoch_size))
            self.results_dict["meta_val"].append((meta_valid_error / len(val_labels)))
            self.results_dict["meta_val_acc"].append(meta_valid_acc / len(val_labels))
            self.results_dict["meta_train_acc"].append(meta_train_acc / meta_epoch_size)

            if iteration_num > 0:
                mean_mt = np.mean(self.results_dict["meta_train"][-checkpoint_every:])
                mean_mv = np.mean(self.results_dict["meta_val"][-checkpoint_every:])
                mean_mvacc = np.mean(
                    self.results_dict["meta_val_acc"][-checkpoint_every:]
                )
                mean_mtacc = np.mean(
                    self.results_dict["meta_train_acc"][-checkpoint_every:]
                )

                if mean_mv <= best_val_score:
                    best_val_score = mean_mv
                    if save_best_val:
                        self.checkpoint(iteration=iteration_num)

                if iteration_num % checkpoint_every == 0:
                    print(
                        f"Meta_train: {round(mean_mt, 3)}, meta_val: {round(mean_mv, 3)}, "
                        f"meta_train_acc: {round(mean_mtacc, 3)}, "
                        f"meta_val_acc: {round(mean_mvacc, 3)}"
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
        if self.awareness is not None:
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

    def one_hot_from_idx(self, idx: int, task_noise: float = 0) -> torch.Tensor:
        one_hot = torch.zeros(len(ALPHABETS), device=self.device)
        one_hot[idx] = 1
        if task_noise > 0:
            one_hot += torch.normal(0, task_noise, one_hot.shape, device=self.device)
        return one_hot

    def test(
        self,
        samples_per_way: int = 1,
        num_adaptation_steps=1,
        update_lr: float = 0.1,
    ):

        # load the most recent model
        self.model.load_state_dict(torch.load(self.model_folder / "state_dict.pth"))
        if self.encoder is not None:
            self.encoder.load_state_dict(
                torch.load(self.model_folder / "encoder_state_dict.pth")
            )

        maml = l2l.algorithms.MAML(self.model, lr=update_lr, first_order=False)

        test_results = {}
        for test_label in ALPHABETS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # currently getting a UserWarning from PyTorch here, but it can be ignored
                learner = maml.clone()

            val_train_batch, val_eval_batch = self.tasks[test_label].sample_task(
                mode="test",
                train_samples_per_way=samples_per_way,
                eval_samples_per_way=-1,
            )
            val_task_info = self.one_hot_from_idx(self.tasks[test_label].idx)

            with torch.no_grad():
                if self.awareness is not None:
                    self.encoder.eval()
                    assert self.encoder is not None
                    if self.awareness == "timl":
                        task_encodings = self.encoder(val_task_info)
                    elif self.awareness == "mmaml":
                        task_encodings = self.encoder(val_train_batch)
                    learner.module.update_embeddings(task_encodings)

            test_error, test_acc = self.fast_adapt(
                num_adaptation_steps, val_train_batch, val_eval_batch, learner
            )

            if test_label not in test_results:
                test_results[test_label] = {}

            test_results[test_label]["accuracy"] = test_acc.item()
            test_results[test_label]["loss"] = test_error.item()
        return test_results


def train_timl_model(
    awareness: str,
    update_lr: float = 0.1,
    meta_lr: float = 0.001,
    min_meta_lr: float = 0.00001,
    num_adaptation_steps: int = 1,
    task_batch_size: int = 4,
    num_iterations: int = 60000,
    save_best_val: bool = True,
    checkpoint_every: int = 20,
    schedule: bool = True,
    seed: int = 42,
) -> Learner:
    model = Learner(awareness, seed)

    model.train(
        update_lr=update_lr,
        meta_lr=meta_lr,
        min_meta_lr=min_meta_lr,
        num_adaptation_steps=num_adaptation_steps,
        task_batch_size=task_batch_size,
        num_iterations=num_iterations,
        save_best_val=save_best_val,
        checkpoint_every=checkpoint_every,
        schedule=schedule,
    )
    return model
