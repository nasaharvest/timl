from pathlib import Path
import pandas as pd
import torch
import numpy as np
import random
from sklearn.metrics import r2_score

from .utils import HISTOGRAM_NAME

from typing import Dict, List, Tuple, Optional


class CropYieldDataset:
    def __init__(
        self,
        root_dir="data",
        is_test: bool = False,
        min_test_year: int = 2011,
        sampling_buffer: Optional[float] = 0.1,
        model_type: str = "lstm",
        include_year: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        assert model_type in ["lstm", "cnn"]
        if model_type == "cnn":
            assert include_year is False
        self.model_type = model_type
        self.include_year = include_year
        self._data_dir = root_dir
        self.device = device

        with np.load(Path(root_dir) / HISTOGRAM_NAME) as hist:
            histograms = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        lat, lon = locations[:, 0], locations[:, 1]

        lon_r, lat_r = np.radians(lon), np.radians(lat)
        cos_lon = np.cos(lon_r)
        cos_lat = np.cos(lat_r)
        sin_lon = np.sin(lon_r)

        s1 = cos_lon * cos_lat
        s2 = cos_lat * sin_lon
        s3 = np.sin(lat_r)

        # TODO - this is a boolean in the test dataloader
        # but a float in the training dataloader. Everything
        # behaves as expected, thankfully
        self.sampling_buffer = sampling_buffer
        if sampling_buffer is not None:
            self.s1_buffer = np.std(s1) * sampling_buffer
            self.s2_buffer = np.std(s2) * sampling_buffer
            self.s3_buffer = np.std(s3) * sampling_buffer
        else:
            self.s1_buffer = self.s2_buffer = self.s3_buffer = None

        self.root = Path(self._data_dir)

        if is_test:
            mask = years >= min_test_year
        else:
            mask = years < min_test_year

        self._histograms = histograms[mask]

        self.locations = pd.DataFrame(
            data={
                "state_fip": indices[:, 0],
                "county_fip": indices[:, 1],
                "s1": s1,
                "s2": s2,
                "s3": s3,
            }
        ).drop_duplicates()

        self.metadata = pd.DataFrame(
            data={
                "year": years[mask],
                "y": yields[mask],
                "state_fip": indices[:, 0][mask],
                "county_fip": indices[:, 1][mask],
                "s1": s1[mask],
                "s2": s2[mask],
                "s3": s3[mask],
            }
        )

        self._metadata_fields = ["y", "year"]
        self._metadata_array = torch.from_numpy(self.metadata[self._metadata_fields].to_numpy())

        self.state_to_index = {state: idx for idx, state in enumerate(np.unique(indices[:, 0]))}

        if not is_test:
            self.hist_norm_dict, self.task_norm_dict = self._calculate_normalizing_dicts()
        else:
            self.hist_norm_dict: Dict = None
            self.task_norm_dict: Dict = None

    def load_locations(self):
        with np.load(Path(self._data_dir) / "histogram_all_full.npz") as hist:
            locations = hist["output_locations"]
            indices = hist["output_index"]

        lat, lon = locations[:, 0], locations[:, 1]

        lon_r, lat_r = np.radians(lon), np.radians(lat)
        cos_lon = np.cos(lon_r)
        cos_lat = np.cos(lat_r)
        sin_lon = np.sin(lon_r)

        s1 = cos_lon * cos_lat
        s2 = cos_lat * sin_lon
        s3 = np.sin(lat_r)

        self.locations = pd.DataFrame(
            data={
                "state_fip": indices[:, 0],
                "county_fip": indices[:, 1],
                "s1": s1,
                "s2": s2,
                "s3": s3,
                "lat": lat,
                "lon": lon,
            }
        ).drop_duplicates()

    def update_normalizing_dicts(self, hist_norm_dict: Dict, task_norm_dict: Dict) -> None:
        self.hist_norm_dict = hist_norm_dict
        self.task_norm_dict = task_norm_dict

    def _calculate_normalizing_dicts(self) -> Tuple[Dict, Dict, Dict]:
        # calculate a normalizing dict for the a) reshaped histogram and
        # b) the latlons, c) the years

        # first, the histogram
        if self.model_type == "lstm":
            h_t = np.transpose(self._histograms, axes=(0, 2, 1, 3))
            hist_reshaped = np.reshape(
                h_t, (h_t.shape[0], h_t.shape[1], h_t.shape[2] * h_t.shape[3])
            )

            mean = np.mean(hist_reshaped, axis=(0, 1))
            std = np.std(hist_reshaped, axis=(0, 1))
            std[std == 0] = 1
            if self.include_year:
                # we append the year normalizing info to the hist, since they
                # will be normalized together
                mean = np.append(mean, self.metadata["year"].mean())
                std = np.append(std, self.metadata["year"].std())
            hist_normalizing_dict = {"mean": mean, "std": std}
        else:
            hist_normalizing_dict = {
                "mean": np.mean(self._histograms, axis=(0, 1, 2)),
                "std": np.std(self._histograms, axis=(0, 1, 2)),
            }

        # then, the task info
        task_info = []
        for _, row in self.region_tuples.iterrows():
            task_info.append(
                self.metadata_to_task_info(
                    self.get_metadata(row.state_fip, row.county_fip), normalize=False
                )
            )
        task_info_np = np.array(task_info)
        task_normalizing_dict = {
            "mean": np.mean(task_info_np, axis=(0)),
            "std": np.std(task_info_np, axis=(0)),
        }

        return hist_normalizing_dict, task_normalizing_dict

    @property
    def region_tuples(self) -> pd.DataFrame:
        region_df = self.metadata[["state_fip", "county_fip"]]
        return region_df.drop_duplicates()

    def train_val_split(
        self, val_size: float = 0.2, min_k: int = 10, max_val_tasks: Optional[int] = None
    ) -> Tuple[List, List]:
        train_tuples, val_tuples = [], []

        for _, row in self.region_tuples.iterrows():
            task_k = self.task_k(row.state_fip, row.county_fip)
            if task_k >= min_k:
                is_val = random.random() <= val_size
                if max_val_tasks is not None:
                    is_val = is_val & (len(val_tuples) < max_val_tasks)
                if is_val:
                    val_tuples.append((row.state_fip, row.county_fip))
                else:
                    train_tuples.append((row.state_fip, row.county_fip))
        return train_tuples, val_tuples

    def to_latlon(self, state_fip: int, county_fip: int) -> np.ndarray:
        if "lat" not in self.locations:
            self.load_locations()

        try:
            row = self.locations[
                (
                    (self.locations.state_fip == state_fip)
                    & (self.locations.county_fip == county_fip)
                )
            ].iloc[0]
        except IndexError as e:
            print(state_fip, county_fip)
            raise e
        return np.array([row.lat, row.lon])

    def task_k(self, state_fip: int, county_fip: int) -> int:
        return len(self.region_indices(state_fip, county_fip))

    def region_indices(self, state_fip: int, county_fip: int, buffer: bool = True) -> List[int]:
        if not buffer:
            return self.metadata.index[
                ((self.metadata.state_fip == state_fip) & (self.metadata.county_fip == county_fip))
            ].tolist()
        else:
            if not hasattr(self, "locations"):
                self.load_locations()
            single_row = self.locations[
                (
                    (self.locations.state_fip == state_fip)
                    & (self.locations.county_fip == county_fip)
                )
            ].iloc[0]

            s1, s2, s3 = single_row.s1, single_row.s2, single_row.s3

            return self.metadata.index[
                (
                    (self.metadata.state_fip == state_fip)
                    & (self.metadata.s1 <= s1 + self.s1_buffer)
                    & (self.metadata.s1 >= s1 - self.s1_buffer)
                    & (self.metadata.s2 >= s2 - self.s2_buffer)
                    & (self.metadata.s2 <= s2 + self.s2_buffer)
                    & (self.metadata.s3 >= s3 - self.s3_buffer)
                    & (self.metadata.s3 <= s3 + self.s3_buffer)
                )
            ].tolist()

    @property
    def num_bands(self) -> int:
        x, _, metadata = self[0]
        reshaped_x = self._reshape_x(x, metadata)
        if self.model_type == "cnn":
            return reshaped_x.shape[0]
        else:
            return reshaped_x.shape[-1]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        x, _, metadata = self[0]
        return self._reshape_x(x, metadata).shape

    @property
    def task_info_size(self) -> int:
        # hardcoded (ish) for now; its the
        # (spherical coordinates of) latitude and longitude,
        # and one hot encoding for
        # the state id. We make
        # up for the hardcoding by adding an assert
        # in sample()
        return len(self.state_to_index) + 3

    def _reshape_x(self, x: np.ndarray, metadata: pd.Series) -> np.ndarray:
        # input is 9 channels x 32 timestep x 32 bin histogram derived from MODIS
        # reshape to [32 timestep x (32 bin * 9 band)] (this is equivalent to)
        # what happens in the original paper
        if self.model_type == "lstm":
            x = np.transpose(x, (1, 0, 2))
            x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

            # also, add the year info
            if self.include_year:
                year = metadata.year
                year_time = np.expand_dims(np.stack([year] * x.shape[0]), axis=-1)
                x = np.concatenate([x, year_time], axis=-1)
        # note that the cnn doesn't receive the year data
        return x

    def get_metadata(self, state_fip: int, county_fip: int) -> pd.Series:
        try:
            return self.metadata[
                (self.metadata.state_fip == state_fip) & (self.metadata.county_fip == county_fip)
            ].iloc[0]
        except IndexError:
            return self.locations[
                (self.locations.state_fip == state_fip) & (self.locations.county_fip == county_fip)
            ].iloc[0]

    def sample(
        self,
        state_fip: int,
        county_fip: int,
        k: Optional[int] = None,
        task_noise_scale: float = 0,
        buffer: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, List[int], Tuple[torch.Tensor, torch.Tensor]]:
        """
        If buffer is not None, it will overwrite self.sampling_buffer
        """
        assert self.hist_norm_dict is not None
        assert self.task_norm_dict is not None

        if k is not None:
            indices_to_sample = random.sample(
                self.region_indices(
                    state_fip,
                    county_fip,
                    buffer=buffer if buffer is not None else self.sampling_buffer,
                ),
                k=k,
            )
        else:
            indices_to_sample = self.region_indices(
                state_fip,
                county_fip,
                buffer=buffer if buffer is not None else self.sampling_buffer,
            )

        task_info = self.metadata_to_task_info(
            self.get_metadata(state_fip, county_fip), normalize=True
        )
        if task_noise_scale > 0:
            task_info = [t + np.random.normal(loc=0, scale=task_noise_scale) for t in task_info]
        out_x, out_y, years = [], [], []
        for id in indices_to_sample:
            x, y, metadata = self[id]
            out_x.append(self._reshape_x(x, metadata))
            out_y.append(y)
            years.append(metadata.year)
        out_x_np, out_y_np = np.stack(out_x), np.stack(out_y)

        out_x_np = (out_x_np - self.hist_norm_dict["mean"]) / self.hist_norm_dict["std"]
        x_t, y_t, t_t = (
            torch.as_tensor(out_x_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(out_y_np, dtype=torch.float32, device=self.device),
            torch.as_tensor(task_info, dtype=torch.float32, device=self.device),
        )
        assert len(t_t) == self.task_info_size
        return t_t, years, (x_t, y_t)

    def metadata_to_task_info(self, metadata: pd.Series, normalize: bool = False) -> List[float]:
        state_one_hot = [0] * len(self.state_to_index)
        state_one_hot[int(self.state_to_index[metadata.state_fip])] = 1
        task_info = np.array([metadata.s1, metadata.s2, metadata.s3] + state_one_hot)

        if normalize:
            assert self.task_norm_dict is not None
            task_info = (task_info - self.task_norm_dict["mean"]) / self.task_norm_dict["std"]

        return task_info.tolist()

    def __getitem__(self, idx):
        # Any transformations are handled by the SustainBenchSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self._histograms[idx]
        metadata_row = self.metadata.loc[idx]
        y = metadata_row.y
        return x, y, metadata_row

    def crop_yield_metrics(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        assert y_true.shape == y_pred.shape

        error = y_pred - y_true
        RMSE = np.sqrt(np.mean(error ** 2)).item()
        R2 = r2_score(y_true, y_pred)

        return RMSE, R2

    def eval(self, y_pred, y_true):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model.
            - y_true (Tensor): Ground-truth boundary images
            - metadata (Tensor): Metadata
            - binarized: Whether to use binarized prediction
        Output:
            - results (list): List of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        # Overall evaluation
        RMSE, R2 = self.crop_yield_metrics(y_true, y_pred)
        results = [RMSE, R2]
        results_str = f"RMSE: {RMSE:.3f}, R2: {R2:.3f}"
        return results, results_str

    def to(self, device: torch.device) -> None:
        self.device = device
