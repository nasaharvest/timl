import torch
import random
from pathlib import Path
from tqdm import tqdm
import tarfile
from urllib.request import urlopen, Request

from typing import List, Optional, Tuple


HISTOGRAM_NAME = "histogram_all_full.npz"
DATASET_VERSION_ID = 5948877
DATASET_URL = f"https://zenodo.org/record/{DATASET_VERSION_ID}"


def download_from_url(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)


def extract_archive(file_path: Path, remove_tar: bool = True):
    with tarfile.open(str(file_path)) as f:
        f.extractall(str(file_path.parent))
    if remove_tar:
        file_path.unlink()


def download_and_extract_archive(root: str, filename: str, targz: bool = True) -> None:
    file_path_str = f"{root}/{filename}"
    file_path = Path(file_path_str)

    if file_path.exists():
        return
    elif targz:
        targz_path_str = f"{file_path_str}.tar.gz"
        targz_path = Path(targz_path_str)
        url = f"{DATASET_URL}/files/{targz_path.name}?download=1"
        if not targz_path.exists():
            download_from_url(url, targz_path_str)
        extract_archive(targz_path)
    else:
        url = f"{DATASET_URL}/files/{file_path.name}?download=1"
        download_from_url(url, file_path_str)


def concatenate_task_info(x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
    r"""
    x should be of shape [n_batches, time, channels]

    task info will be stacked and appended to x so that the returned array
    has shape [n_batches, time, channels + task_info_dims]
    """
    task_info_batches_time = torch.stack([torch.stack([task_info] * x.shape[1])] * x.shape[0])
    return torch.cat([x, task_info_batches_time], dim=-1)


def sample_with_memory(
    indices: List[int], k: int, state: Optional[List[int]] = None
) -> Tuple[List[int], List[int]]:

    if state is None:
        state = []

    indices_to_sample = list(set(indices) - set(state))
    if len(indices_to_sample) < k:
        # restart the state
        state, indices_to_sample = [], indices
    selected_indices = random.sample(indices_to_sample, k)
    state.extend(selected_indices)

    return selected_indices, state
