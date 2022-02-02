import torch


def concatenate_task_info(x: torch.Tensor, task_info: torch.Tensor) -> torch.Tensor:
    r"""
    x should be of shape [n_batches, time, channels]

    task info will be stacked and appended to x so that the returned array
    has shape [n_batches, time, channels + task_info_dims]
    """
    task_info_batches_time = torch.stack(
        [torch.stack([task_info] * x.shape[1])] * x.shape[0]
    )
    return torch.cat([x, task_info_batches_time], dim=-1)
