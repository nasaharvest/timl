import torch

from cropharvest.countries import BBox, COUNTRY_SHAPEFILE
from shapely.geometry import Polygon, MultiPolygon


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


def get_largest_country_bbox(country_name: str) -> BBox:

    country = COUNTRY_SHAPEFILE[COUNTRY_SHAPEFILE.NAME_EN == country_name]
    if len(country) != 1:
        raise RuntimeError(f"Unrecognized country {country_name}")
    polygon = country.geometry.iloc[0]
    if isinstance(polygon, Polygon):
        return BBox.polygon_to_bbox(polygon, country_name)
    elif isinstance(polygon, MultiPolygon):
        relevant_polygon = max([x for x in polygon.geoms], key=lambda p: p.area)
        return BBox.polygon_to_bbox(relevant_polygon, country_name)
    raise RuntimeError(f"Unrecognize geometry {type(polygon)}")
