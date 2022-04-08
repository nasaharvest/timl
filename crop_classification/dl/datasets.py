from dataclasses import dataclass
import warnings

from cropharvest.countries import BBox
from cropharvest.datasets import Task, CropHarvestLabels, CropHarvest
from cropharvest.config import TEST_REGIONS, TEST_DATASETS
from cropharvest import countries

from .config import TEST_CROP_TO_CLASSIFICATION

from typing import List, Optional, Union, Tuple


@dataclass
class TIMLTask(Task):
    classification_label: Optional[str] = None

    def __post_init__(self):
        if self.target_label is None:
            self.target_label = "crop"
            self.classification_label = "crop"

            if self.balance_negative_crops is True:
                warnings.warn(
                    "Balance negative crops not meaningful for the crop vs. non crop tasks"
                )
        else:
            assert (
                self.classification_label is not None
            ), "Higher level classsification must be provided if a target label is provided"

        if self.bounding_box is None:
            self.bounding_box = BBox(
                min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, name="global"
            )


class TIMLCropHarvestLabels(CropHarvestLabels):
    def classes_in_bbox(
        self, bounding_box: BBox, return_classifications: bool = False
    ) -> Union[List[str], List[Tuple[str, str]]]:
        bbox_geojson = self.filter_geojson(self.as_geojson(), bounding_box)
        unique_labels = [x for x in bbox_geojson.label.unique() if x is not None]
        if not return_classifications:
            return unique_labels
        else:
            label_classification_pairs: List[Tuple[str, str]] = []
            for unique_label in unique_labels:
                classification = bbox_geojson[
                    bbox_geojson.label == unique_label
                ].classification_label.iloc[0]
                label_classification_pairs.append((unique_label, classification))
            return label_classification_pairs


class TIMLCropHarvest(CropHarvest):
    @classmethod
    def create_benchmark_datasets(
        cls,
        root,
        balance_negative_crops: bool = True,
        download: bool = True,
    ) -> List:
        r"""
        Create the benchmark datasets.
        :param root: The path to the data, where the training data and labels are (or will be)
            saved
        :param balance_negative_crops: Whether to ensure the crops are equally represented in
            a dataset's negative labels. This is only used for datasets where there is a
            target_label, and that target_label is a crop
        :param download: Whether to download the labels and training data if they don't
            already exist
        :returns: A list of evaluation CropHarvest datasets according to the TEST_REGIONS and
            TEST_DATASETS in the config
        """

        output_datasets: List = []

        for identifier, bbox in TEST_REGIONS.items():
            country, crop, _, _ = identifier.split("_")

            country_bboxes = countries.get_country_bbox(country)
            for country_bbox in country_bboxes:
                task = TIMLTask(
                    country_bbox,
                    crop,
                    balance_negative_crops,
                    f"{country}_{crop}",
                    classification_label=TEST_CROP_TO_CLASSIFICATION[crop].name,
                )
                if task.id not in [x.id for x in output_datasets]:
                    if country_bbox.contains_bbox(bbox):
                        output_datasets.append(cls(root, task, download=download))

        for country, test_dataset in TEST_DATASETS.items():
            # TODO; for now, the only country here is Togo, which
            # only has one bounding box. In the future, it might
            # be nice to confirm its the right index (maybe by checking against
            # some points in the test h5py file?)
            country_bbox = countries.get_country_bbox(country)[0]
            output_datasets.append(
                cls(
                    root,
                    TIMLTask(country_bbox, None, test_identifier=test_dataset),
                    download=download,
                )
            )
        return output_datasets
