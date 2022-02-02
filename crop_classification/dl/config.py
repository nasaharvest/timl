from cropharvest.crops import CropClassifications

from typing import Dict


TEST_CROP_TO_CLASSIFICATION: Dict[str, CropClassifications] = {
    "maize": CropClassifications.cereals,
    "coffee": CropClassifications.beverage_spice,
    "almond": CropClassifications.fruits_nuts,
}
