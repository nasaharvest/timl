from cropharvest.crops import CropClassifications

from typing import Dict


TEST_CROP_TO_CLASSIFICATION: Dict[str, CropClassifications] = {
    "maize": CropClassifications.cereals,
    "coffee": CropClassifications.beverage_spice,
    "almond": CropClassifications.fruits_nuts,
}

# LSTM model configurations
HIDDEN_VECTOR_SIZE = 128
NUM_CLASSIFICATION_LAYERS = 2
CLASSIFIER_DROPOUT = 0.2
CLASSIFIER_BASE_LAYERS = 1

ENCODER_VECTOR_SIZES = [32, 64, 128]
ENCODER_DROPOUT = 0.2
